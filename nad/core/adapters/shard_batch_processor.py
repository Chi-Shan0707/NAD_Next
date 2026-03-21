"""
Optimized shard processor that processes entire shards at once.

This is much faster than the iter_runs() approach which re-loads shards repeatedly.
"""

from __future__ import annotations
import os
import json
import logging
import numpy as np
from glob import glob
from typing import Dict, Tuple, List, Optional, Iterator
from dataclasses import dataclass
from tqdm import tqdm

logger = logging.getLogger(__name__)


@dataclass
class RunDataBatch:
    """Batch of run data from a single shard."""
    run_ids: List[int]
    sample_ids: np.ndarray
    slice_ids: np.ndarray
    problem_ids: np.ndarray

    # Per-run neuron data (list of arrays)
    neuron_keys_list: List[np.ndarray]
    neuron_scores_list: List[np.ndarray]

    # Per-run token data (list of dicts)
    token_data_list: Optional[List[Dict[str, np.ndarray]]] = None
    num_tokens_list: Optional[List[int]] = None


class ShardBatchProcessor:
    """
    Process shards in batches for maximum efficiency.

    This processor loads each shard only ONCE and extracts all runs from it.
    For 883K runs across 60 shards, this is ~15K times faster than individual run extraction.
    """

    def __init__(self, shard_dir: str, run_index_dir: str, include_token_metadata: bool = True):
        self.shard_dir = shard_dir
        self.include_token_metadata = include_token_metadata

        # Load run index
        with open(os.path.join(run_index_dir, "run_index.json"), 'r') as f:
            data = json.load(f)
            # Convert "sample_slice" -> run_id mapping to (sample, slice) -> run_id
            self.run_index = {}
            for key, run_id in data["mapping"].items():
                sample_id, slice_id = map(int, key.split('_'))
                self.run_index[(sample_id, slice_id)] = run_id

        with open(os.path.join(run_index_dir, "run_metadata.json"), 'r') as f:
            self.run_metadata = {int(k): v for k, v in json.load(f).items()}

        self.num_runs = len(self.run_metadata)
        self.shard_files = sorted(glob(os.path.join(shard_dir, "*.npz")))

        logger.info(f"Initialized ShardBatchProcessor: {len(self.shard_files)} shards, {self.num_runs:,} total runs")

    def iter_shards(self) -> Iterator[RunDataBatch]:
        """
        Iterate over shards, yielding batches of runs.

        This is the key optimization: each shard is loaded exactly once.
        """
        for shard_path in tqdm(self.shard_files, desc="Processing shards"):
            try:
                data = np.load(shard_path, mmap_mode='r')

                if 'idx_samples' not in data or 'slice_ids' not in data:
                    continue

                # Get all unique (sample_id, slice_id) pairs in this shard
                idx_samples = data['idx_samples'][:]
                slice_ids_arr = data['slice_ids'][:]

                # Find which runs are in this shard
                run_ids = []
                sample_ids_list = []
                slice_ids_list = []
                indices_list = []  # Which rows in the shard belong to each run

                seen_keys = set()
                for i, (sample_id, slice_id) in enumerate(zip(idx_samples, slice_ids_arr)):
                    key = (int(sample_id), int(slice_id))

                    # Skip duplicates within the shard
                    if key in seen_keys:
                        continue

                    # Check if this (sample_id, slice_id) is a known run
                    if key in self.run_index:
                        run_id = self.run_index[key]
                        run_ids.append(run_id)
                        sample_ids_list.append(sample_id)
                        slice_ids_list.append(slice_id)

                        # Find all indices for this run
                        mask = (idx_samples == sample_id) & (slice_ids_arr == slice_id)
                        indices = np.where(mask)[0]
                        indices_list.append(indices)

                        seen_keys.add(key)

                if not run_ids:
                    continue

                # Extract neuron data for all runs in this shard
                neuron_keys_list = []
                neuron_scores_list = []

                for indices in indices_list:
                    keys, scores = self._extract_neuron_data(data, indices)
                    neuron_keys_list.append(keys)
                    neuron_scores_list.append(scores)

                # Extract token metadata if requested
                token_data_list = None
                num_tokens_list = None

                if self.include_token_metadata and 'token_row_ptr' in data:
                    token_data_list = []
                    num_tokens_list = []

                    for indices in indices_list:
                        token_data, num_tokens = self._extract_token_data(data, indices)
                        token_data_list.append(token_data)
                        num_tokens_list.append(num_tokens)

                # Get problem IDs
                problem_ids = np.array([self.run_metadata[rid]["problem_id"] for rid in run_ids], dtype=np.int16)

                yield RunDataBatch(
                    run_ids=run_ids,
                    sample_ids=np.array(sample_ids_list, dtype=np.int32),
                    slice_ids=np.array(slice_ids_list, dtype=np.int32),
                    problem_ids=problem_ids,
                    neuron_keys_list=neuron_keys_list,
                    neuron_scores_list=neuron_scores_list,
                    token_data_list=token_data_list,
                    num_tokens_list=num_tokens_list
                )

            except Exception as e:
                logger.error(f"Error processing {shard_path}: {e}")
                continue

    def _extract_neuron_data(self, shard_data, indices: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Extract neuron activation data for given indices."""
        layers = shard_data['layers'][indices]
        neurons = shard_data['neurons'][indices]
        scores = shard_data['scores'][indices]

        # Valid mask
        valid_mask = (layers >= 0) & (neurons >= 0) & (neurons != 65535)

        # Flatten and filter
        layers_flat = layers[valid_mask].astype(np.uint32)
        neurons_flat = neurons[valid_mask].astype(np.uint32)
        scores_flat = scores[valid_mask].astype(np.float32)

        # Pack keys
        keys = (layers_flat << 16) | neurons_flat

        return keys, scores_flat

    def _extract_token_data(self, shard_data, indices: np.ndarray) -> Tuple[Optional[Dict[str, np.ndarray]], int]:
        """Extract token metadata for given indices."""
        if len(indices) == 0:
            return {}, 0

        # For multiple indices, concatenate token data
        all_tokens = {}
        total_tokens = 0

        for idx in indices:
            idx = int(idx)
            start = int(shard_data['token_row_ptr'][idx])
            end = int(shard_data['token_row_ptr'][idx + 1])

            for field in ['tok_logprob', 'tok_conf', 'tok_neg_entropy', 'tok_gini', 'tok_selfcert']:
                if field not in shard_data:
                    continue

                values = shard_data[field][start:end].astype(np.float32)

                # Special handling for tok_neg_entropy
                if field == 'tok_neg_entropy':
                    field_name = 'entropy'
                    values = -values
                else:
                    field_name = field.replace('tok_', '')

                if field_name not in all_tokens:
                    all_tokens[field_name] = []
                all_tokens[field_name].append(values)

            total_tokens += (end - start)

        # Concatenate all token arrays
        token_data = {k: np.concatenate(v) if v else np.array([], dtype=np.float32)
                     for k, v in all_tokens.items()}

        return token_data, total_tokens
