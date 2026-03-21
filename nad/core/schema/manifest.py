
from __future__ import annotations
from dataclasses import dataclass, asdict, field
from typing import Dict, List, Optional, Any
import json, os, hashlib, logging

logger = logging.getLogger(__name__)

MANIFEST_VERSION = "4.0"

@dataclass
class Manifest:
    """
    Manifest v4.0 - Extended to support token-level metadata

    New in v4.0:
    - token_metadata: Configuration for token-level fields (logprob, conf, entropy, etc.)
    - run_metadata: Configuration for run-level fields (sample_id, slice_id, problem_id)
    - token_row_ptr_sum: Total number of tokens across all runs
    """
    version: str
    dataset_id: str
    model_id: str
    num_runs: int

    # Neuron activation aggregations
    aggregations: List[str]  # e.g., ["max", "sum"]

    # Data types
    dtypes: Dict[str, str]   # e.g., {"keys": "uint32", "w_max": "float16", ...}

    # Sizes
    row_ptr_sum: int  # Total number of unique neuron keys across all runs
    token_row_ptr_sum: Optional[int] = None  # Total number of tokens across all runs

    # Metadata configurations
    token_metadata: Optional[Dict[str, Any]] = field(default_factory=lambda: {
        "enabled": True,
        "fields": ["logprob", "conf", "entropy", "gini", "selfcert"],
        "dtype": "float32",
        "description": "Token-level uncertainty and confidence metrics"
    })

    run_metadata: Optional[Dict[str, Any]] = field(default_factory=lambda: {
        "fields": ["sample_id", "slice_id", "problem_id", "num_tokens"],
        "dtypes": {
            "sample_id": "int32",
            "slice_id": "int32",
            "problem_id": "int16",
            "num_tokens": "int32"
        },
        "description": "Run-level identification and metadata"
    })

    # Hash for integrity checking
    global_key_dict_hash: str = ""

    # File checksums (SHA256)
    files_sha256: Dict[str, str] = field(default_factory=dict)

    # Additional metadata
    created_at: Optional[str] = None
    endianness: str = "little"

    @staticmethod
    def load(path: str) -> "Manifest":
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return Manifest(**data)

    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(asdict(self), f, indent=2, sort_keys=True)

    def verify(self, cache_root: str) -> bool:
        """
        Verify integrity of cache files against stored SHA256 checksums.

        Returns:
            True if all files match their checksums, False otherwise
        """
        for rel_path, expected_sha in self.files_sha256.items():
            full_path = os.path.join(cache_root, rel_path)
            if not os.path.exists(full_path):
                logger.error(f"Missing file: {rel_path}")
                return False

            actual_sha = sha256_file(full_path)
            if actual_sha != expected_sha:
                logger.error(f"Checksum mismatch for {rel_path}")
                logger.error(f"  Expected: {expected_sha}")
                logger.error(f"  Actual:   {actual_sha}")
                return False

        return True

def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1<<20), b""):
            h.update(chunk)
    return h.hexdigest()
