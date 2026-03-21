
import os
from dataclasses import dataclass

# NAD v4.1: Row-CSR Bank + Window Cache + position semantics
@dataclass(frozen=True)
class CachePaths:
    """
    Cache paths for NAD v4.1 with Row-CSR Bank and Window Cache support.

    Directory structure:
        cache_root/
        ├── manifest.json
        ├── base/              # Neuron activation data (CSR format)
        │   ├── row_ptr.int64
        │   ├── keys.uint32
        │   ├── w_max.float16
        │   └── w_sum.float16
        ├── index/             # Sorting and prefix indices
        │   ├── perm_max.int32
        │   ├── perm_sum.int32
        │   ├── prefix_max.float16
        │   └── prefix_sum.float16
        ├── token_data/        # Token-level metadata (NEW in v4.0)
        │   ├── token_row_ptr.int64
        │   ├── token_ids.int32
        │   ├── tok_conf.float32
        │   ├── tok_neg_entropy.float32
        │   ├── tok_gini.float32
        │   ├── tok_selfcert.float32
        │   └── tok_logprob.float32
        └── run_metadata/      # Run-level metadata (NEW in v4.0)
            ├── sample_ids.int32
            ├── slice_ids.int32
            ├── problem_ids.int16
            └── num_tokens.int32
    """
    root: str

    @property
    def manifest(self) -> str:
        return os.path.join(self.root, "manifest.json")

    # ===== Neuron Activation Data (Base CSR) =====
    @property
    def base_dir(self) -> str:
        return os.path.join(self.root, "base")

    @property
    def row_ptr(self) -> str:
        return os.path.join(self.base_dir, "row_ptr.int64")

    @property
    def keys(self) -> str:
        return os.path.join(self.base_dir, "keys.uint32")

    @property
    def w_max(self) -> str:
        return os.path.join(self.base_dir, "w_max.float16")

    @property
    def w_sum(self) -> str:
        return os.path.join(self.base_dir, "w_sum.float16")

    # ===== Index (Sorting and Prefix) =====
    @property
    def index_dir(self) -> str:
        return os.path.join(self.root, "index")

    @property
    def perm_max(self) -> str:
        return os.path.join(self.index_dir, "perm_max.int32")

    @property
    def perm_sum(self) -> str:
        return os.path.join(self.index_dir, "perm_sum.int32")

    @property
    def prefix_max(self) -> str:
        return os.path.join(self.index_dir, "prefix_max.float16")

    @property
    def prefix_sum(self) -> str:
        return os.path.join(self.index_dir, "prefix_sum.float16")

    # ===== Token-Level Metadata (NEW in v4.0) =====
    @property
    def token_dir(self) -> str:
        return os.path.join(self.root, "token_data")

    @property
    def token_row_ptr(self) -> str:
        return os.path.join(self.token_dir, "token_row_ptr.int64")

    @property
    def token_ids(self) -> str:
        return os.path.join(self.token_dir, "token_ids.int32")

    @property
    def tok_logprob(self) -> str:
        return os.path.join(self.token_dir, "tok_logprob.float32")

    @property
    def tok_conf(self) -> str:
        return os.path.join(self.token_dir, "tok_conf.float32")

    @property
    def tok_neg_entropy(self) -> str:
        """Token negative entropy (primary, aligned with format.md)"""
        return os.path.join(self.token_dir, "tok_neg_entropy.float32")

    @property
    def tok_entropy(self) -> str:
        """Token entropy (legacy, for backward compatibility)"""
        return os.path.join(self.token_dir, "tok_entropy.float32")

    @property
    def tok_gini(self) -> str:
        return os.path.join(self.token_dir, "tok_gini.float32")

    @property
    def tok_selfcert(self) -> str:
        return os.path.join(self.token_dir, "tok_selfcert.float32")

    # ===== Run-Level Metadata (NEW in v4.0) =====
    @property
    def metadata_dir(self) -> str:
        return os.path.join(self.root, "run_metadata")

    @property
    def sample_ids(self) -> str:
        return os.path.join(self.metadata_dir, "sample_ids.int32")

    @property
    def slice_ids(self) -> str:
        return os.path.join(self.metadata_dir, "slice_ids.int32")

    @property
    def problem_ids(self) -> str:
        return os.path.join(self.metadata_dir, "problem_ids.int16")

    @property
    def num_tokens(self) -> str:
        return os.path.join(self.metadata_dir, "num_tokens.int32")

    # ===== Row-CSR Bank (optional, NEW in v4.1) =====
    @property
    def rows_dir(self) -> str:
        """Row-level CSR storage for arbitrary window queries"""
        return os.path.join(self.root, "rows")

    @property
    def rows_sample_row_ptr(self) -> str:
        """Sample-level row pointer (length: num_samples+1)"""
        return os.path.join(self.rows_dir, "sample_row_ptr.int64")

    @property
    def rows_row_ptr(self) -> str:
        """Row-level CSR pointer (length: total_rows+1)"""
        return os.path.join(self.rows_dir, "row_ptr.int64")

    @property
    def rows_keys(self) -> str:
        """Row-level keys (unique per row)"""
        return os.path.join(self.rows_dir, "keys.uint32")

    @property
    def rows_w_max(self) -> str:
        """Row-level max weights"""
        return os.path.join(self.rows_dir, "w_max.float16")

    @property
    def rows_w_sum(self) -> str:
        """Row-level sum weights"""
        return os.path.join(self.rows_dir, "w_sum.float16")

    @property
    def rows_slice_ids(self) -> str:
        """Slice ID for each row"""
        return os.path.join(self.rows_dir, "slice_ids.int32")

    @property
    def rows_token_row_ptr(self) -> str:
        """
        Global cumulative token pointer across all rows.

        Note: Readers compute per-sample local coordinates by subtracting the
        base value at the sample's first row. This guarantees monotonic pointers
        while avoiding boundary ambiguity between adjacent samples.
        """
        return os.path.join(self.rows_dir, "token_row_ptr.int64")

    # ===== Window Cache (lazy, optional, NEW in v4.1) =====
    def window_cache_dir(self, pos_lo: int, pos_hi: int) -> str:
        """Window cache directory for position range [pos_lo, pos_hi)"""
        return os.path.join(self.root, f"window_cache/pos_{pos_lo}_{pos_hi}")

    def window_row_ptr(self, pos_lo: int, pos_hi: int) -> str:
        """Window cache row pointer"""
        return os.path.join(self.window_cache_dir(pos_lo, pos_hi), "row_ptr.int64")

    def window_keys(self, pos_lo: int, pos_hi: int) -> str:
        """Window cache keys"""
        return os.path.join(self.window_cache_dir(pos_lo, pos_hi), "keys.uint32")

    def window_w(self, pos_lo: int, pos_hi: int, agg: str) -> str:
        """Window cache weights (agg='max' or 'sum')"""
        return os.path.join(self.window_cache_dir(pos_lo, pos_hi), f"w_{agg}.float16")
