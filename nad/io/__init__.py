# NAD Next - IO subpackage
# Exposes high-level cache loader and helpers for visualization tools
from .loader import NadNextLoader, detect_nad_next_cache
from .index import ensure_loader, load_nad_next_index
from .viz_catalog import build_problem_catalog
