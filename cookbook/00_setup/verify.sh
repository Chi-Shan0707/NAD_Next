#!/bin/bash
# verify.sh — Check that all packages and the MUI_HUB symlink are ready
#
# Usage:
#   cd /path/to/NAD_Next
#   bash cookbook/00_setup/verify.sh
#
# Exit code: 0 = all checks passed, 1 = one or more checks failed

set -uo pipefail

GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

PASS=0
FAIL=0

ok()   { echo -e "  ${GREEN}✓${NC} $1"; PASS=$((PASS + 1)); }
fail() { echo -e "  ${RED}✗${NC} $1"; FAIL=$((FAIL + 1)); }
warn() { echo -e "  ${YELLOW}⚠${NC} $1"; }

# ── 1. Python version ─────────────────────────────────────────────────────────
echo -e "${BLUE}[1/3] Python version${NC}"
PY_VER=$(python3 -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
PY_MAJOR=$(echo "$PY_VER" | cut -d. -f1)
PY_MINOR=$(echo "$PY_VER" | cut -d. -f2)
if [ "$PY_MAJOR" -ge 3 ] && [ "$PY_MINOR" -ge 9 ]; then
    ok "Python ${PY_VER} (>= 3.9)"
else
    fail "Python ${PY_VER} — requires >= 3.9"
fi
echo ""

# ── 2. Package checks ─────────────────────────────────────────────────────────
echo -e "${BLUE}[2/3] Python packages${NC}"

check_pkg() {
    local module="$1"
    local label="$2"
    local version
    version=$(python3 -c "import ${module}; print(${module}.__version__)" 2>/dev/null || echo "")
    if [ -n "$version" ]; then
        ok "${label}  (${version})"
    else
        fail "${label}  — not found  →  run install.sh"
    fi
}

# Required
check_pkg "numpy"     "numpy     [required]"
check_pkg "pyroaring" "pyroaring [required]"

# Web UI & Visualization
check_pkg "flask"     "flask     [web UI]"
check_pkg "plotly"    "plotly    [visualization]"
check_pkg "hmmlearn"  "hmmlearn  [visualization server]"

# Tokenizer
check_pkg "tokenizers"   "tokenizers   [token decoding]"
check_pkg "transformers" "transformers [token decoding]"

# Utilities
check_pkg "psutil"    "psutil    [profiling]"
check_pkg "tqdm"      "tqdm      [progress bars]"

echo ""

# ── 3. MUI_HUB symlink ────────────────────────────────────────────────────────
echo -e "${BLUE}[3/3] MUI_HUB symlink${NC}"

SCRIPT_DIR="$(cd "$(dirname "$0")/../.." && pwd)"   # NAD_Next root
SYMLINK="${SCRIPT_DIR}/MUI_HUB"
TARGET="/home/jovyan/public-ro/MUI_HUB"

if [ -L "$SYMLINK" ]; then
    RESOLVED=$(readlink -f "$SYMLINK")
    if [ "$RESOLVED" = "$TARGET" ]; then
        ok "MUI_HUB → ${TARGET}"
    else
        warn "MUI_HUB symlink exists but points to ${RESOLVED} (expected ${TARGET})"
    fi
elif [ -d "$SYMLINK" ]; then
    warn "MUI_HUB is a real directory, not a symlink — that is fine if it contains the caches"
else
    fail "MUI_HUB symlink missing — create it with:"
    echo ""
    echo "      ln -s ${TARGET} ${SYMLINK}"
    echo ""
fi

if [ -L "$SYMLINK" ] || [ -d "$SYMLINK" ]; then
    CACHE_ROOT="${SYMLINK}/cache"
    if [ -d "$CACHE_ROOT" ]; then
        N=$(find "$CACHE_ROOT" -mindepth 3 -maxdepth 3 -type d | wc -l)
        ok "cache root accessible  (${N} cache directories found)"
    else
        fail "cache root not accessible: ${CACHE_ROOT}"
    fi
fi

echo ""

# ── Summary ───────────────────────────────────────────────────────────────────
echo "────────────────────────────────────"
if [ "$FAIL" -eq 0 ]; then
    echo -e "${GREEN}All ${PASS} checks passed.${NC} You are ready to proceed."
    exit 0
else
    echo -e "${RED}${FAIL} check(s) failed${NC}, ${PASS} passed."
    echo "Fix the issues above, then re-run this script."
    exit 1
fi
