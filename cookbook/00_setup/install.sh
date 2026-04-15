#!/bin/bash
# install.sh — Install all packages required by NAD Next
#
# Usage:
#   cd /path/to/NAD_Next
#   bash cookbook/00_setup/install.sh

set -euo pipefail

GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

echo -e "${BLUE}=== NAD Next — Dependency Installer ===${NC}"
echo ""

# ── helper: install only if missing ──────────────────────────────────────────
install_if_missing() {
    local pkg="$1"       # importable module name  (e.g. "flask")
    local spec="$2"      # pip install spec        (e.g. "Flask>=2.0.0")
    if python3 -c "import ${pkg}" 2>/dev/null; then
        echo -e "  ${GREEN}✓${NC} ${spec}  (already installed)"
    else
        echo -e "  ${YELLOW}↓${NC} installing ${spec} ..."
        pip install "${spec}" -q
        echo -e "  ${GREEN}✓${NC} ${spec}  installed"
    fi
}

# ── Required ──────────────────────────────────────────────────────────────────
echo -e "${BLUE}[1/3] Required packages${NC}"
install_if_missing "numpy"     "numpy>=1.20.0"
install_if_missing "pyroaring" "pyroaring>=0.4.5"
echo ""

# ── Web UI & Visualization ────────────────────────────────────────────────────
echo -e "${BLUE}[2/3] Web UI & visualization packages${NC}"
install_if_missing "flask"   "Flask>=2.0.0"
install_if_missing "plotly"  "plotly>=5.0.0"
install_if_missing "hmmlearn" "hmmlearn"
echo ""

# ── Tokenizer ─────────────────────────────────────────────────────────────────
echo -e "${BLUE}[3/4] Tokenizer packages${NC}"
install_if_missing "tokenizers"   "tokenizers"
install_if_missing "transformers" "transformers"

# ── Utilities ─────────────────────────────────────────────────────────────────
echo -e "${BLUE}[4/4] Utility packages${NC}"
install_if_missing "psutil" "psutil>=5.8.0"
install_if_missing "tqdm"   "tqdm>=4.50.0"
install_if_missing "torch"  "torch>=2.5.0"
echo ""

echo -e "${GREEN}All packages installed. Run verify.sh to confirm.${NC}"
