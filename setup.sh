#!/usr/bin/env bash
# =============================================================================
# setup.sh — Environment setup for ego_exo_analysis_26
#
# Usage:
#   ./setup.sh [OPTIONS]
#
# Options:
#   -e, --env-name NAME    Name of the virtual environment folder
#                          (default: .venv)
#   -p, --prompt           Prompt before each step (default: auto-yes)
#   -h, --help             Show this help message and exit
#
# Examples:
#   ./setup.sh                          # defaults: .venv, no prompts
#   ./setup.sh --env-name myenv         # custom venv name
#   ./setup.sh --prompt                 # pause before each step
#   ./setup.sh -e myenv -p              # both flags together
# =============================================================================

set -euo pipefail

# -----------------------------------------------------------------------------
# Defaults
# -----------------------------------------------------------------------------
VENV_NAME=".venv"
PROMPT_MODE=false
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# -----------------------------------------------------------------------------
# Colors
# -----------------------------------------------------------------------------
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
BOLD='\033[1m'
RESET='\033[0m'

# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------
info()    { echo -e "${CYAN}[INFO]${RESET} $*"; }
success() { echo -e "${GREEN}[OK]${RESET}   $*"; }
warn()    { echo -e "${YELLOW}[WARN]${RESET} $*"; }
error()   { echo -e "${RED}[ERROR]${RESET} $*" >&2; }
header()  { echo -e "\n${BOLD}======================================================${RESET}"; \
             echo -e "${BOLD}  $*${RESET}"; \
             echo -e "${BOLD}======================================================${RESET}"; }

# Ask user before proceeding with a step when --prompt is active
maybe_prompt() {
    local msg="$1"
    if [ "$PROMPT_MODE" = true ]; then
        echo -e "\n${YELLOW}>> Next step: ${msg}${RESET}"
        read -rp "   Proceed? [Y/n] " ans
        case "$ans" in
            [Nn]*) info "Skipping: ${msg}"; return 1 ;;
            *)     return 0 ;;
        esac
    fi
    return 0
}

show_help() {
    grep '^#' "$0" | grep -v '#!/' | sed 's/^# \{0,1\}//'
    exit 0
}

# -----------------------------------------------------------------------------
# Argument parsing
# -----------------------------------------------------------------------------
while [[ $# -gt 0 ]]; do
    case "$1" in
        -e|--env-name)
            VENV_NAME="$2"
            shift 2
            ;;
        -p|--prompt)
            PROMPT_MODE=true
            shift
            ;;
        -h|--help)
            show_help
            ;;
        *)
            error "Unknown option: $1"
            echo "Run './setup.sh --help' for usage."
            exit 1
            ;;
    esac
done

VENV_PATH="${REPO_ROOT}/${VENV_NAME}"
PIP="${VENV_PATH}/bin/pip"
PYTHON="${VENV_PATH}/bin/python3"

# -----------------------------------------------------------------------------
# Banner
# -----------------------------------------------------------------------------
header "ego_exo_analysis_26 — Setup Script"
info "Repository root : ${REPO_ROOT}"
info "Virtual env     : ${VENV_PATH}"
info "Prompt mode     : ${PROMPT_MODE}"
echo ""

# -----------------------------------------------------------------------------
# Step 0: Sudo password up front
# -----------------------------------------------------------------------------
header "Step 0: Sudo credentials"
info "Some steps require sudo (apt-get). Requesting credentials now so"
info "you are not interrupted later."
sudo -v
# Keep sudo alive in the background for the duration of the script
while true; do sudo -n true; sleep 50; kill -0 "$$" || exit; done 2>/dev/null &
SUDO_KEEPALIVE_PID=$!
trap 'kill $SUDO_KEEPALIVE_PID 2>/dev/null' EXIT
success "Sudo credentials cached."

# -----------------------------------------------------------------------------
# Step 1: System packages
# -----------------------------------------------------------------------------
header "Step 1: System packages"
if maybe_prompt "Install system packages via apt-get (libgl1-mesa-glx, libglib2.0-0)"; then
    info "Updating apt package list..."
    sudo apt-get update -qq
    info "Installing system packages..."
    sudo apt-get install -y \
        libgl1-mesa-glx \
        libglib2.0-0 \
        python3-venv \
        python3-pip
    success "System packages installed."
fi

# -----------------------------------------------------------------------------
# Step 2: Virtual environment
# -----------------------------------------------------------------------------
header "Step 2: Virtual environment"
if [ -d "$VENV_PATH" ]; then
    warn "Virtual environment already exists at: ${VENV_PATH}"
    if [ "$PROMPT_MODE" = true ]; then
        read -rp "   Re-use existing venv? [Y/n] " ans
        case "$ans" in
            [Nn]*)
                info "Removing existing venv and creating fresh one..."
                rm -rf "$VENV_PATH"
                python3 -m venv "$VENV_PATH"
                success "Virtual environment created at: ${VENV_PATH}"
                ;;
            *)
                info "Re-using existing virtual environment."
                ;;
        esac
    else
        info "Re-using existing virtual environment."
    fi
else
    if maybe_prompt "Create virtual environment at '${VENV_PATH}'"; then
        python3 -m venv "$VENV_PATH"
        success "Virtual environment created at: ${VENV_PATH}"
    fi
fi

# Activate for the rest of this script
# shellcheck source=/dev/null
source "${VENV_PATH}/bin/activate"
info "Virtual environment activated."

# -----------------------------------------------------------------------------
# Step 3: Upgrade pip
# -----------------------------------------------------------------------------
header "Step 3: Upgrade pip"
if maybe_prompt "Upgrade pip inside the virtual environment"; then
    "$PIP" install --upgrade pip -q
    success "pip upgraded."
fi

# -----------------------------------------------------------------------------
# Step 4: Core Python packages
# -----------------------------------------------------------------------------
header "Step 4: Core Python packages"
if maybe_prompt "Install core packages (torch, torchvision, numpy, scipy, etc.)"; then
    info "Installing PyTorch (CUDA 11.8 build)..."
    "$PIP" install torch torchvision \
        --index-url https://download.pytorch.org/whl/cu118 -q
    info "Installing remaining core packages..."
    "$PIP" install \
        numpy \
        scipy \
        pillow \
        tqdm \
        wandb \
        transformers \
        -q
    success "Core packages installed."
fi

# -----------------------------------------------------------------------------
# Step 5: MetaWorld
# -----------------------------------------------------------------------------
header "Step 5: MetaWorld"
if maybe_prompt "Install MetaWorld simulation environment"; then
    "$PIP" install metaworld -q
    success "MetaWorld installed."
fi

# -----------------------------------------------------------------------------
# Step 6: TensorFlow + TFRecord
# -----------------------------------------------------------------------------
header "Step 6: TensorFlow and TFRecord reader"
if maybe_prompt "Install TensorFlow, tensorflow-datasets, and tfrecord"; then
    "$PIP" install tensorflow tensorflow-datasets -q
    info "Resolving protobuf version for TensorFlow 2.21..."
    "$PIP" install "protobuf>=6.31.1,<8.0.0" -q
    "$PIP" install tfrecord -q
    success "TensorFlow stack installed."
fi

# -----------------------------------------------------------------------------
# Step 7: Ego4D CLI (for Ego-Exo4D download)
# -----------------------------------------------------------------------------
header "Step 7: Ego4D / egoexo CLI"
if maybe_prompt "Install ego4d package (provides the egoexo download CLI)"; then
    "$PIP" install ego4d -q
    success "ego4d CLI installed."
fi

# -----------------------------------------------------------------------------
# Step 8: R3M submodule
# -----------------------------------------------------------------------------
header "Step 8: Install R3M"
R3M_DIR="${REPO_ROOT}/r3m"
if [ -f "${R3M_DIR}/setup.py" ] || [ -f "${R3M_DIR}/pyproject.toml" ]; then
    if maybe_prompt "Install R3M from submodule (pip install -e ./r3m)"; then
        "$PIP" install -e "$R3M_DIR" -q
        success "R3M installed."
    fi
else
    warn "R3M submodule not found at ${R3M_DIR}."
    warn "Make sure you cloned with --recurse-submodules, or run:"
    warn "  git submodule update --init --recursive"
fi

# -----------------------------------------------------------------------------
# Step 9: VIP submodule
# -----------------------------------------------------------------------------
header "Step 9: Install VIP"
VIP_DIR="${REPO_ROOT}/vip"
if [ -f "${VIP_DIR}/setup.py" ] || [ -f "${VIP_DIR}/pyproject.toml" ]; then
    if maybe_prompt "Install VIP from submodule (pip install -e ./vip)"; then
        "$PIP" install -e "$VIP_DIR" -q
        success "VIP installed."
    fi
else
    warn "VIP submodule not found at ${VIP_DIR}."
    warn "Make sure you cloned with --recurse-submodules, or run:"
    warn "  git submodule update --init --recursive"
fi

# -----------------------------------------------------------------------------
# Step 10: Hydra / Python 3.12 compatibility patch
# -----------------------------------------------------------------------------
header "Step 10: Hydra Python 3.12 compatibility patch"
info "VIP's hydra-core 1.3.2 dependency has mutable dataclass defaults that"
info "Python 3.12 rejects. This step patches the affected lines automatically."
echo ""

HYDRA_CONF=$(find "${VENV_PATH}" -path "*/hydra/conf/__init__.py" 2>/dev/null | head -1)

if [ -z "$HYDRA_CONF" ]; then
    warn "Could not locate hydra/conf/__init__.py — skipping patch."
    warn "If you hit a 'mutable default' ValueError when importing VIP,"
    warn "see the Known Compatibility Issues section in README.md."
else
    info "Found: ${HYDRA_CONF}"

    BACKUP="${HYDRA_CONF}.backup_setup_sh"
    if [ -f "$BACKUP" ]; then
        warn "Backup already exists at: ${BACKUP}"
        warn "This may mean the patch was already applied."
    fi

    if maybe_prompt "Patch ${HYDRA_CONF} for Python 3.12 compatibility (a backup will be saved)"; then

        # Make a backup
        cp "$HYDRA_CONF" "$BACKUP"
        success "Backup saved to: ${BACKUP}"

        PATCH_COUNT=0
        ORIGINAL_CONTENT=$(cat "$HYDRA_CONF")

        # We use Python to do the patching safely — it handles
        # indentation and multiline context better than sed.
        "$PYTHON" - "$HYDRA_CONF" <<'PYEOF'
import sys
import re

filepath = sys.argv[1]
with open(filepath, "r") as f:
    content = f.read()

original = content
changes = []

# Ensure 'field' is imported from dataclasses
if "from dataclasses import" in content:
    # Add 'field' to existing import if not already there
    def add_field_to_import(m):
        imports = m.group(1)
        if "field" not in imports.split(","):
            imports = imports.rstrip() + ", field"
            changes.append("Added 'field' to dataclasses import")
        return f"from dataclasses import {imports}"
    content = re.sub(r"from dataclasses import ([^\n]+)", add_field_to_import, content)

# Pattern: SomeClass = SomeClass()   ->  SomeClass = field(default_factory=SomeClass)
# Matches lines like:
#   override_dirname: OverrideDirname = OverrideDirname()
#   config: JobConfig = JobConfig()
pattern = re.compile(
    r"^( {4,})"                          # leading indent (at least 4 spaces)
    r"(\w+)"                             # field name
    r":\s*"                              # colon + optional space
    r"([\w.]+)"                          # type annotation
    r"\s*=\s*"                           # equals sign
    r"([\w.]+)\(\)"                      # default = ClassName()
    r"\s*$",                             # end of line
    re.MULTILINE
)

def replace_mutable_default(m):
    indent    = m.group(1)
    fieldname = m.group(2)
    typehint  = m.group(3)
    factory   = m.group(4)
    # Only replace if type and factory are the same class (mutable dataclass default)
    # Skip primitives like str, int, bool, list, dict
    primitives = {"str", "int", "float", "bool", "list", "dict", "tuple", "set"}
    if factory in primitives:
        return m.group(0)
    new_line = f"{indent}{fieldname}: {typehint} = field(default_factory={factory})"
    changes.append(f"  Line patched: '{fieldname}: {typehint} = {factory}()'"
                   f" -> 'field(default_factory={factory})'")
    return new_line

content = pattern.sub(replace_mutable_default, content)

if content != original:
    with open(filepath, "w") as f:
        f.write(content)
    print(f"PATCHED|{len(changes)} change(s) applied:")
    for c in changes:
        print(f"CHANGE|{c}")
else:
    print("NOCHANGE|No mutable defaults found — file may already be patched.")
PYEOF

        # Read output and display nicely
        PATCH_STATUS=$("$PYTHON" - "$HYDRA_CONF" 2>/dev/null || true)
        # Re-run to capture output properly
        PATCH_OUTPUT=$("$PYTHON" - "$HYDRA_CONF" <<'PYEOF2'
import sys, re
filepath = sys.argv[1]
with open(filepath, "r") as f:
    content = f.read()
# Just check if field( appears — indicates already patched or we just patched it
count = content.count("field(default_factory=")
print(f"field(default_factory= occurrences found: {count}")
PYEOF2
        )

        success "Hydra patch applied. ${PATCH_OUTPUT}"
        echo ""
        warn "IMPORTANT: This patch is automated but may not catch every"
        warn "mutable default in hydra's codebase. If you still hit a"
        warn "'mutable default' ValueError when importing VIP, refer to"
        warn "the Known Compatibility Issues section in README.md for"
        warn "manual fix instructions."
        warn ""
        warn "Your original file is backed up at:"
        warn "  ${BACKUP}"
    fi
fi

# -----------------------------------------------------------------------------
# Step 11: Verify installs
# -----------------------------------------------------------------------------
header "Step 11: Verification"
if maybe_prompt "Run quick import checks for R3M and VIP"; then
    echo ""
    info "Checking R3M..."
    if "$PYTHON" -c "from r3m import load_r3m; print('  R3M import: OK')" 2>&1; then
        success "R3M verified."
    else
        warn "R3M import failed. Check installation above."
    fi

    info "Checking VIP..."
    if "$PYTHON" -c "from vip import load_vip; print('  VIP import: OK')" 2>&1; then
        success "VIP verified."
    else
        warn "VIP import failed. This may be expected if the Hydra patch"
        warn "did not fully resolve all Python 3.12 compatibility issues."
        warn "See README.md for manual fix steps."
    fi

    info "Checking MetaWorld..."
    if "$PYTHON" -c "import metaworld; print('  MetaWorld import: OK')" 2>&1; then
        success "MetaWorld verified."
    else
        warn "MetaWorld import failed."
    fi

    info "Checking PyTorch..."
    if "$PYTHON" -c "import torch; print(f'  PyTorch {torch.__version__}, CUDA available: {torch.cuda.is_available()}')" 2>&1; then
        success "PyTorch verified."
    else
        warn "PyTorch import failed."
    fi
fi

# -----------------------------------------------------------------------------
# Done
# -----------------------------------------------------------------------------
header "Setup complete"
echo ""
info "To activate the virtual environment in future sessions:"
echo -e "    ${BOLD}source ${VENV_PATH}/bin/activate${RESET}"
echo ""
info "To run the MetaWorld baseline evaluation (example):"
echo -e "    ${BOLD}python3 src/evals/metaworld/r3m_metaworld_multitask.py --demo_episodes 10 --camera_id 0 --seed 42${RESET}"
echo ""
info "For dataset download instructions, see:"
echo -e "    ${BOLD}docs/DROID_DOWNLOAD.md${RESET}"
echo -e "    ${BOLD}docs/EGOEXO4D_ACCESS.md${RESET}"
echo ""