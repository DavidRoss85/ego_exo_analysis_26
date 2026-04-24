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
# Pre-flight: Python and pip availability checks
# -----------------------------------------------------------------------------
header "Pre-flight checks"

# Check that python3 exists
if ! command -v python3 &>/dev/null; then
    error "python3 not found. Please install Python 3.12 before running this script."
    error "On Ubuntu: sudo apt-get install python3.12 python3.12-venv"
    exit 1
fi

# Check Python version — require 3.12.x
PY_VERSION=$(python3 -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
PY_MAJOR=$(echo "$PY_VERSION" | cut -d. -f1)
PY_MINOR=$(echo "$PY_VERSION" | cut -d. -f2)

info "Detected Python ${PY_VERSION}"

if [ "$PY_MAJOR" -ne 3 ] || [ "$PY_MINOR" -ne 12 ]; then
    warn "This project was developed and tested on Python 3.12."
    warn "Your system default is Python ${PY_VERSION}."
    warn ""
    warn "You have two options:"
    warn "  1. Install Python 3.12 and re-run this script."
    warn "     On Ubuntu: sudo apt-get install python3.12 python3.12-venv"
    warn "  2. Continue anyway -- things may work on Python ${PY_VERSION},"
    warn "     but the Hydra compatibility patch targets 3.12 specifically"
    warn "     and other dependency issues may arise."
    echo ""
    read -rp "Continue with Python ${PY_VERSION}? [y/N] " ans
    case "$ans" in
        [Yy]*) info "Continuing with Python ${PY_VERSION}." ;;
        *) error "Aborted. Please install Python 3.12 and re-run."; exit 1 ;;
    esac
else
    success "Python ${PY_VERSION} -- OK"
fi

# Check that pip is available
if ! python3 -m pip --version &>/dev/null; then
    warn "pip not found for Python ${PY_VERSION}. Attempting to install..."
    if command -v apt-get &>/dev/null; then
        sudo apt-get install -y python3-pip
    else
        error "Could not install pip automatically. Please install it manually:"
        error "  python3 -m ensurepip --upgrade"
        exit 1
    fi
fi
success "pip -- OK"

# Check that python3-venv is available by doing a dry-run
if ! python3 -m venv --help &>/dev/null; then
    warn "python3-venv module not available. Attempting to install..."
    if command -v apt-get &>/dev/null; then
        sudo apt-get install -y python3.12-venv
    else
        error "Could not install python3-venv automatically."
        error "Please run: sudo apt-get install python3.12-venv"
        exit 1
    fi
fi
success "python3-venv -- OK"

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
info "python3-venv and python3-pip are required. OpenGL and GLib packages"
info "are only needed if MetaWorld rendering fails with a missing shared"
info "library error -- most desktop Ubuntu installs already have them."

# Detect Ubuntu version to select correct package names
UBUNTU_VERSION=$(lsb_release -rs 2>/dev/null || echo "0")
UBUNTU_MAJOR=$(echo "$UBUNTU_VERSION" | cut -d. -f1)
info "Detected Ubuntu ${UBUNTU_VERSION}"

if [ "$UBUNTU_MAJOR" -ge 24 ]; then
    # Ubuntu 24.04+: libgl1-mesa-glx was removed, libglib2.0-0 renamed to libglib2.0-0t64
    GL_PKG="libgl1"
    GLIB_PKG="libglib2.0-0t64"
else
    # Ubuntu 22.04 and earlier
    GL_PKG="libgl1-mesa-glx"
    GLIB_PKG="libglib2.0-0"
fi

info "OpenGL package : ${GL_PKG}"
info "GLib package   : ${GLIB_PKG}"

if maybe_prompt "Install/verify system packages via apt-get"; then
    info "Updating apt package list..."
    sudo apt-get update -qq
    info "Installing system packages (skipped if already present)..."
    sudo apt-get install -y \
        python3-venv \
        python3-pip \
        "$GL_PKG" \
        "$GLIB_PKG"
    success "System packages verified."
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
# Step 3: Upgrade pip and setuptools
# -----------------------------------------------------------------------------
header "Step 3: Upgrade pip and setuptools"
if maybe_prompt "Upgrade pip and setuptools inside the virtual environment"; then
    "$PYTHON" -m pip install --upgrade pip -q 2>/dev/null
    # Install common dependencies early so pip's resolver sees them as
    # satisfied before anything else runs.
    "$PYTHON" -m pip install pyyaml jinja2 typeguard -q 2>/dev/null
    # setuptools==81.0.0 is pinned -- required for pkg_resources to be importable
    # (needed by gdown and r3m at import time on Python 3.12).
    "$PYTHON" -m pip install "setuptools==81.0.0" "wheel==0.47.0" -q 2>/dev/null
    if "$PYTHON" -c "import pkg_resources" &>/dev/null; then
        success "pip, setuptools==81.0.0, wheel -- OK"
    else
        warn "pkg_resources still not importable after setuptools install."
        warn "Continuing -- this may cause r3m import to fail."
    fi
fi

# -----------------------------------------------------------------------------
# Step 4: PyTorch (must be installed before requirements.txt due to custom index)
# -----------------------------------------------------------------------------
header "Step 4: PyTorch"
if maybe_prompt "Install PyTorch"; then
    info "Installing torch and torchvision from PyPI..."
    "$PIP" install torch torchvision -q
    success "PyTorch installed."
fi

# -----------------------------------------------------------------------------
# Step 5: All remaining packages from requirements.txt
# -----------------------------------------------------------------------------
header "Step 5: Install packages from requirements.txt"
REQUIREMENTS_FILE="${REPO_ROOT}/requirements.txt"
if [ ! -f "$REQUIREMENTS_FILE" ]; then
    error "requirements.txt not found at ${REQUIREMENTS_FILE}"
    error "Make sure the file is in the repository root."
    exit 1
fi
if maybe_prompt "Install all packages from requirements.txt"; then
    info "This installs pinned versions from the known-good working environment."
    "$PIP" install -r "$REQUIREMENTS_FILE" -q
    success "requirements.txt packages installed."
fi

# -----------------------------------------------------------------------------
# Step 6: R3M submodule
# -----------------------------------------------------------------------------
header "Step 6: Install R3M"
R3M_DIR="${REPO_ROOT}/r3m"
if [ -f "${R3M_DIR}/setup.py" ] || [ -f "${R3M_DIR}/pyproject.toml" ]; then
    if maybe_prompt "Install R3M from submodule (pip install -e ./r3m)"; then
        # Uninstall any prior r3m to clear stale egg-links or direct_url.json
        # entries that cause 'unknown location' errors when the venv is sourced.
        "$PIP" uninstall r3m -y 2>/dev/null || true
        # Also remove any stale .egg-link files left by previous editable installs
        find "${VENV_PATH}" -name "r3m.egg-link" -delete 2>/dev/null || true
        find "${VENV_PATH}" -name "r3m*.dist-info" -type d \
            -exec rm -rf {} + 2>/dev/null || true
        # Clear PYTHONPATH so no other venv's packages bleed in during install
        PYTHONPATH="" "$PIP" install -e "$R3M_DIR"
        success "R3M installed."
    fi
else
    warn "R3M submodule not found at ${R3M_DIR}."
    warn "Make sure you cloned with --recurse-submodules, or run:"
    warn "  git submodule update --init --recursive"
fi

# -----------------------------------------------------------------------------
# Step 7: VIP submodule
# -----------------------------------------------------------------------------
header "Step 7: Install VIP"
VIP_DIR="${REPO_ROOT}/vip"
if [ -f "${VIP_DIR}/setup.py" ] || [ -f "${VIP_DIR}/pyproject.toml" ]; then
    if maybe_prompt "Install VIP from submodule (pip install -e ./vip)"; then
        "$PIP" uninstall vip -y 2>/dev/null || true
        find "${VENV_PATH}" -name "vip.egg-link" -delete 2>/dev/null || true
        find "${VENV_PATH}" -name "vip*.dist-info" -type d \
            -exec rm -rf {} + 2>/dev/null || true
        PYTHONPATH="" "$PIP" install -e "$VIP_DIR"
        success "VIP installed."
    fi
else
    warn "VIP submodule not found at ${VIP_DIR}."
    warn "Make sure you cloned with --recurse-submodules, or run:"
    warn "  git submodule update --init --recursive"
fi

# -----------------------------------------------------------------------------
# Step 8: Hydra / Python 3.12 compatibility patch
# -----------------------------------------------------------------------------
header "Step 8: Hydra Python 3.12 compatibility patch"
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

        # Write the patch script to a temp file so we only run it once
        PATCH_SCRIPT=$(mktemp /tmp/hydra_patch_XXXXXX.py)
        cat > "$PATCH_SCRIPT" << 'PYEOF'
import sys
import re

filepath = sys.argv[1]
with open(filepath, "r") as f:
    content = f.read()

original = content
changes = []

# Ensure 'field' is imported from dataclasses
if "from dataclasses import" in content:
    def add_field_to_import(m):
        imports = m.group(1)
        if "field" not in [x.strip() for x in imports.split(",")]:
            imports = imports.rstrip() + ", field"
            changes.append("Added 'field' to dataclasses import")
        return f"from dataclasses import {imports}"
    content = re.sub(r"from dataclasses import ([^\n]+)", add_field_to_import, content)

# Replace mutable dataclass defaults: SomeClass = SomeClass() -> field(default_factory=SomeClass)
pattern = re.compile(
    r"^( {4,})(\w+):\s*([\w.]+)\s*=\s*([\w.]+)\(\)\s*$",
    re.MULTILINE
)

primitives = {"str", "int", "float", "bool", "list", "dict", "tuple", "set"}

def replace_mutable_default(m):
    indent, fieldname, typehint, factory = m.group(1), m.group(2), m.group(3), m.group(4)
    if factory in primitives:
        return m.group(0)
    changes.append(f"  Line patched: '{fieldname}: {typehint} = {factory}()' -> 'field(default_factory={factory})'")
    return f"{indent}{fieldname}: {typehint} = field(default_factory={factory})"

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

        # Run the patch script once and capture output
        PATCH_OUTPUT=$("$PYTHON" "$PATCH_SCRIPT" "$HYDRA_CONF" 2>&1)
        PATCH_EXIT=$?
        rm -f "$PATCH_SCRIPT"

        # Display results
        while IFS= read -r line; do
            case "$line" in
                PATCHED*)  success "${line#PATCHED|}" ;;
                CHANGE*)   echo -e "    ${GREEN}+${RESET} ${line#CHANGE|}" ;;
                NOCHANGE*) warn "${line#NOCHANGE|}" ;;
                *)         info "$line" ;;
            esac
        done <<< "$PATCH_OUTPUT"

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
# Step 9: Verify installs
# -----------------------------------------------------------------------------
header "Step 9: Verification"
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
info "The virtual environment is ready at: ${VENV_PATH}"
echo ""
warn "NOTE: This script cannot activate the virtual environment in your"
warn "current terminal because child processes cannot modify the parent shell."
echo ""
info "To activate now, run:"
echo -e "    ${BOLD}source ${VENV_PATH}/bin/activate${RESET}"
echo ""
info "TIP: If you want the venv activated automatically after setup in future"
info "runs, source this script instead of executing it:"
echo -e "    ${BOLD}. ./setup.sh${RESET}"
echo ""
info "To run the MetaWorld baseline evaluation (example):"
echo -e "    ${BOLD}python3 src/evals/metaworld/r3m_metaworld_multitask.py --encoder baseline --single_run --demos 10 --camera 0${RESET}"
echo ""
info "For dataset download instructions, see:"
echo -e "    ${BOLD}docs/DROID_DOWNLOAD.md${RESET}"
echo -e "    ${BOLD}docs/EGOEXO4D_ACCESS.md${RESET}"
echo ""