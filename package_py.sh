#!/bin/bash
# package_py.sh - Install the stampli package into the active Python (pyenv 'bio' recommended)
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENV_NAME="stampli"

# Ensure we are inside the expected pyenv environment if pyenv is available
if command -v pyenv >/dev/null 2>&1; then
  current_env="$(pyenv version-name)"
  if [[ "${current_env}" != "${ENV_NAME}" ]]; then
    echo "[!] Detected pyenv environment: ${current_env}"
    echo "[!] Recommended env is '${ENV_NAME}'. Consider:"
    echo "    pyenv virtualenv 3.11 ${ENV_NAME}   # once"
    echo "    pyenv activate ${ENV_NAME}"
    echo "Or rerun with PYENV_VERSION=${ENV_NAME} prefixed."
  fi
fi

echo "Using Python: $(command -v python)"
echo "Installing stampli package in editable mode..."
python -m pip install -e "${SCRIPT_DIR}"
echo "Install complete."
