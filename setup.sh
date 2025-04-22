#!/usr/bin/env bash
#
# setup.sh — ready‑to‑run environment for LLM via Ollama + uv
# Works on Ubuntu/Debian cloud VMs (systemd, amd64). GPU is optional.
# -------------------------------------------------------------

set -euo pipefail

### ───── Configurable knobs ───────────────────────────────────
PY_VERSION="3.12"                         # Python version for the venv
MODEL_TAG="gemma3:27b-it-qat"             # Ollama model to pull/run
PROJECT_DIR="${PROJECT_DIR:-$HOME/text2sql-workshop}" # Updated project dir name
VENV_DIR="$PROJECT_DIR/.venv"
# REQUIREMENTS_FILE="requirements.in"        # Removed, using pyproject.toml
### ────────────────────────────────────────────────────────────

echo "==> Installing uv (fast pip/venv) if missing..."
if ! command -v uv >/dev/null 2>&1; then
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="$HOME/.local/bin:$PATH"
fi

echo "==> Creating project folder & Python $PY_VERSION venv with uv..."
mkdir -p "$PROJECT_DIR"
cd "$PROJECT_DIR"
# Add check to ensure cd was successful
if [[ "$PWD" != "$PROJECT_DIR" ]]; then
  echo "Error: Failed to change directory to $PROJECT_DIR. Current directory is $PWD." >&2
  exit 1
fi
uv venv --python "$PY_VERSION"
# activate for the rest of this script
source "$VENV_DIR/bin/activate"

echo "==> Installing Python dependencies from pyproject.toml..."
# Add check for pyproject.toml before syncing
if [[ ! -f "pyproject.toml" ]]; then
  echo "Error: pyproject.toml not found in $PROJECT_DIR. Please add it before running setup." >&2
  exit 1
fi
# uv sync automatically finds pyproject.toml or requirements.lock
uv sync

echo "==> Installing the Ollama server binary..."
if ! command -v ollama >/dev/null 2>&1; then
  curl -fsSL https://ollama.com/install.sh | sh
fi

# systemd service (Ubuntu package adds it automatically); start now if not running
if ! pgrep -x ollama >/dev/null; then
  echo "==> Starting Ollama daemon..."
  ollama serve &
  sleep 5
fi

echo "==> Pulling the model $MODEL_TAG (first run only, adjust size expectation)..."
ollama pull "$MODEL_TAG"

cat <<EOF

🎉  All done!

➡  Activate environment later with:
    source "$VENV_DIR/bin/activate"

➡  Chat with the model (replace if needed):
    ollama run $MODEL_TAG

(If you want Ollama to start on boot, enable its systemd service:)
    sudo systemctl enable --now ollama

Happy hacking!
EOF
