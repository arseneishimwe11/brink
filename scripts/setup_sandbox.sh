#!/usr/bin/env bash
# Set up a fresh machine (or cloud sandbox) to run Brink.
# Run from the repository root:  bash scripts/setup_sandbox.sh
set -euo pipefail

# System libraries that the Open3D wheel loads at import time, plus ffmpeg,
# which the benchmark uses to encode synthetic videos.
if command -v apt-get >/dev/null 2>&1; then
  SUDO=""
  if [ "$(id -u)" -ne 0 ]; then SUDO="sudo"; fi
  $SUDO apt-get update -qq
  $SUDO apt-get install -y --no-install-recommends libegl1 libgl1 libgomp1 ffmpeg
fi

# Project virtual environment (Python 3.12: Open3D has no wheel for every newer version).
if command -v uv >/dev/null 2>&1; then
  uv venv --python 3.12 .venv
  . .venv/bin/activate
  uv pip install -e ".[dev]"
else
  python3.12 -m venv .venv
  . .venv/bin/activate
  pip install -e ".[dev]"
fi

python -c "import open3d, pycolmap, cv2; print('Brink environment ready')"
