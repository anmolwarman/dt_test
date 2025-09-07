#!/usr/bin/env bash
set -euo pipefail
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
echo "Virtual env ready. Activate with: source .venv/bin/activate"
