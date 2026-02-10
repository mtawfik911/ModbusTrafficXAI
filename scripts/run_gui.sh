#!/usr/bin/env bash
set -e
cd "$(dirname "$0")/.."
export PYTHONPATH=src
export API_URL=${API_URL:-http://127.0.0.1:8000}
streamlit run gui/app.py --server.address 0.0.0.0 --server.port 8501
