#!/usr/bin/env bash
set -e
cd "$(dirname "$0")/.."
export PYTHONPATH=src
uvicorn api.app:app --host 0.0.0.0 --port 8000 --reload
