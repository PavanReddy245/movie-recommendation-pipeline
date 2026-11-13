#!/usr/bin/env bash
set -euo pipefail
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_DIR"
mkdir -p logs
exec .venv/bin/python run_pipeline.py >> logs/scheduled_pipeline.log 2>&1
