#!/usr/bin/env bash
set -euo pipefail

cd /workspace

python -m ipykernel install --name python3 --display-name "Python 3 (vllm_learn)" --user >/dev/null 2>&1

APP="${APP:-/workspace/docker/hub.py}"
TOKEN="${JUPYTER_TOKEN:-vllm_learn}"

jupyter lab \
    --ip=0.0.0.0 \
    --port="${JUPYTER_PORT:-8888}" \
    --no-browser \
    --allow-root \
    --IdentityProvider.token="${TOKEN}" \
    --ServerApp.allow_origin="*" \
    --ServerApp.root_dir=/workspace &

exec streamlit run "${APP}" \
    --server.address=0.0.0.0 \
    --server.port="${STREAMLIT_PORT:-8501}" \
    --server.headless=true \
    --browser.gatherUsageStats=false
