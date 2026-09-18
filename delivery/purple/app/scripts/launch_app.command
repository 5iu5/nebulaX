#!/bin/bash
cd "$(dirname "$0")/.." || exit 1
exec .venv/bin/streamlit run app/main.py --server.address 127.0.0.1 --server.port 8501
