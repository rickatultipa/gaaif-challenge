#!/usr/bin/env bash
# Start the GAAIF Structured Forward Pricer dashboard
cd "$(dirname "$0")/src"
python3 -m streamlit run dashboard.py --server.headless true &
echo $! > ../.dashboard.pid
echo "Dashboard started (PID $(cat ../.dashboard.pid))"
echo "Open http://localhost:8501"
