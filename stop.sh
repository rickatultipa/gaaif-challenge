#!/usr/bin/env bash
# Stop the GAAIF dashboard
PID_FILE="$(dirname "$0")/.dashboard.pid"
if [ -f "$PID_FILE" ]; then
    PID=$(cat "$PID_FILE")
    kill "$PID" 2>/dev/null && echo "Dashboard stopped (PID $PID)" || echo "Process $PID not running"
    rm "$PID_FILE"
else
    echo "No PID file found — dashboard may not be running"
fi
