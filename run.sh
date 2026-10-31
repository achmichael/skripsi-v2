#!/bin/bash

# ==========================================
# Run Frontend + Backend
# ==========================================

set -e

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

FRONTEND_PID=""
BACKEND_PID=""

# Cleanup when stopped
cleanup() {
    echo ""
    echo "Stopping all services..."

    [ -n "$FRONTEND_PID" ] && kill "$FRONTEND_PID" 2>/dev/null || true
    [ -n "$BACKEND_PID" ] && kill "$BACKEND_PID" 2>/dev/null || true

    wait 2>/dev/null || true

    echo "All services stopped."
}

trap cleanup SIGINT SIGTERM EXIT

# Check virtual environment
if [ ! -d "venv" ]; then
    echo "Error: venv not found."
    echo "Run: python3 -m venv venv"
    exit 1
fi

# Check frontend
if [ ! -f "frontend/package.json" ]; then
    echo "Error: frontend/package.json not found."
    exit 1
fi

# Activate Python environment
source venv/bin/activate

# Start frontend
echo "=========================================="
echo "Starting Frontend..."
echo "=========================================="

cd "$ROOT_DIR/frontend"

if [ ! -d "node_modules" ]; then
    npm install
fi

npm run dev -- --host 0.0.0.0 &
FRONTEND_PID=$!

# Start backend
echo "=========================================="
echo "Starting Backend..."
echo "=========================================="

cd "$ROOT_DIR"

python -m uvicorn backend.app.main:app \
    --host 0.0.0.0 \
    --port 8000 \
    --reload &
BACKEND_PID=$!

# Display URLs
echo ""
echo "=========================================="
echo "APPLICATION RUNNING"
echo "=========================================="
echo "Frontend : http://localhost:5173"
echo "Backend  : http://localhost:8000"
echo "API Docs : http://localhost:8000/docs"
echo "=========================================="
echo "Press CTRL+C to stop both services."
echo ""

# Monitor both processes
wait -n "$FRONTEND_PID" "$BACKEND_PID"
