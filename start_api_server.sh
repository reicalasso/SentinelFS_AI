#!/bin/bash
# Start SentinelFS AI REST API Server

echo "=========================================="
echo "  SentinelFS AI - REST API Server"
echo "=========================================="
echo ""

# Check if virtual environment exists
if [ ! -d ".venv" ]; then
    echo "❌ Virtual environment not found!"
    echo "   Run: python -m venv .venv"
    exit 1
fi

# Activate virtual environment
echo "✓ Activating virtual environment..."
source .venv/bin/activate

# Install dependencies if needed
echo "✓ Checking dependencies..."
pip install -q fastapi uvicorn pydantic httpx 2>/dev/null

echo ""
echo "Starting API server..."
echo "  • URL: http://localhost:8000"
echo "  • Docs: http://localhost:8000/docs"
echo "  • ReDoc: http://localhost:8000/redoc"
echo ""
echo "Press Ctrl+C to stop the server"
echo "=========================================="
echo ""

# Start server
uvicorn sentinelfs_ai.api.server:app \
    --host 0.0.0.0 \
    --port 8000 \
    --reload \
    --log-level info
