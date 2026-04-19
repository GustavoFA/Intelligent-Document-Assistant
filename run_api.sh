#!/bin/bash
# FastAPI server startup script

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}Starting Intelligent Document Assistant FastAPI Server${NC}"
echo ""

# Check if artifacts exist
if [ ! -f "artifacts/faiss_index.bin" ]; then
    echo "FAISS index not found. Running build_index.py..."
    python src/build_index.py
fi

if [ ! -f "artifacts/chunks.pkl" ]; then
    echo "Chunks file not found. Running doc_process.py..."
    python src/doc_process.py
fi

echo ""
echo -e "${GREEN}Starting FastAPI server...${NC}"
echo "API Documentation: http://localhost:8000/docs"
echo "Alternative docs: http://localhost:8000/redoc"
echo ""

# Run FastAPI server
python -m uvicorn api.main:app --reload --host 0.0.0.0 --port 8000
