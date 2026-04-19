#!/bin/bash

set -e

GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo -e "${BLUE}Starting Intelligent Document Assistant API${NC}"
echo ""

# Ensure artifacts folder exists
mkdir -p artifacts

# Step 1: chunks first
if [ ! -f "artifacts/chunks.pkl" ]; then
    echo -e "${YELLOW}Chunks not found. Running doc_process...${NC}"
    python3 -m src.doc_process
fi

# Step 2: index second
if [ ! -f "artifacts/faiss_index.bin" ]; then
    echo -e "${YELLOW}FAISS index not found. Running build_index...${NC}"
    python3 -m src.build_index
fi

echo ""
echo -e "${GREEN}Starting FastAPI server...${NC}"
echo "Swagger Docs: http://localhost:8000/docs"
echo "ReDoc:        http://localhost:8000/redoc"
echo ""

python3 -m uvicorn api.main:app --reload --host 0.0.0.0 --port 8000