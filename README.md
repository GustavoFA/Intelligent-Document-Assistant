# Intelligent-Document-Assistant

An end-to-end intelligent document assistant for Portuguese-Brazilian (PT-BR) legal documents. This system leverages **Retrieval-Augmented Generation (RAG)**, **semantic search**, and **FAISS indexing** to provide accurate, grounded, and explainable answers to questions about legal documents.

## Project Overview

This project implements a production-ready pipeline for:
- **Document Processing**: Chunk PT-BR legal documents respecting legal structure (Articles, Paragraphs, Inciso)
- **Semantic Embeddings**: Generate multilingual embeddings optimized for PT-BR legal content
- **Vector Search**: Build and query FAISS index for efficient similarity search
- **Retrieval**: Retrieve relevant legal document chunks based on semantic similarity

The system currently focuses on Brazilian income tax (IRPF) legal documents from the [unicamp-dl/rag-rfb](https://huggingface.co/datasets/unicamp-dl/rag-rfb) Hugging Face dataset.

## Workflow

### Step 1: Document Processing (`doc_process.py`)

Loads and chunks documents from Hugging Face dataset:

```bash
python src/doc_process.py
```

**What it does:**
- Loads legal documents from `unicamp-dl/rag-rfb` dataset
- Cleans text (removes glyphs, normalizes whitespace)
- Chunks documents with legal structure awareness:
  - Separators: Articles, Paragraphs, Inciso, etc.
  - Chunk size: 700 tokens
  - Overlap: 100 tokens
- Saves chunks to `artifacts/chunks.pkl`

**Output:**
- `artifacts/chunks.pkl` - Serialized chunks with metadata (doc_id, filename, text, position)

### Step 2: Build FAISS Index (`build_index.py`)

Generates embeddings and builds vector index:

```bash
python src/build_index.py
```

**What it does:**
- Loads pre-processed chunks from `artifacts/chunks.pkl`
- Generates embeddings using `intfloat/multilingual-e5-base`:
  - Multilingual model optimized for semantic similarity
  - Batch processing (32 chunks per batch)
  - Generates 768-dimensional embeddings
- Creates FAISS index:
  - IndexFlatL2 for L2 distance-based search
  - Supports efficient similarity search
- Saves index and metadata to artifacts

**Output:**
- `artifacts/faiss_index.bin` - Serialized FAISS index
- `artifacts/chunks_metadata.json` - Chunk metadata for reference
- `artifacts/index_summary.txt` - Index statistics and configuration

### Step 3: Search (`search.py`)

Query the indexed documents:

```bash
# Interactive CLI
python src/search.py

# Programmatic usage
from src.search import search
results = search("Como calcular o imposto de renda?", top_k=5)
```

**Features:**
- Interactive command-line interface for searching
- Programmatic search function for integration
- Returns ranked results with:
  - Rank and L2 distance score
  - Document metadata (filename, doc_id, chunk_id)
  - Full chunk text

## Project Structure

```
Intelligent-Document-Assistant/
├── README.md
├── LICENSE
├── notebooks/
│   └── retrieval.ipynb          # Exploratory notebook with examples
├── src/
│   ├── doc_process.py           # Document processing and chunking
│   ├── build_index.py           # Embedding generation and indexing
│   └── search.py                # Search interface
└── artifacts/                   # Generated artifacts (indexed files)
    ├── chunks.pkl               # Processed chunks
    ├── faiss_index.bin          # FAISS index
    ├── chunks_metadata.json     # Metadata
    └── index_summary.txt        # Summary
```

## Quick Start

**Full pipeline:**
```bash
# 1. Process documents
python src/doc_process.py

# 2. Build FAISS index
python src/build_index.py

# 3. Start searching
python src/search.py
```

**Example queries (PT-BR):**
- "Como calcular o imposto de renda?"
- "Qual é a alíquota do IRPF?"
- "Como fazer dedução fiscal?"

## Configuration

Key parameters in each script:

**doc_process.py:**
- `CHUNK_SIZE = 700` - Token size for each chunk
- `CHUNK_OVERLAP = 100` - Token overlap between chunks
- `SEPARATORS` - Legal structure patterns (Articles, Paragraphs, etc.)

**build_index.py:**
- `BATCH_SIZE = 32` - Batch size for embedding generation
- `MODEL_NAME = "intfloat/multilingual-e5-base"` - Embedding model
- `TOP_K = 5` - Default number of results

## Dependencies

- `faiss-cpu` or `faiss-gpu` - Vector similarity search
- `sentence-transformers` - Multilingual embeddings
- `langchain-text-splitters` - Document chunking
- `datasets` - Hugging Face datasets
- `numpy`, `pandas` - Data processing

Install with:
```bash
pip install faiss-cpu sentence-transformers langchain-text-splitters datasets numpy pandas
```

## License

See [LICENSE](LICENSE) file for details.
