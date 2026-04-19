
import sys
import faiss
import pickle
import numpy as np
from pathlib import Path
from typing import List, Dict
from sentence_transformers import SentenceTransformer

"""
Search functionality for retrieved documents using FAISS and pre-computed embeddings.

This script:
1. Loads pre-computed FAISS index from artifacts/faiss_index.bin
2. Loads chunks from artifacts/chunks.pkl
3. Provides search interface to query relevant documents
"""

# Configuration
MODEL_NAME = "intfloat/multilingual-e5-base"
TOP_K = 5


def setup_directories() -> Path:
    """Get artifacts directory."""
    artifacts_dir = Path(__file__).parent.parent / "artifacts"
    if not artifacts_dir.exists():
        raise FileNotFoundError(f"Artifacts directory not found: {artifacts_dir}")
    print(f"Artifacts directory: {artifacts_dir}")
    return artifacts_dir


def load_index(artifacts_dir: Path) -> faiss.Index:
    """Load FAISS index from file."""
    print("\nLoading FAISS index...")
    index_path = artifacts_dir / "faiss_index.bin"
    
    if not index_path.exists():
        raise FileNotFoundError(
            f"FAISS index not found: {index_path}\n"
            f"Please run build_index.py first."
        )
    
    index = faiss.read_index(str(index_path))
    print(f"Loaded FAISS index with {index.ntotal} vectors")
    return index


def load_chunks(artifacts_dir: Path) -> List[Dict]:
    """Load chunks from pickle file."""
    print("\nLoading chunks...")
    chunks_path = artifacts_dir / "chunks.pkl"
    
    if not chunks_path.exists():
        raise FileNotFoundError(
            f"Chunks file not found: {chunks_path}\n"
            f"Please run doc_process.py first."
        )
    
    with open(chunks_path, "rb") as f:
        chunks = pickle.load(f)
    
    print(f"Loaded {len(chunks)} chunks")
    return chunks


def retrieve(
    query: str,
    index: faiss.Index,
    chunks: List[Dict],
    model: SentenceTransformer,
    top_k: int = TOP_K
) -> List[Dict]:
    """Retrieve top-k relevant chunks for a query."""
    # Encode query
    query_embedding = model.encode([query], convert_to_numpy=True).astype("float32")
    
    # Search in FAISS index
    distances, indices = index.search(query_embedding, top_k)
    
    # Retrieve relevant chunks
    results = []
    for i, idx in enumerate(indices[0]):
        if idx != -1:
            chunk = chunks[idx]
            results.append({
                "rank": i + 1,
                "distance": float(distances[0][i]),
                "doc_id": chunk["doc_id"],
                "chunk_id": chunk["chunk_id"],
                "filename": chunk["filename"],
                "text": chunk["text"]
            })
    
    return results


def format_results(query: str, results: List[Dict]) -> str:
    """Format search results for display."""
    output = []
    output.append("=" * 80)
    output.append(f"Query: {query}")
    output.append("=" * 80)
    
    if not results:
        output.append("No results found.")
    else:
        for result in results:
            output.append(f"\n[Rank {result['rank']}] {result['filename']}")
            output.append(f"Document ID: {result['doc_id']} | Chunk ID: {result['chunk_id']}")
            output.append(f"Distance: {result['distance']:.4f}")
            output.append("-" * 80)
            output.append(f"Text: {result['text']}")
            output.append("")
    
    return "\n".join(output)


def search_cli(model: SentenceTransformer, index: faiss.Index, chunks: List[Dict]) -> None:
    """Interactive command-line search interface."""
    print("\n" + "=" * 80)
    print("PT-BR Legal Documents Search Interface")
    print("=" * 80)
    print("Type 'quit' or 'exit' to close")
    print("=" * 80 + "\n")
    
    while True:
        try:
            query = input("\nEnter your search query (PT-BR): ").strip()
            
            if query.lower() in ["quit", "exit"]:
                print("\nGoodbye!")
                break
            
            if not query:
                print("Please enter a valid query.")
                continue
            
            results = retrieve(query, index, chunks, model)
            print(format_results(query, results))
            
        except KeyboardInterrupt:
            print("\n\nGoodbye!")
            break
        except Exception as e:
            print(f"Error during search: {e}")


def main():
    """Main function to initialize search system."""
    print("=" * 80)
    print("Initializing Search System")
    print("=" * 80)
    
    try:
        # Setup
        artifacts_dir = setup_directories()
        
        # Load model
        print(f"\nLoading model: {MODEL_NAME}")
        model = SentenceTransformer(MODEL_NAME)
        print("✓ Model loaded")
        
        # Load index and chunks
        index = load_index(artifacts_dir)
        chunks = load_chunks(artifacts_dir)
        
        print("\n" + "=" * 80)
        print("Search system initialized successfully!")
        print("=" * 80)
        
        # Start interactive search
        search_cli(model, index, chunks)
        
    except FileNotFoundError as e:
        print(f"\nError: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\nUnexpected error: {e}")
        sys.exit(1)


def search(query: str, top_k: int = TOP_K) -> List[Dict]:
    """
    Standalone search function for programmatic use.
    
    Usage:
        from search import search
        results = search("Como calcular o imposto de renda?")
    """
    artifacts_dir = setup_directories()
    model = SentenceTransformer(MODEL_NAME)
    index = load_index(artifacts_dir)
    chunks = load_chunks(artifacts_dir)
    
    return retrieve(query, index, chunks, model, top_k=top_k)


if __name__ == "__main__":
    main()
