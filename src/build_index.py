
import json
import pickle
import faiss
import numpy as np
from pathlib import Path
from typing import List, Dict

from embeddings import load_model, generate_embeddings

"""
Build and save FAISS index for Portuguese-BR legal documents.

This script:
1. Loads pre-processed chunks from artifacts/chunks.pkl
2. Generates embeddings using multilingual model
3. Creates and saves FAISS index
4. Saves metadata for retrieval
"""

# Configuration
TOP_K = 5


def setup_directories() -> Path:
    """Create artifacts directory if it doesn't exist."""
    artifacts_dir = Path(__file__).parent.parent / "artifacts"
    artifacts_dir.mkdir(exist_ok=True)
    print(f"Artifacts directory: {artifacts_dir}")
    return artifacts_dir


def load_chunks(artifacts_dir: Path) -> List[Dict]:
    """Load pre-processed chunks from pickle file."""
    print("\nLoading chunks from pickle file...")
    chunks_path = artifacts_dir / "chunks.pkl"
    
    if not chunks_path.exists():
        raise FileNotFoundError(
            f"Chunks file not found: {chunks_path}\n"
            f"Please run doc_process.py first to generate chunks."
        )
    
    with open(chunks_path, "rb") as f:
        chunks = pickle.load(f)
    
    print(f"Loaded {len(chunks)} chunks")
    return chunks





def create_faiss_index(embeddings: np.ndarray) -> faiss.Index:
    """Create FAISS index from embeddings."""
    print("\nCreating FAISS index...")
    embeddings = np.array(embeddings).astype("float32")

    # Create index using L2 distance
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)

    print(f"FAISS index created with {index.ntotal} vectors")
    return index


def save_index(
    index: faiss.Index,
    chunks: List[Dict],
    artifacts_dir: Path,
    model_name: str = "intfloat/multilingual-e5-base"
) -> None:
    """Save FAISS index and metadata."""
    print("\nSaving index and metadata...")

    # Save FAISS index
    index_path = artifacts_dir / "faiss_index.bin"
    faiss.write_index(index, str(index_path))
    print(f"Saved FAISS index: {index_path}")

    # Save metadata
    metadata_path = artifacts_dir / "chunks_metadata.json"
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(chunks, f, ensure_ascii=False, indent=2)
    print(f"Saved metadata: {metadata_path}")

    # Save summary
    summary_path = artifacts_dir / "index_summary.txt"
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write(f"FAISS Index Summary\n")
        f.write(f"===================\n\n")
        f.write(f"Total vectors: {index.ntotal}\n")
        f.write(f"Embedding dimension: {index.d}\n")
        f.write(f"Total chunks: {len(chunks)}\n")
        f.write(f"Model: {model_name}\n")
    print(f"Saved summary: {summary_path}")


def main():
    """Main pipeline: load chunks, embed, index, and save."""
    print("="*60)
    print("Building FAISS Index for PT-BR Legal Documents")
    print("="*60)

    # Setup
    artifacts_dir = setup_directories()

    # Load model
    model = load_model()

    # Load chunks
    chunks = load_chunks(artifacts_dir)

    # Generate embeddings
    embeddings = generate_embeddings(chunks, model)

    # Create FAISS index
    index = create_faiss_index(embeddings)

    # Save index and metadata
    save_index(index, chunks, artifacts_dir)

    print("\n" + "="*60)
    print("FAISS index built successfully!")
    print("="*60)
    print(f"\nArtifacts saved to: {artifacts_dir}")


if __name__ == "__main__":
    main()
