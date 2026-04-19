
import sys
import faiss
import pickle
import logging
from pathlib import Path
from typing import List, Dict, Any, Tuple

from .embeddings import load_model, encode_query
from .config import TOP_K, MAX_CONTEXT_DISTANCE

"""
Search functionality for retrieved documents using FAISS and pre-computed embeddings.

This script:
1. Loads pre-computed FAISS index from artifacts/faiss_index.bin
2. Loads chunks from artifacts/chunks.pkl
3. Provides search interface to query relevant documents
"""

logger = logging.getLogger(__name__)


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
            output.append(f"Score: {result['score']:.4f}")
            output.append("-" * 80)
            output.append(f"Text: {result['text']}")
            output.append("")
    
    return "\n".join(output)


def search_cli(engine: "SearchEngine") -> None:
    """Interactive command-line search interface."""
    print("\n" + "=" * 80)
    print("PT-BR Legal Documents Search Interface")
    print("=" * 80)
    print("Type 'quit' or 'exit' to close")
    print("=" * 80 + "\n")

    final_message = "\nThanks. Goodbye!"
    
    while True:
        try:
            query = input("\nEnter your search query (PT-BR):\n").strip()
            
            if query.lower() in ["quit", "exit"]:
                print(final_message)
                break
            
            if not query:
                print("Please enter a valid query.")
                continue
            
            results = engine.retrieve(query)
            print(format_results(query, results))
            
        except KeyboardInterrupt:
            print(final_message)
            break
        except Exception as e:
            print(f"Error during search: {e}")


def main():
    """Main function to initialize search system."""
    print("=" * 80)
    print("Initializing Search System")
    print("=" * 80)
    
    try:
        # Initialize SearchEngine
        engine = SearchEngine()
        
        print(f"\n" + "="*80)
        print("Search system initialized successfully!")
        print("=" * 80)
        
        # Start interactive search
        search_cli(engine)
        
    except FileNotFoundError as e:
        print(f"\nError: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\nUnexpected error: {e}")
        sys.exit(1)

class SearchEngine:
    """Encapsulates retrieval logic and loaded search resources."""

    def __init__(self) -> None:
        self.artifacts_dir = self.get_artifacts_dir()
        self.model = load_model()
        self.index = self.load_index()
        self.chunks = self.load_chunks()

    @staticmethod
    def get_artifacts_dir() -> Path:
        """Resolve and validate the artifacts directory."""
        artifacts_dir = Path(__file__).parent.parent / "artifacts"
        if not artifacts_dir.exists():
            raise FileNotFoundError(f"Artifacts directory not found: {artifacts_dir}")
        logger.info("Artifacts directory: %s", artifacts_dir)
        return artifacts_dir

    def load_index(self) -> faiss.Index:
        """Load the FAISS index from disk."""
        logger.info("Loading FAISS index...")
        index_path = self.artifacts_dir / "faiss_index.bin"

        if not index_path.exists():
            raise FileNotFoundError(
                f"FAISS index not found: {index_path}\n"
                f"Please run build_index.py first."
            )

        index = faiss.read_index(str(index_path))

        if index.ntotal == 0:
            raise ValueError("FAISS index is empty.")

        logger.info("Loaded FAISS index with %d vectors", index.ntotal)
        return index

    def load_chunks(self) -> List[Dict[str, Any]]:
        """Load chunk metadata from disk."""
        logger.info("Loading chunks...")
        chunks_path = self.artifacts_dir / "chunks.pkl"

        if not chunks_path.exists():
            raise FileNotFoundError(
                f"Chunks file not found: {chunks_path}\n"
                f"Please run doc_process.py first."
            )

        with open(chunks_path, "rb") as f:
            chunks = pickle.load(f)

        if len(chunks) != self.index.ntotal:
            raise ValueError(
                f"Mismatch between chunks ({len(chunks)}) and FAISS index ({self.index.ntotal})."
            )

        logger.info("Loaded %d chunks", len(chunks))
        return chunks

    def retrieve(self, query: str, top_k: int = TOP_K) -> List[Dict[str, Any]]:
        """Retrieve the top-k most relevant chunks for a query."""
        if not query or not query.strip():
            raise ValueError("query must be a non-empty string")

        if top_k <= 0:
            raise ValueError("top_k must be greater than 0")

        top_k = min(top_k, self.index.ntotal)

        query_embedding = encode_query(query, self.model)
        scores, indices = self.index.search(query_embedding, top_k)

        results = []
        for i, idx in enumerate(indices[0]):
            if idx == -1:
                continue

            chunk = self.chunks[idx]
            results.append({
                "rank": i + 1,
                "score": float(scores[0][i]),
                "doc_id": chunk["doc_id"],
                "chunk_id": chunk["chunk_id"],
                "filename": chunk["filename"],
                "text": chunk["text"],
            })

        return results
    
    @staticmethod
    def check_context_quality(
            context: List[Dict]
        ) -> bool:
        """Check whether retrieved context is good enough."""

        if not context:
            return False
        
        # Check if any chunk has distance below threshold (higher quality)
        if context and len(context) > 0:
            best_distance = min(chunk.get('score', float('inf')) for chunk in context)
            return best_distance < MAX_CONTEXT_DISTANCE
        
        return False

    def retrieve_with_quality_check(
            self,
            query: str,
            top_k: int = TOP_K
    ) -> Tuple[List[Dict], bool]:
        """
        Retrieve context and validate retrieval quality.
        
        Returns:
            Tuple containing:
            - context: retrieved chunks
            - quality: whether context quality is sufficient
        """

        context = self.retrieve(query, top_k)
        quality = self.check_context_quality(context)
        return context, quality

if __name__ == "__main__":
    main()
