
import sys
import faiss
import pickle
from pathlib import Path
from typing import Optional, List, Dict

from .prompt import build_prompt
from .retrieval import check_context_quality
from .llm import call_ollama, check_ollama_availability, list_models_info
from .embeddings import load_model, encode_query

"""
RAG (Retrieval-Augmented Generation) pipeline orchestration.

This module handles the complete RAG workflow:

User Query
   ↓
retrieve_context (FAISS)
   ↓
check_context_quality
   ↓
build_prompt
   ↓
call_ollama
   ↓
Response
"""


def setup_artifacts_dir() -> Path:
    """Get artifacts directory."""
    artifacts_dir = Path(__file__).parent.parent / "artifacts"
    if not artifacts_dir.exists():
        raise FileNotFoundError(f"Artifacts directory not found: {artifacts_dir}")
    return artifacts_dir


def load_faiss_index(artifacts_dir: Path) -> faiss.Index:
    """Load FAISS index from file."""
    index_path = artifacts_dir / "faiss_index.bin"
    if not index_path.exists():
        raise FileNotFoundError(
            f"FAISS index not found: {index_path}\n"
            f"Please run build_index.py first."
        )
    return faiss.read_index(str(index_path))


def load_chunks(artifacts_dir: Path) -> List[Dict]:
    """Load chunks from pickle file."""
    chunks_path = artifacts_dir / "chunks.pkl"
    if not chunks_path.exists():
        raise FileNotFoundError(
            f"Chunks file not found: {chunks_path}\n"
            f"Please run doc_process.py first."
        )
    with open(chunks_path, "rb") as f:
        chunks = pickle.load(f)
    return chunks


def retrieve_context(
    query: str,
    top_k: int = 5,
) -> Optional[List[Dict]]:
    """
    Retrieve relevant context from FAISS index for a given query.
    
    Flow:
    - Load FAISS index and chunks
    - Encode query to embedding
    - Search FAISS index
    - Return top-k results with metadata
    
    Args:
        query: User's search query
        top_k: Number of top results to retrieve
        
    Returns:
        List of retrieved chunks with distance scores, or None if retrieval fails
    """
    try:
        # Setup
        artifacts_dir = setup_artifacts_dir()
        
        # Load resources
        index = load_faiss_index(artifacts_dir)
        chunks = load_chunks(artifacts_dir)
        model = load_model()
        
        # Encode query
        query_embedding = encode_query(query, model)
        
        # Search FAISS index
        distances, indices = index.search(query_embedding, top_k)
        
        # Build results with metadata
        results = []
        for i, idx in enumerate(indices[0]):
            if idx != -1:
                chunk = chunks[idx]
                results.append({
                    "rank": i + 1,
                    "distance": float(distances[0][i]),
                    "doc_id": chunk.get("doc_id", "unknown"),
                    "chunk_id": chunk.get("chunk_id", i),
                    "filename": chunk.get("filename", "unknown"),
                    "text": chunk.get("text", ""),
                })
        
        return results if results else None
        
    except Exception as e:
        print(f"Error retrieving context: {e}")
        return None


def rag_generate(
    user_prompt: str,
    context: Optional[List[Dict]] = None,
    user_questions: Optional[List[str]] = None,
    model: Optional[str] = None,
    resource_level: str = "medium_resource",
    temperature: float = 0.7,
    top_k: int = 5,
) -> str:
    """
    Complete RAG pipeline with automatic retrieval and LLM generation.
    
    Flow:
    User Query
       ↓
    retrieve_context (FAISS) [if context not provided]
       ↓
    check_context_quality
       ↓
    build_prompt
       ↓
    call_ollama
       ↓
    Response
    
    Args:
        user_prompt: User's main question
        context: Pre-retrieved document chunks (optional, will retrieve if None)
        user_questions: Additional clarifying questions
        model: Ollama model to use (auto-selected if None)
        resource_level: Resource constraint level
        temperature: LLM temperature parameter
        top_k: Number of context chunks to retrieve
        
    Returns:
        Generated response
    """
    # Step 1: Retrieve context if not provided
    if context is None:
        print(f"\nRetrieving context for query: {user_prompt}")
        context = retrieve_context(user_prompt, top_k=top_k)
    
    # Step 2: Check context quality
    if not check_context_quality(context):
        return "Não encontrei informação suficiente nos documentos para responder esta pergunta."
    
    # Step 3: Build prompt with context
    system_message, user_message = build_prompt(user_prompt, user_questions, context)
    
    # Step 4: Call Ollama for response
    response = call_ollama(
        system_message,
        user_message,
        model=model,
        resource_level=resource_level,
        temperature=temperature,
    )
    
    # Step 5: Return response
    return response



if __name__ == "__main__":
    # Display available models
    list_models_info()
    
    # Check if Ollama is running
    if not check_ollama_availability():
        print("Ollama is not running. Start it with: ollama serve")
        sys.exit(1)
    
    # Example: End-to-end RAG flow
    print("\n" + "="*80)
    print("RAG PIPELINE FLOW DEMONSTRATION")
    print("="*80)
    
    user_question = "Como calcular o imposto de renda?"
    print(f"\nUser Query: {user_question}")
    
    # Step 1: Retrieve context
    print("\n→ Step 1: Retrieving context from FAISS...")
    context = retrieve_context(user_question, top_k=3)
    
    if context:
        print(f"✓ Retrieved {len(context)} relevant chunks")
        for result in context:
            print(f"  - Rank {result['rank']}: distance={result['distance']:.4f}")
    else:
        print("✗ No context retrieved")
        sys.exit(1)
    
    # Step 2-5: Generate RAG response
    print("\n→ Step 2-5: Building prompt and calling Ollama...")
    try:
        response = rag_generate(
            user_prompt=user_question,
            context=context,  # Use retrieved context
            resource_level="medium_resource"
        )
        print(f"\n✓ Response:\n{response}")
    except RuntimeError as e:
        print(f"✗ Error: {e}")
        sys.exit(1)
    
    print("\n" + "="*80)

