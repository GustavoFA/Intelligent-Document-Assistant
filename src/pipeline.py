import sys
from typing import Optional, List, Dict

from .search import SearchEngine
from .prompt import build_prompt
# from .retrieval import check_context_quality
from .llm import call_ollama, check_ollama_availability, list_models_info
from .embeddings import load_model, encode_query

"""
RAG (Retrieval-Augmented Generation) pipeline orchestration.

This module handles the complete RAG workflow:

User Query
   |
   V
retrieve_context (FAISS)
   |
   V
check_context_quality
   |
   V
build_prompt
   |
   V
call_ollama
   |
   V
Response
"""


# TODO - fix a stable temperature value
def rag_generate(
    user_prompt: str,
    search_engine: SearchEngine,
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
        context, quality_ok = search_engine.retrieve_with_quality_check(
            user_prompt, 
            top_k=top_k
        )
    else:
        quality_ok = search_engine.check_context_quality(context)

    # Step 2: Check context quality
    if not quality_ok:
        return "Não encontrei informação suficiente nos documentos para responder esta pergunta."
    
    # Step 3: Build prompt with context
    system_message, user_message = build_prompt(
        user_prompt, 
        user_questions, 
        context
    )
    
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

    search_engine = SearchEngine()
    
    while True:

        user_question = input("\nFaça sua pergunta (ou digite 'fim' para sair):\n").strip()

        if user_question.lower() in ["fim", "sair", "exit", "quit"]:
            print("\nEncerrando. Até logo!")
            break

        # user_question = "Como calcular o imposto de renda?"
        print(f"\nUser Query: {user_question}")
        
        # Step 1: Retrieve context
        print("\n→ Step 1: Retrieving context from FAISS...")
        context = search_engine.retrieve(user_question, top_k=3)
        
        if context:
            print(f"Retrieved {len(context)} relevant chunks")
            for result in context:
                print(f"  - Rank {result['rank']}: scores={result['score']:.4f}")
        else:
            print("No context retrieved")
            sys.exit(1)
        
        # Step 2-5: Generate RAG response
        print("\nBuilding prompt and calling Ollama")
        try:
            response = rag_generate(
                user_prompt=user_question,
                search_engine=search_engine,
                context=context,  # Use retrieved context
                resource_level="medium_resource"
            )
            print(f"\nResponse:\n{response}")
        except RuntimeError as e:
            print(f"Error retrieving context: {e}")
            sys.exit(1)
        
        print("\n" + "="*80)

