"""
RAG (Retrieval-Augmented Generation) pipeline orchestration.

This module handles the complete RAG workflow:
1. Validates retrieved context quality
2. Builds prompts with context and user questions
3. Calls LLM for response generation
4. Provides end-to-end interface
"""

import sys
from typing import Optional, List, Dict

from .prompt import build_prompt
from .retrieval import check_context_quality
from .llm import call_ollama, check_ollama_availability, list_models_info


def rag_generate(
    user_prompt: str,
    context: Optional[List[Dict]] = None,
    user_questions: Optional[List[str]] = None,
    model: Optional[str] = None,
    resource_level: str = "medium_resource",
    temperature: float = 0.7,
) -> str:
    """
    Complete RAG pipeline: validate context, build prompt, and generate response.
    
    Args:
        user_prompt: User's main question
        context: Retrieved document chunks
        user_questions: Additional clarifying questions
        model: Ollama model to use (auto-selected if None)
        resource_level: Resource constraint level
        temperature: LLM temperature parameter
        
    Returns:
        Generated response
    """
    # Check context quality
    if not check_context_quality(context):
        return "Não encontrei informação suficiente nos documentos para responder esta pergunta."
    
    # Build messages with system and user roles
    system_message, user_message = build_prompt(user_prompt, user_questions, context)
    
    # Generate response
    response = call_ollama(
        system_message,
        user_message,
        model=model,
        resource_level=resource_level,
        temperature=temperature,
    )
    
    return response


if __name__ == "__main__":
    # Example usage
    list_models_info()
    
    # Check if Ollama is running
    if not check_ollama_availability():
        print("Ollama is not running. Start it with: ollama serve")
        sys.exit(1)
    
    # Example context from search results
    example_context = [
        {
            "filename": "IRPF_2024.pdf",
            "distance": 0.1234,
            "text": "O Imposto de Renda da Pessoa Fisica (IRPF) é um tributo federal...",
        }
    ]
    
    # Build and test prompt
    user_question = "Como calcular o imposto de renda?"
    system_msg, user_msg = build_prompt(user_question, context=example_context)
    print("\nGenerated system message:")
    print(system_msg)
    print("\nGenerated user message:")
    print(user_msg)
