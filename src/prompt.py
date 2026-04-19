
from typing import Optional, List, Dict

"""
Prompt building utilities for RAG pipeline.
"""

def format_context_for_prompt(context: Optional[List[Dict]]) -> str:
    """
    Format retrieved context into a readable string for the prompt.
    
    Args:
        context: Retrieved document chunks
        
    Returns:
        Formatted context string
    """
    if not context:
        return "Nenhum contexto de documento foi encontrado.\n"
    
    context_parts = ["=== CONTEXTO DOS DOCUMENTOS ==="]
    for i, chunk in enumerate(context, 1):
        context_parts.append(f"\n[Documento {i} - {chunk.get('filename', 'Unknown')}]")
        distance = chunk.get('distance', 'N/A')
        #NOTE - Avoiding formatting distance if it's not a number to prevent errors in prompt construction
        if isinstance(distance, (int, float)):
            context_parts.append(f"Relevancia (distancia): {distance:.4f}")
        else:
            context_parts.append(f"Relevancia: N/A")
        context_parts.append(f"\nTexto:\n{chunk.get('text', '')}")
    
    context_parts.append("\n=== FIM DO CONTEXTO ===")
    return "\n".join(context_parts)

def build_prompt(
    user_prompt: str,
    user_questions: Optional[List[str]] = None,
    context: Optional[List[Dict]] = None,
    system_prompt: Optional[str] = None
) -> tuple:
    """
    Build structured messages for the LLM using user query and retrieved context.
    
    Args:
        user_prompt: The user's main query or question
        user_questions: Optional list of follow-up questions
        context: Optional list of retrieved document chunks with metadata
        system_prompt: Optional system message to guide the LLM (default provided if None)
        
    Returns:
        Tuple of (system_message, user_message)
    """
    # System message
    default_system_message = (
        "Você é um assistente especializado em Imposto de Renda no Brasil. "
        "Responda apenas com base no contexto fornecido. "
        "Se a informação não estiver nos documentos, indique claramente que não foi encontrada."
    )
    system_prompt = system_prompt or default_system_message
    
    # Build user message
    user_parts = []
    
    # Add context 
    user_parts.append(format_context_for_prompt(context))
    
    # Add user questions
    if user_questions:
        user_parts.append("\n=== QUESTOES DO USUARIO ===")
        for question in user_questions:
            user_parts.append(f"- {question}")
        user_parts.append("=== FIM DAS QUESTOES ===")
    
    # Add main user prompt
    user_parts.append("\n=== PERGUNTA PRINCIPAL ===")
    user_parts.append(user_prompt)
    user_parts.append("=== FIM DA PERGUNTA ===")
    
    user_message = "\n".join(user_parts)
    
    return system_message, user_message
