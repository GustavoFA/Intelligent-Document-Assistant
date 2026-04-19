"""
LLM (Ollama) integration utilities for RAG pipeline.
"""

import time
import requests
from typing import Optional, List

from .config import (
    OLLAMA_CHAT_ENDPOINT,
    OLLAMA_TAGS_ENDPOINT,
    MODELS_CONFIG,
    MAX_RETRIES,
    RETRY_DELAY,
    REQUEST_TIMEOUT,
)


def check_ollama_availability() -> bool:
    """
    Check if Ollama is running and accessible.
    
    Returns:
        True if Ollama is available, False otherwise
    """
    try:
        response = requests.get(OLLAMA_TAGS_ENDPOINT, timeout=5)
        return response.status_code == 200
    except (requests.ConnectionError, requests.Timeout):
        return False


def get_available_models() -> List[str]:
    """
    Get list of available models installed in Ollama.
    
    Returns:
        List of model names, empty list if none available
    """
    try:
        response = requests.get(OLLAMA_TAGS_ENDPOINT, timeout=5)
        if response.status_code == 200:
            data = response.json()
            models = data.get("models", [])
            # return [model.get("name") for model in models if models]
            return [model.get("name") for model in models if model and model.get("name")]
        return []
    except (requests.ConnectionError, requests.Timeout, ValueError):
        return []


def select_model(resource_level: str = "medium_resource") -> Optional[str]:
    """
    Select an appropriate model based on resource constraints.
    
    Args:
        resource_level: One of 'high_resource', 'medium_resource', or 'low_resource'
        
    Returns:
        Model name to use, None if no suitable model found
    """
    if resource_level not in MODELS_CONFIG:
        print(f"Invalid resource level: {resource_level}")
        resource_level = "medium_resource"
    
    available_models = get_available_models()
    
    if not available_models:
        print("No models found in Ollama. Please pull a model first:")
        print("  ollama pull llama2:7b")
        return None
    
    # Try to find a model from the preferred list
    preferred_models = MODELS_CONFIG[resource_level]["models"]
    
    for model in preferred_models:
        for available in available_models:
            if model in available or available.startswith(model.split(":")[0]):
                return available
    
    # Fall back to first available model
    print(f"Preferred models not found. Using: {available_models[0]}")
    return available_models[0]

# TODO - temperature should be optional
def call_ollama(
    system_message: str,
    user_message: str,
    model: Optional[str] = None,
    resource_level: str = "medium_resource",
    temperature: float = 0.7,
    top_p: float = 0.9,
) -> str:
    """
    Call Ollama API to generate a response with retry logic.
    
    Args:
        system_message: System role message
        user_message: User role message
        model: Model name to use (auto-selected if None)
        resource_level: Resource constraint level for auto-selection
        temperature: Controls randomness (0.0-1.0)
        top_p: Controls diversity (0.0-1.0)
        
    Returns:
        Generated response text
        
    Raises:
        RuntimeError: If Ollama is not available or model fails
    """
    # Check Ollama availability
    if not check_ollama_availability():
        raise RuntimeError(
            "Ollama is not running. Start it with: ollama serve"
        )
    
    # Select model if not provided
    if model is None:
        model = select_model(resource_level)
        if model is None:
            raise RuntimeError(
                "No Ollama model available. Install one with: ollama pull llama2"
            )
    
    print(f"\nUsing model: {model}")
    
    # Prepare messages with separate system and user roles
    messages = [
        {
            "role": "system",
            "content": system_message,
        },
        {
            "role": "user",
            "content": user_message,
        },
    ]
    
    # Retry logic
    last_error = None
    for attempt in range(MAX_RETRIES + 1):
        try:
            # Call Ollama API
            payload = {
                "model": model,
                "messages": messages,
                "stream": False,
                "options": {
                    "temperature": temperature,
                    "top_p": top_p,
                },
            }
            
            print(f"Request attempt {attempt + 1}/{MAX_RETRIES + 1}...")
            
            response = requests.post(
                OLLAMA_CHAT_ENDPOINT,
                json=payload,
                timeout=REQUEST_TIMEOUT,
            )
            
            if response.status_code != 200:
                raise RuntimeError(f"Ollama API error: {response.status_code}: {response.text}")
            
            result = response.json()
            return result.get("message", {}).get("content", "")
            
        except requests.Timeout as e:
            last_error = e
            if attempt < MAX_RETRIES:
                print(f"Timeout on attempt {attempt + 1}. Retrying in {RETRY_DELAY}s...")
                time.sleep(RETRY_DELAY)
            else:
                print(f"All {MAX_RETRIES + 1} attempts failed.")
        except requests.ConnectionError as e:
            last_error = e
            if attempt < MAX_RETRIES:
                print(f"Connection error on attempt {attempt + 1}. Retrying in {RETRY_DELAY}s...")
                time.sleep(RETRY_DELAY)
            else:
                print(f"All {MAX_RETRIES + 1} attempts failed.")
        except Exception as e:
            last_error = e
            # Don't retry on non-timeout/connection errors
            raise RuntimeError(f"Error calling Ollama: {e}")
    
    # All retries exhausted
    raise RuntimeError(
        f"Ollama request failed after {MAX_RETRIES + 1} attempts. "
        f"Last error: {str(last_error)}. Try with a smaller model or check your connection."
    )


def list_models_info() -> None:
    """Display information about available models and resource requirements."""
    print("\n" + "="*80)
    print("Ollama Model Selection Guide")
    print("="*80)
    
    for level, config in MODELS_CONFIG.items():
        print(f"\n{level.upper()}")
        print(f"Description: {config['description']}")
        print(f"Recommended models: {', '.join(config['models'])}")
    
    print("\n" + "="*80)
    print("Available models installed:")
    available = get_available_models()
    if available:
        for model in available:
            print(f"  - {model}")
    else:
        print("  No models found. Install with: ollama pull <model>")
    print("="*80 + "\n")
