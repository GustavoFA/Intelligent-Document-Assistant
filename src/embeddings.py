
import logging 
import numpy as np
from typing import List, Dict

from sentence_transformers import SentenceTransformer
from config import EMBEDDING_MODEL_NAME, EMBEDDING_BATCH_SIZE

logger = logging.getLogger(__name__)

"""
Embedding generation and management for PT-BR legal documents.

This module handles:
1. Loading pretrained multilingual embedding models
2. Generating embeddings for document chunks
3. Encoding query strings into embeddings
"""


def load_model(model_name: str = EMBEDDING_MODEL_NAME) -> SentenceTransformer:
    """
    Load pretrained embedding model.
    
    Args:
        model_name: Name of the model from Hugging Face
        
    Returns:
        Loaded SentenceTransformer model
    """
    logger.info(f"\nLoading model: {model_name}")
    model = SentenceTransformer(model_name)
    logger.info("Model loaded successfully")
    return model


def generate_embeddings(
    chunks: List[Dict],
    model: SentenceTransformer,
    batch_size: int = EMBEDDING_BATCH_SIZE
) -> np.ndarray:
    """
    Generate embeddings for all document chunks.
    
    Args:
        chunks: List of chunk dictionaries with 'text' key
        model: SentenceTransformer model instance
        batch_size: Number of chunks to process per batch
        
    Returns:
        Array of embeddings with shape (num_chunks, embedding_dim)
    """
    logger.info("\nGenerating embeddings...")
    #BUG - E5 need prefixes
    # texts = [chunk["text"] for chunk in chunks]
    texts = [f"passage: {chunk['text']}" for chunk in chunks]

    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True
    ).astype("float32")

    logger.info(f"Generated embeddings with shape {embeddings.shape}")
    return embeddings


def encode_query(
    query: str,
    model: SentenceTransformer
) -> np.ndarray:
    """
    Encode a query string into an embedding vector.
    
    Args:
        query: Query text in PT-BR
        model: SentenceTransformer model instance
        
    Returns:
        Query embedding as float32 numpy array
    """
    query_embedding = model.encode([f"query: {query}"], convert_to_numpy=True).astype("float32")
    return query_embedding
