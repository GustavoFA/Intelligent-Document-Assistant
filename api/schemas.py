"""
Pydantic schemas for API requests and responses.
"""

from typing import Optional, List, Dict
from pydantic import BaseModel, Field


class ContextChunk(BaseModel):
    """Schema for a single retrieved context chunk."""
    rank: int = Field(..., description="Rank of the chunk")
    distance: float = Field(..., description="L2 distance score (lower is better)")
    doc_id: int | str = Field(..., description="Document ID")
    chunk_id: int = Field(..., description="Chunk ID within document")
    filename: str = Field(..., description="Source filename")
    text: str = Field(..., description="Chunk text content")


class SearchRequest(BaseModel):
    """Schema for semantic search request."""
    query: str = Field(..., description="Search query in PT-BR", min_length=1, max_length=500)
    top_k: int = Field(5, description="Number of results to retrieve", ge=1, le=20)


class SearchResponse(BaseModel):
    """Schema for semantic search response."""
    query: str = Field(..., description="Original search query")
    results: List[ContextChunk] = Field(..., description="Retrieved context chunks")
    count: int = Field(..., description="Number of results returned")


class RAGRequest(BaseModel):
    """Schema for RAG generation request."""
    user_prompt: str = Field(
        ..., 
        description="User's question in PT-BR", 
        min_length=1, 
        max_length=500
    )
    context: Optional[List[Dict]] = Field(
        None, 
        description="Pre-retrieved context (optional, auto-retrieves if None)"
    )
    user_questions: Optional[List[str]] = Field(
        None, 
        description="Additional clarifying questions"
    )
    model: Optional[str] = Field(
        None, 
        description="Ollama model to use (auto-selected if None)"
    )
    resource_level: str = Field(
        "medium_resource",
        description="Resource constraint level",
        pattern="^(high_resource|medium_resource|low_resource)$"
    )
    temperature: float = Field(
        0.7, 
        description="LLM temperature (0.0-1.0)", 
        ge=0.0, 
        le=1.0
    )
    top_k: int = Field(
        5, 
        description="Number of context chunks to retrieve", 
        ge=1, 
        le=20
    )


class RAGResponse(BaseModel):
    """Schema for RAG generation response."""
    user_prompt: str = Field(..., description="Original user prompt")
    response: str = Field(..., description="Generated response")
    context_count: Optional[int] = Field(None, description="Number of context chunks used")
    model_used: Optional[str] = Field(None, description="Model used for generation")


class ModelInfo(BaseModel):
    """Schema for model information."""
    name: str = Field(..., description="Model name")
    size: Optional[str] = Field(None, description="Model size/parameters")
    availability: str = Field(..., description="Whether model is available")


class ModelsResponse(BaseModel):
    """Schema for available models response."""
    high_resource: List[str] = Field(..., description="Models for high resource systems")
    medium_resource: List[str] = Field(..., description="Models for medium resource systems")
    low_resource: List[str] = Field(..., description="Models for low resource systems")
    available_models: List[str] = Field(..., description="Currently installed models")


class HealthResponse(BaseModel):
    """Schema for health check response."""
    status: str = Field(..., description="Service status")
    ollama_available: bool = Field(..., description="Whether Ollama is running")
    faiss_index_available: bool = Field(..., description="Whether FAISS index is available")
    chunking_data_available: bool = Field(..., description="Whether chunks data is available")


class ErrorResponse(BaseModel):
    """Schema for error responses."""
    error: str = Field(..., description="Error message")
    detail: Optional[str] = Field(None, description="Additional error details")
