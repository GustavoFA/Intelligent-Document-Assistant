"""
Pydantic schemas for API requests and responses.
"""
from enum import Enum
from typing import Optional, List
from pydantic import BaseModel, Field


class ResourceLevel(str, Enum):
    high_resource = "high_resource"
    medium_resource = "medium_resource"
    low_resource = "low_resource"


class ContextChunk(BaseModel):
    rank: int = Field(..., description="Rank of the chunk")
    distance: float = Field(..., description="L2 distance score (lower is better)")
    doc_id: int | str = Field(..., description="Document ID")
    chunk_id: int = Field(..., description="Chunk ID within document")
    filename: str = Field(..., description="Source filename")
    text: str = Field(..., description="Chunk text content")


class SearchRequest(BaseModel):
    query: str = Field(..., description="Search query in PT-BR", min_length=1, max_length=500)
    top_k: int = Field(5, description="Number of results to retrieve", ge=1, le=20)


class SearchResponse(BaseModel):
    query: str = Field(..., description="Original search query")
    results: List[ContextChunk] = Field(..., description="Retrieved context chunks")
    count: int = Field(..., description="Number of results returned")


class RAGRequest(BaseModel):
    user_prompt: str = Field(..., description="User's question in PT-BR", min_length=1, max_length=500)
    context: Optional[List[ContextChunk]] = Field(
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
    resource_level: ResourceLevel = Field(
        ResourceLevel.medium_resource,
        description="Resource constraint level"
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

class ModelsResponse(BaseModel):
    high_resource: List[str] = Field(..., description="Models for high resource systems")
    medium_resource: List[str] = Field(..., description="Models for medium resource systems")
    low_resource: List[str] = Field(..., description="Models for low resource systems")
    available_models: List[str] = Field(..., description="Installed Ollama models")

class RAGResponse(BaseModel):
    user_prompt: str
    response: str
    context_count: Optional[int] = None
    model_used: Optional[str] = None

class HealthResponse(BaseModel):
    status: str
    ollama_available: bool
    faiss_index_available: bool
    chunking_data_available: bool