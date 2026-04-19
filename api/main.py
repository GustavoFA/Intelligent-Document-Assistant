"""
FastAPI server for Intelligent Document Assistant (RAG System).

Provides REST API endpoints for:
- Semantic search (FAISS index)
- RAG generation (LLM with Ollama)
- Health checks
- Model management

Flow:
User → HTTP request → FastAPI → rag_generate() → response JSON
"""

import logging
from contextlib import asynccontextmanager
from typing import Optional, List, Dict

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from src.pipeline import rag_generate, retrieve_context
from src.config import MODELS_CONFIG
from src.llm import list_models_info

from api.schemas import (
    SearchRequest,
    SearchResponse,
    RAGRequest,
    RAGResponse,
    ContextChunk,
    ModelsResponse,
    HealthResponse,
    ErrorResponse,
)
from api.dependencies import (
    get_faiss_index,
    get_chunks,
    get_embedding_model,
    check_faiss_availability,
    check_ollama_status,
    get_available_ollama_models,
    clear_cache,
)

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# Lifespan context for startup/shutdown
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context for startup and shutdown events."""
    # Startup
    logger.info("Starting Intelligent Document Assistant API")
    logger.info(f"FAISS available: {check_faiss_availability()}")
    logger.info(f"Ollama available: {check_ollama_status()}")
    
    yield
    
    # Shutdown
    logger.info("Shutting down API")
    clear_cache()


# Create FastAPI app
app = FastAPI(
    title="Intelligent Document Assistant API",
    description="RAG system for Brazilian Portuguese legal documents",
    version="1.0.0",
    lifespan=lifespan,
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================================
# Welcome Endpoint
# ============================================================================

@app.get("/", tags=["Welcome"])
async def root():
    """Welcome endpoint with API information."""
    return {
        "message": "Intelligent Document Assistant API",
        "version": "1.0.0",
        "description": "RAG system for Brazilian Portuguese legal documents",
        "docs": "/docs",
        "endpoints": {
            "health": "/health",
            "models": "/models",
            "search": "/search",
            "rag": "/rag/generate",
        }
    }


# ============================================================================
# Health Check Endpoint
# ============================================================================

@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check():
    """Check system health and component availability."""
    try:
        faiss_ok = check_faiss_availability()
        ollama_ok = check_ollama_status()
        
        status = "healthy" if (faiss_ok and ollama_ok) else "degraded"
        
        return HealthResponse(
            status=status,
            faiss_index_available=faiss_ok,
            ollama_available=ollama_ok,
            chunking_data_available=faiss_ok,
        )
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return HealthResponse(
            status="unhealthy",
            faiss_index_available=False,
            ollama_available=False,
            chunking_data_available=False,
        )


# ============================================================================
# Models Endpoint
# ============================================================================

@app.get("/models", response_model=ModelsResponse, tags=["Models"])
async def list_models():
    """Get available models and resource configurations."""
    try:
        available = get_available_ollama_models()
        
        return ModelsResponse(
            high_resource=MODELS_CONFIG["high_resource"]["models"],
            medium_resource=MODELS_CONFIG["medium_resource"]["models"],
            low_resource=MODELS_CONFIG["low_resource"]["models"],
            available_models=available,
        )
    except Exception as e:
        logger.error(f"Error listing models: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================================
# Search Endpoint
# ============================================================================

@app.post("/search", response_model=SearchResponse, tags=["Search"])
async def search(request: SearchRequest):
    """
    Semantic search in FAISS index.
    
    Returns top-k relevant document chunks based on query similarity.
    """
    try:
        logger.info(f"Search query: {request.query}")
        
        # Validate FAISS is available
        if not check_faiss_availability():
            raise HTTPException(
                status_code=503,
                detail="FAISS index not available. Please run build_index.py first."
            )
        
        # Retrieve context
        context = retrieve_context(request.query, top_k=request.top_k)
        
        if not context:
            return SearchResponse(
                query=request.query,
                results=[],
                count=0,
            )
        
        # Convert to schema
        results = [ContextChunk(**chunk) for chunk in context]
        
        return SearchResponse(
            query=request.query,
            results=results,
            count=len(results),
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Search error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================================
# RAG Generation Endpoint
# ============================================================================

@app.post("/rag/generate", response_model=RAGResponse, tags=["RAG"])
async def generate_response(request: RAGRequest):
    """
    Generate response using RAG pipeline.
    
    Flow:
    1. Retrieve relevant context from FAISS (if not provided)
    2. Validate context quality
    3. Build prompt with system and user messages
    4. Call Ollama LLM
    5. Return generated response
    """
    try:
        logger.info(f"RAG request: {request.user_prompt[:50]}...")
        
        # Validate FAISS is available
        if not request.context and not check_faiss_availability():
            raise HTTPException(
                status_code=503,
                detail="FAISS index not available. Please run build_index.py first."
            )
        
        # Validate Ollama is available
        if not check_ollama_status():
            raise HTTPException(
                status_code=503,
                detail="Ollama service not available. Start it with: ollama serve"
            )
        
        # Generate response
        response = rag_generate(
            user_prompt=request.user_prompt,
            context=request.context,
            user_questions=request.user_questions,
            model=request.model,
            resource_level=request.resource_level,
            temperature=request.temperature,
            top_k=request.top_k,
        )
        
        # Count context chunks if used
        context_count = len(request.context) if request.context else None
        
        return RAGResponse(
            user_prompt=request.user_prompt,
            response=response,
            context_count=context_count,
            model_used=request.model,
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"RAG generation error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================================
# Advanced RAG Endpoint (with automatic retrieval)
# ============================================================================

@app.post("/rag/generate/auto", response_model=RAGResponse, tags=["RAG"])
async def generate_response_auto(
    query: str = Query(..., description="User question in PT-BR", min_length=1, max_length=500),
    top_k: int = Query(5, description="Number of context chunks to retrieve", ge=1, le=20),
    model: Optional[str] = Query(None, description="Ollama model to use"),
    resource_level: str = Query("medium_resource", description="Resource level"),
    temperature: float = Query(0.7, description="LLM temperature", ge=0.0, le=1.0),
):
    """
    Generate response with automatic context retrieval.
    
    Simpler endpoint using query parameters instead of request body.
    """
    try:
        logger.info(f"Auto RAG request: {query[:50]}...")
        
        # Validate services
        if not check_faiss_availability():
            raise HTTPException(
                status_code=503,
                detail="FAISS index not available. Please run build_index.py first."
            )
        
        if not check_ollama_status():
            raise HTTPException(
                status_code=503,
                detail="Ollama service not available. Start it with: ollama serve"
            )
        
        # Generate response (context auto-retrieved)
        response = rag_generate(
            user_prompt=query,
            context=None,  # Auto-retrieve
            model=model,
            resource_level=resource_level,
            temperature=temperature,
            top_k=top_k,
        )
        
        return RAGResponse(
            user_prompt=query,
            response=response,
            context_count=None,
            model_used=model,
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Auto RAG error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================================
# Error Handlers
# ============================================================================

@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    """Handle HTTP exceptions."""
    return JSONResponse(
        status_code=exc.status_code,
        content={"error": exc.detail},
    )


@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    """Handle general exceptions."""
    logger.error(f"Unhandled exception: {exc}")
    return JSONResponse(
        status_code=500,
        content={"error": "Internal server error", "detail": str(exc)},
    )


if __name__ == "__main__":
    import uvicorn
    
    # Run with: python -m uvicorn api.main:app --reload --host 0.0.0.0 --port 8000
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="info",
    )
