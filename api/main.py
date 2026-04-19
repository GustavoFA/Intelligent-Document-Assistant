"""
FastAPI server for Intelligent Document Assistant (RAG System).

Provides REST API endpoints for:
- Semantic search (FAISS index)
- RAG generation (LLM with Ollama)
- Health checks
- Model management
"""

import logging
from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from src.config import MODELS_CONFIG
from src.pipeline import rag_generate
from src.search import SearchEngine

from api.schemas import (
    SearchRequest,
    SearchResponse,
    RAGRequest,
    RAGResponse,
    ContextChunk,
    ModelsResponse,
    HealthResponse,
)
from api.dependencies import (
    get_search_engine,
    check_faiss_availability,
    check_ollama_status,
    get_available_ollama_models,
    clear_cache,
)

# -----------------------------------------------------------------------------
# Logging
# -----------------------------------------------------------------------------

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# -----------------------------------------------------------------------------
# Lifespan
# -----------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup/shutdown lifecycle for the API."""
    logger.info("Starting Intelligent Document Assistant API")

    try:
        faiss_ok = check_faiss_availability()
        ollama_ok = check_ollama_status()

        logger.info("FAISS available: %s", faiss_ok)
        logger.info("Ollama available: %s", ollama_ok)

        # Optional preload: initialize SearchEngine only if FAISS artifacts exist
        if faiss_ok:
            try:
                get_search_engine()
                logger.info("SearchEngine preloaded successfully")
            except Exception as e:
                logger.warning("SearchEngine preload failed: %s", e)

        yield

    finally:
        logger.info("Shutting down API")
        clear_cache()


# -----------------------------------------------------------------------------
# App
# -----------------------------------------------------------------------------

app = FastAPI(
    title="Intelligent Document Assistant API",
    description="RAG system for Brazilian Portuguese legal documents",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # restrict in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# -----------------------------------------------------------------------------
# Welcome
# -----------------------------------------------------------------------------

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
            "rag_generate": "/rag/generate",
            "rag_generate_auto": "/rag/generate/auto",
        },
    }


# -----------------------------------------------------------------------------
# Health
# -----------------------------------------------------------------------------

@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check():
    """Check system health and component availability."""
    try:
        faiss_ok = check_faiss_availability()
        ollama_ok = check_ollama_status()

        status = "healthy" if (faiss_ok and ollama_ok) else "degraded"

        return HealthResponse(
            status=status,
            ollama_available=ollama_ok,
            faiss_index_available=faiss_ok,
            chunking_data_available=faiss_ok,
        )
    except Exception as e:
        logger.error("Health check failed: %s", e)
        return HealthResponse(
            status="unhealthy",
            ollama_available=False,
            faiss_index_available=False,
            chunking_data_available=False,
        )


# -----------------------------------------------------------------------------
# Models
# -----------------------------------------------------------------------------

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
        logger.error("Error listing models: %s", e)
        raise HTTPException(status_code=500, detail=str(e))


# -----------------------------------------------------------------------------
# Search
# -----------------------------------------------------------------------------

@app.post("/search", response_model=SearchResponse, tags=["Search"])
async def search(
    request: SearchRequest,
    search_engine: SearchEngine = Depends(get_search_engine),
):
    """
    Semantic search in FAISS index.

    Returns top-k relevant document chunks based on vector similarity.
    """
    try:
        logger.info("Search query: %s", request.query)

        if not check_faiss_availability():
            raise HTTPException(
                status_code=503,
                detail="FAISS index not available. Please run build_index.py first.",
            )

        context = search_engine.retrieve(request.query, top_k=request.top_k)

        if not context:
            return SearchResponse(
                query=request.query,
                results=[],
                count=0,
            )

        results = [ContextChunk(**chunk) for chunk in context]

        return SearchResponse(
            query=request.query,
            results=results,
            count=len(results),
        )

    except HTTPException:
        raise
    except ValueError as e:
        logger.warning("Invalid search request: %s", e)
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error("Search error: %s", e)
        raise HTTPException(status_code=500, detail=str(e))


# -----------------------------------------------------------------------------
# RAG
# -----------------------------------------------------------------------------

@app.post("/rag/generate", response_model=RAGResponse, tags=["RAG"])
async def generate_response(
    request: RAGRequest,
    search_engine: SearchEngine = Depends(get_search_engine),
):
    """
    Generate a response using the full RAG pipeline.

    Flow:
    1. Retrieve relevant context from FAISS (if not provided)
    2. Validate context quality
    3. Build prompt
    4. Call Ollama
    5. Return generated response
    """
    try:
        logger.info("RAG request: %s...", request.user_prompt[:50])

        if request.context is None and not check_faiss_availability():
            raise HTTPException(
                status_code=503,
                detail="FAISS index not available. Please run build_index.py first.",
            )

        if not check_ollama_status():
            raise HTTPException(
                status_code=503,
                detail="Ollama service not available. Start it with: ollama serve",
            )

        response = rag_generate(
            user_prompt=request.user_prompt,
            search_engine=search_engine,
            context=request.context,
            user_questions=request.user_questions,
            model=request.model,
            resource_level=request.resource_level,
            temperature=request.temperature,
            top_k=request.top_k,
        )

        context_count = len(request.context) if request.context else None

        return RAGResponse(
            user_prompt=request.user_prompt,
            response=response,
            context_count=context_count,
            model_used=request.model,
        )

    except HTTPException:
        raise
    except ValueError as e:
        logger.warning("Invalid RAG request: %s", e)
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error("RAG generation error: %s", e)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/rag/generate/auto", response_model=RAGResponse, tags=["RAG"])
async def generate_response_auto(
    query: str = Query(..., description="User question in PT-BR", min_length=1, max_length=500),
    top_k: int = Query(5, description="Number of context chunks to retrieve", ge=1, le=20),
    model: str | None = Query(None, description="Ollama model to use"),
    resource_level: str = Query("medium_resource", description="Resource level"),
    temperature: float = Query(0.7, description="LLM temperature", ge=0.0, le=1.0),
    search_engine: SearchEngine = Depends(get_search_engine),
):
    """
    Generate a response with automatic context retrieval.

    Simpler endpoint using query parameters instead of request body.
    """
    try:
        logger.info("Auto RAG request: %s...", query[:50])

        if not check_faiss_availability():
            raise HTTPException(
                status_code=503,
                detail="FAISS index not available. Please run build_index.py first.",
            )

        if not check_ollama_status():
            raise HTTPException(
                status_code=503,
                detail="Ollama service not available. Start it with: ollama serve",
            )

        response = rag_generate(
            user_prompt=query,
            search_engine=search_engine,
            context=None,
            user_questions=None,
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
    except ValueError as e:
        logger.warning("Invalid auto RAG request: %s", e)
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error("Auto RAG error: %s", e)
        raise HTTPException(status_code=500, detail=str(e))


# -----------------------------------------------------------------------------
# Error handlers
# -----------------------------------------------------------------------------

@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    """Handle HTTP exceptions."""
    return JSONResponse(
        status_code=exc.status_code,
        content={"error": exc.detail},
    )


@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    """Handle unexpected exceptions."""
    logger.error("Unhandled exception: %s", exc)
    return JSONResponse(
        status_code=500,
        content={"error": "Internal server error", "detail": str(exc)},
    )


# -----------------------------------------------------------------------------
# Local run
# -----------------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "api.main:app",
        host="0.0.0.0",
        port=8000,
        reload=False,
        log_level="info",
    )