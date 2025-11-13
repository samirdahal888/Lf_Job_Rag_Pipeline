"""
FastAPI application for the LF Jobs RAG system
Provides REST API endpoints for job search
"""

from datetime import datetime
from typing import Any, Dict, List, Optional

import uvicorn
from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from src.config import config
from src.llm_response import ResponseGenerator
from src.retriever import JobRetriever

# ==================== REQUEST/RESPONSE MODELS ====================


class QueryRequest(BaseModel):
    """Request model for job search query"""

    query: str = Field(
        ...,
        min_length=1,
        max_length=500,
        description="Natural language job search query",
        example="senior Python developer in San Francisco",
    )
    top_k: Optional[int] = Field(
        default=5,
        ge=1,
        le=50,
        description="Number of results to return (1-50)",
        example=5,
    )
    response_type: Optional[str] = Field(
        default="detailed",
        description="Response format: detailed, summary, or brief",
        example="detailed",
    )
    filters: Optional[Dict[str, str]] = Field(
        default=None,
        description="Optional manual filters to apply",
        example={"job_level": "Senior Level"},
    )


class JobResult(BaseModel):
    """Job result model"""

    rank: int
    job_id: str
    job_title: str
    company: str
    category: str
    location: str
    job_level: str
    publication_date: str
    tags: str
    relevance_score: float
    relevance_label: str
    similarity_percentage: float
    matched_chunks: int
    snippet: str


class QueryResponse(BaseModel):
    """Response model for job search query"""

    success: bool
    query: str
    filters_applied: Dict[str, Any]
    total_results: int
    response: str
    jobs: List[JobResult]
    timestamp: str


class HealthResponse(BaseModel):
    """Health check response"""

    status: str
    version: str
    components: Dict[str, str]
    timestamp: str


class StatsResponse(BaseModel):
    """Statistics response"""

    total_jobs_indexed: int
    total_chunks_indexed: int
    avg_chunks_per_job: float
    categories: List[str]
    job_levels: List[str]
    top_k_results: int


# ==================== FASTAPI APPLICATION ====================

app = FastAPI(
    title=config.API_TITLE,
    description=config.API_DESCRIPTION,
    version=config.API_VERSION,
    docs_url="/docs",
    redoc_url="/redoc",
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=config.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize components (singleton pattern)
retriever: Optional[JobRetriever] = None
response_generator: Optional[ResponseGenerator] = None


def get_retriever() -> JobRetriever:
    """Get or create JobRetriever instance"""
    global retriever
    if retriever is None:
        # Use local embeddings if database was built with local model
        # TODO: Make this configurable via environment variable
        retriever = JobRetriever(use_local_embeddings=True)
    return retriever


def get_response_generator() -> ResponseGenerator:
    """Get or create ResponseGenerator instance"""
    global response_generator
    if response_generator is None:
        response_generator = ResponseGenerator()
    return response_generator


# ==================== API ENDPOINTS ====================


@app.get("/", tags=["Root"])
async def root():
    """Root endpoint - API information"""
    return {
        "message": "LF Jobs RAG API",
        "version": config.API_VERSION,
        "description": "Intelligent job search using Retrieval-Augmented Generation",
        "endpoints": {
            "POST /api/query": "Search for jobs with natural language",
            "GET /api/health": "Health check",
            "GET /api/stats": "System statistics",
            "GET /docs": "Interactive API documentation",
            "GET /redoc": "Alternative API documentation",
        },
    }


@app.get("/api/health", response_model=HealthResponse, tags=["System"])
async def health_check():
    """
    Health check endpoint

    Returns system health status and component availability
    """
    try:
        # Check if components can be initialized
        ret = get_retriever()
        _ = get_response_generator()  # Initialize to verify it works

        stats = ret.get_statistics()

        return HealthResponse(
            status="healthy",
            version=config.API_VERSION,
            components={
                "retriever": "operational",
                "response_generator": "operational",
                "vector_store": f"{stats['total_jobs_indexed']} jobs indexed",
                "embeddings": "operational",
                "query_parser": "operational",
            },
            timestamp=datetime.now().isoformat(),
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Service unhealthy: {str(e)}",
        )


@app.get("/api/stats", response_model=StatsResponse, tags=["System"])
async def get_statistics():
    """
    Get system statistics

    Returns information about indexed jobs and configuration
    """
    try:
        ret = get_retriever()
        stats = ret.get_statistics()

        return StatsResponse(
            total_jobs_indexed=stats["total_jobs_indexed"],
            total_chunks_indexed=stats["total_chunks_indexed"],
            avg_chunks_per_job=round(stats["avg_chunks_per_job"], 2),
            categories=stats["categories"],
            job_levels=stats["job_levels"],
            top_k_results=stats["top_k_results"],
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get statistics: {str(e)}",
        )


@app.post("/api/query", response_model=QueryResponse, tags=["Search"])
async def query_jobs(request: QueryRequest):
    """
    Search for jobs using natural language query

    This is the main endpoint for the RAG system. It:
    1. Parses the natural language query
    2. Extracts filters and semantic intent
    3. Searches the vector database
    4. Ranks and deduplicates results
    5. Generates a natural language response

    **Example queries:**
    - "Python developer jobs"
    - "senior software engineer in San Francisco"
    - "entry level data scientist at Google"
    - "UX designer positions, remote"
    """
    try:
        # Validate response type
        if request.response_type not in ["detailed", "summary", "brief"]:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="response_type must be 'detailed', 'summary', or 'brief'",
            )

        # Get components
        ret = get_retriever()
        gen = get_response_generator()

        # Perform search
        results = ret.retrieve(
            query=request.query, top_k=request.top_k, filters=request.filters
        )

        # Generate natural language response
        nl_response = gen.generate_response(
            results, response_type=request.response_type
        )

        # Format job results
        job_results = []
        for job in results["jobs"]:
            job_results.append(
                JobResult(
                    rank=job["rank"],
                    job_id=job["job_id"],
                    job_title=job["job_title"],
                    company=job["company"],
                    category=job["category"],
                    location=job["location"],
                    job_level=job["job_level"],
                    publication_date=job["publication_date"],
                    tags=job["tags"],
                    relevance_score=round(job["relevance_score"], 4),
                    relevance_label=job["relevance_label"],
                    similarity_percentage=job["similarity_percentage"],
                    matched_chunks=job["matched_chunks"],
                    snippet=job["snippet"],
                )
            )

        # Build response
        return QueryResponse(
            success=True,
            query=request.query,
            filters_applied=results["query_info"]["filters"],
            total_results=len(job_results),
            response=nl_response,
            jobs=job_results,
            timestamp=datetime.now().isoformat(),
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Query failed: {str(e)}",
        )


@app.get("/api/job/{job_id}", tags=["Search"])
async def get_job_details(job_id: str):
    """
    Get detailed information for a specific job

    Returns complete job information including all chunks
    """
    try:
        ret = get_retriever()
        job = ret.retrieve_by_id(job_id)

        if not job:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Job with ID '{job_id}' not found",
            )

        # Generate detailed description
        gen = get_response_generator()
        description = gen.generate_job_detail_response(job)

        return {
            "success": True,
            "job": job,
            "description": description,
            "timestamp": datetime.now().isoformat(),
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get job details: {str(e)}",
        )


# ==================== ERROR HANDLERS ====================


@app.exception_handler(404)
async def not_found_handler(request, exc):
    """Custom 404 handler"""
    return {
        "success": False,
        "error": "Not Found",
        "message": "The requested endpoint does not exist",
        "available_endpoints": [
            "POST /api/query",
            "GET /api/health",
            "GET /api/stats",
            "GET /api/job/{job_id}",
        ],
    }


@app.exception_handler(500)
async def internal_error_handler(request, exc):
    """Custom 500 handler"""
    return {
        "success": False,
        "error": "Internal Server Error",
        "message": "An unexpected error occurred",
        "detail": str(exc),
    }


# ==================== STARTUP/SHUTDOWN ====================


@app.on_event("startup")
async def startup_event():
    """Initialize components on startup"""
    print("\n" + "=" * 70)
    print("🚀 Starting LF Jobs RAG API")
    print("=" * 70)

    try:
        # Initialize components
        print("\n📦 Initializing components...")
        ret = get_retriever()
        _ = get_response_generator()  # Initialize to verify it works

        # Get stats
        stats = ret.get_statistics()

        print("\n✅ API Ready!")
        print(f"   Jobs indexed: {stats['total_jobs_indexed']}")
        print(f"   Chunks indexed: {stats['total_chunks_indexed']}")
        print(f"   Server: http://{config.API_HOST}:{config.API_PORT}")
        print(f"   Docs: http://{config.API_HOST}:{config.API_PORT}/docs")
        print("=" * 70 + "\n")

    except Exception as e:
        print(f"\n❌ Failed to initialize: {e}")
        print("Make sure to run scripts/setup_database.py first!")
        print("=" * 70 + "\n")


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    print("\n" + "=" * 70)
    print("👋 Shutting down LF Jobs RAG API")
    print("=" * 70 + "\n")


# ==================== MAIN ====================

if __name__ == "__main__":
    print("Starting FastAPI server...")
    print(f"Open http://localhost:{config.API_PORT}/docs for interactive documentation")

    uvicorn.run(
        "src.api:app",
        host=config.API_HOST,
        port=config.API_PORT,
        reload=True,  # Auto-reload on code changes (development only)
        log_level="info",
    )
