"""
Configuration management for the RAG system
Loads environment variables and defines all settings in one place
"""

import os
from pathlib import Path

from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()


class Config:
    """
    Centralized configuration for the LF Jobs RAG Pipeline
    All settings are defined here for easy modification
    """

    # ==================== API KEYS ====================
    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

    if not GEMINI_API_KEY or GEMINI_API_KEY == "your_gemini_api_key_here":
        print("⚠️  WARNING: GEMINI_API_KEY not set in .env file!")
        print("   Get your key from: https://makersuite.google.com/app/apikey")

    # ==================== PATHS ====================
    # Base directory (project root)
    BASE_DIR = Path(__file__).parent.parent

    # Data paths
    DATA_DIR = BASE_DIR / "data"
    DATA_PATH = DATA_DIR / "lf_jobs.csv"

    # ChromaDB path
    CHROMA_DB_PATH = os.getenv("CHROMA_DB_PATH", str(BASE_DIR / "chroma_db"))

    # ==================== EMBEDDING SETTINGS ====================
    # Gemini Embedding Model
    EMBEDDING_MODEL = "models/embedding-001"
    EMBEDDING_DIMENSION = 768  # Gemini embedding-001 produces 768-dimensional vectors
    EMBEDDING_TASK_TYPE_DOCUMENT = "retrieval_document"  # For storing documents
    EMBEDDING_TASK_TYPE_QUERY = "retrieval_query"  # For search queries

    # ==================== LLM SETTINGS ====================
    # Gemini for response generation
    LLM_MODEL = "gemini-2.5-flash"  # Stable Gemini 2.5 Flash
    LLM_TEMPERATURE = 0.3  # Lower = more focused/deterministic, Higher = more creative
    LLM_MAX_OUTPUT_TOKENS = 1024  # Maximum tokens in LLM response

    # ==================== CHROMADB SETTINGS ====================
    COLLECTION_NAME = "lf_jobs"
    COLLECTION_METADATA = {"description": "LF Jobs RAG System - Job Listings"}

    # ==================== CHUNKING SETTINGS ====================
    # Text splitting parameters for LangChain
    CHUNK_SIZE = 500  # Maximum characters per chunk
    CHUNK_OVERLAP = 50  # Overlap between chunks to preserve context

    # Chunk separators (in order of preference)
    CHUNK_SEPARATORS = [
        "\n\n",  # Paragraph breaks
        "\n",  # Line breaks
        ".",  # Sentences
        "!",  # Exclamations
        "?",  # Questions
        ",",  # Commas
        " ",  # Spaces
        "",  # Characters
    ]

    # ==================== RETRIEVAL SETTINGS ====================
    # Number of results to return to user
    TOP_K_RESULTS = 5

    # Retrieval multiplier (fetch more for deduplication)
    # We fetch TOP_K_RESULTS * RETRIEVAL_MULTIPLIER chunks, then deduplicate
    RETRIEVAL_MULTIPLIER = 4  # Fetch 20 chunks, deduplicate to 5 jobs

    # Similarity threshold (optional - can filter low-quality results)
    MIN_SIMILARITY_SCORE = 0.0  # 0-1, lower = more permissive

    # ==================== API SETTINGS ====================
    # FastAPI server configuration
    API_HOST = os.getenv("API_HOST", "0.0.0.0")
    API_PORT = int(os.getenv("API_PORT", 8000))
    API_TITLE = "LF Jobs RAG API"
    API_DESCRIPTION = "Intelligent job search using Retrieval-Augmented Generation"
    API_VERSION = "1.0.0"

    # CORS settings (for frontend integration if needed)
    CORS_ORIGINS = ["*"]  # Allow all origins in development

    # ==================== RATE LIMITING ====================
    # Gemini API rate limits (free tier)
    GEMINI_REQUESTS_PER_MINUTE = 60
    GEMINI_DELAY_BETWEEN_REQUESTS = 0.1  # seconds

    # ==================== LOGGING SETTINGS ====================
    LOG_LEVEL = "INFO"  # DEBUG, INFO, WARNING, ERROR, CRITICAL
    LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

    # ==================== DATA VALIDATION ====================
    # Required columns in the CSV file
    REQUIRED_COLUMNS = [
        "ID",
        "Job Category",
        "Job Title",
        "Company Name",
        "Publication Date",
        "Job Location",
        "Job Level",
        "Tags",
        "Job Description",
    ]

    # ==================== QUERY PARSING ====================
    # Known job levels for query parsing
    JOB_LEVELS = ["Senior Level", "Mid Level", "Entry Level", "Internship"]

    # Job level keyword mappings
    JOB_LEVEL_KEYWORDS = {
        "senior": "Senior Level",
        "sr": "Senior Level",
        "lead": "Senior Level",
        "principal": "Senior Level",
        "staff": "Senior Level",
        "mid level": "Mid Level",
        "mid": "Mid Level",
        "intermediate": "Mid Level",
        "junior": "Entry Level",
        "jr": "Entry Level",
        "entry": "Entry Level",
        "entry level": "Entry Level",
        "intern": "Internship",
        "internship": "Internship",
    }

    # Date filter keywords
    DATE_FILTER_KEYWORDS = {
        "today": 0,
        "yesterday": 1,
        "last week": 7,
        "past week": 7,
        "last month": 30,
        "past month": 30,
        "recent": 7,
    }

    # ==================== HELPER METHODS ====================
    @classmethod
    def validate(cls):
        """Validate configuration settings"""
        errors = []

        # Check if data file exists
        if not cls.DATA_PATH.exists():
            errors.append(f"Data file not found: {cls.DATA_PATH}")

        # Check API key
        if not cls.GEMINI_API_KEY or cls.GEMINI_API_KEY == "your_gemini_api_key_here":
            errors.append("GEMINI_API_KEY not configured in .env file")

        # Check chunk settings
        if cls.CHUNK_SIZE <= cls.CHUNK_OVERLAP:
            errors.append(
                f"CHUNK_SIZE ({cls.CHUNK_SIZE}) must be greater than CHUNK_OVERLAP ({cls.CHUNK_OVERLAP})"
            )

        if errors:
            print("\n❌ Configuration Validation Errors:")
            for error in errors:
                print(f"   - {error}")
            return False

        print("✅ Configuration validated successfully!")
        return True

    @classmethod
    def display(cls):
        """Display current configuration (for debugging)"""
        print("\n" + "=" * 60)
        print("LF JOBS RAG PIPELINE - CONFIGURATION")
        print("=" * 60)
        print(f"📁 Data Path: {cls.DATA_PATH}")
        print(f"🗄️  ChromaDB Path: {cls.CHROMA_DB_PATH}")
        print(f"🤖 LLM Model: {cls.LLM_MODEL}")
        print(f"🔢 Embedding Model: {cls.EMBEDDING_MODEL}")
        print(f"📊 Embedding Dimension: {cls.EMBEDDING_DIMENSION}")
        print(f"✂️  Chunk Size: {cls.CHUNK_SIZE}")
        print(f"🔗 Chunk Overlap: {cls.CHUNK_OVERLAP}")
        print(f"🎯 Top-K Results: {cls.TOP_K_RESULTS}")
        print(f"🌐 API Host: {cls.API_HOST}:{cls.API_PORT}")
        print(
            f"🔑 API Key Configured: {'Yes' if cls.GEMINI_API_KEY and cls.GEMINI_API_KEY != 'your_gemini_api_key_here' else 'No'}"
        )
        print("=" * 60 + "\n")


# Create a global config instance
config = Config()

# Validate on import (optional - can be disabled)
if __name__ == "__main__":
    config.display()
    config.validate()
