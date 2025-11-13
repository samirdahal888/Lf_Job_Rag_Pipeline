#!/usr/bin/env python3
"""
Main entry point for LF Jobs RAG API
Simple wrapper to start the FastAPI server
"""

import os
import sys

import uvicorn


def main():
    """
    Start the LF Jobs RAG API server
    """
    # Print startup banner
    print("\n" + "=" * 80)
    print("🚀 LF Jobs RAG API - Starting Server")
    print("=" * 80)
    print()
    print("📋 Quick Info:")
    print("   • API Server: http://localhost:8000")
    print("   • Interactive Docs: http://localhost:8000/docs")
    print("   • ReDoc: http://localhost:8000/redoc")
    print("   • Health Check: http://localhost:8000/api/health")
    print()
    print("💡 Press CTRL+C to stop the server")
    print("=" * 80)
    print()

    # Check if .env file exists
    if not os.path.exists(".env"):
        print("⚠️  WARNING: .env file not found!")
        print("   Please create a .env file with your GEMINI_API_KEY")
        print()

    # Check if database is populated
    if not os.path.exists("chroma_db"):
        print("⚠️  WARNING: Vector database not found!")
        print("   Please run: python scripts/setup_database.py --use-local --force")
        print()

    # Start the server
    try:
        uvicorn.run(
            "src.api:app",
            host="0.0.0.0",
            port=8000,
            reload=True,  # Auto-reload on code changes
            log_level="info",
        )
    except KeyboardInterrupt:
        print("\n" + "=" * 80)
        print("👋 LF Jobs RAG API - Server Stopped")
        print("=" * 80)
        print()
        sys.exit(0)


if __name__ == "__main__":
    main()
