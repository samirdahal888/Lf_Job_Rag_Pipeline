# LF Jobs RAG Pipeline

**Intelligent Job Search System using Retrieval-Augmented Generation (RAG)**

This project implements a complete RAG pipeline for searching and retrieving job listings using semantic search, filters, and LLM-generated responses.

## 🎯 Features

- **Semantic Search**: Find jobs based on meaning, not just keywords
- **Hybrid Search**: Combines vector similarity with metadata filters
- **Smart Filtering**: Filter by location, company, job level, category, and date
- **LLM Responses**: Natural language responses powered by Google Gemini
- **RESTful API**: FastAPI-based endpoint for easy integration
- **Intelligent Chunking**: Smart text splitting for better search accuracy

## 🏗️ Architecture

```
User Query → Query Parser → Embedding Generation → Vector Search (+ Filters) 
→ Deduplication → LLM Response → JSON Result
```

## 📊 Technology Stack

- **Framework**: FastAPI
- **Embeddings**: Google Gemini Embedding API
- **Vector DB**: ChromaDB
- **LLM**: Google Gemini Pro
- **Text Processing**: LangChain, BeautifulSoup4
- **Data**: Pandas

## 🚀 Quick Start

### 1. Clone the Repository

```bash
git clone <repository-url>
cd GenAI_Takeaway_assignment
```

### 2. Create Virtual Environment

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Set Up Environment Variables

```bash
cp .env.example .env
# Edit .env and add your Gemini API key
```

Get your Gemini API key from: https://makersuite.google.com/app/apikey

### 5. Prepare Data

Place your job data CSV file in the `data/` directory as `lf_jobs.csv`

### 6. Setup Database (One-Time)

```bash
python scripts/setup_database.py
```

This will:
- Load and clean the job data
- Create embeddings for all jobs
- Store in ChromaDB vector database
- Takes ~10-15 minutes for 1000 jobs

### 7. Run the API

```bash
python main.py
```

The API will be available at: http://localhost:8000

## 📖 API Documentation

Once the server is running, visit:
# LF Jobs — RAG Pipeline

This repository contains a Retrieval-Augmented-Generation (RAG) pipeline for job search. It uses local Hugging Face sentence-transformers for embeddings, ChromaDB as the vector store, and a Gemini LLM for query parsing and natural language responses (when available). The project exposes a FastAPI REST API with interactive Swagger UI.

---

## Quick status
- Embeddings: local `all-MiniLM-L6-v2` (384D) — used by default (no quota limits)
- LLM (parsing + responses): `gemini-2.5-flash` (needs GEMINI_API_KEY; used sparingly)
- Vector DB: ChromaDB, path: `./chroma_db`
- Current indexed data: 2 jobs (30 chunks) — ready for full indexing

---

## Contents
- `src/` — application code (API, retriever, embeddings, preprocessing, vector store)
- `scripts/setup_database.py` — one-time job indexing script with checkpointing and rate limiter
- `data/lf_jobs.csv` — input spreadsheet (job listings)
- `chroma_db/` — persisted ChromaDB (created after indexing)
- `main.py` — convenient entry point to run the API
- `test_api.py` — small testing harness (optional)

---

## Prerequisites
- Linux/macOS/Windows WSL
- Python 3.10+ (project uses venv)
- Recommended: 4+ CPU cores; GPU optional but not required

---

## Install & Setup
From project root (`GenAI_Takeaway_assignment`):

```bash
# create & activate venv
python3 -m venv venv
source venv/bin/activate

# install dependencies
pip install -r requirements.txt
```

Create a `.env` file with your Gemini key (optional if you plan to use only local embeddings):

```text
GEMINI_API_KEY=your_gemini_api_key_here
CHROMA_DB_PATH=./chroma_db
```

Notes:
- If `GEMINI_API_KEY` is not present or invalid, the system will fall back to the local embedding model for indexing/search.

---

## Data Preparation
Place your exported spreadsheet at:

```
data/lf_jobs.csv
```

Expected columns (example):
- id, job_category, job_title, company_name, publication_date, job_location, job_level, tags, job_description

---

## How chunking works
- `src/preprocessing.py` uses LangChain's `RecursiveCharacterTextSplitter` with:
  - CHUNK_SIZE = 500 characters
  - CHUNK_OVERLAP = 50 characters
  - Separators (priority): `\n\n`, `\n`, `.`, `!`, `?`, `,`, ` `, ``
- Chunk 0: Title + first 300 characters of description (high importance)
- Later chunks: split the remaining description; chunk type auto-detected (responsibilities, requirements, benefits, general)

Reason: 500 chars (~100 words) gives good semantic context without diluting matching.

---

## Indexing (one-time)
If you're ready to index the whole CSV with the local model (recommended when Gemini embedding quota is exhausted):

```bash
# from repo root
source venv/bin/activate
python scripts/setup_database.py --use-local --force
```

Notes:
- `--use-local` : Use sentence-transformers local model (`all-MiniLM-L6-v2`) for embeddings
- `--force` : Rebuild (resets the checkpoint if present)
- The script supports checkpointing (JSON) and a rate-limiter for Gemini mode. If interrupted, re-run with the same flags and it will resume.

Expected run time: ~4 minutes for 1000 jobs (≈15,000 chunks) on CPU (batch encoding).

---

## Running the API
You can start the API either with `main.py` or directly with Uvicorn.

Option A (convenience wrapper):

```bash
source venv/bin/activate
python main.py
```

Option B (uvicorn):

```bash
source venv/bin/activate
uvicorn src.api:app --reload --host 0.0.0.0 --port 8000
```

Open interactive docs (Swagger UI):

```
http://localhost:8000/docs
```

Or ReDoc:

```
http://localhost:8000/redoc
```

Health endpoint:

```
GET /api/health
```

---

## Example API usage
### POST /api/query
Request body example (JSON):

```json
{
  "query": "senior Python developer in San Francisco",
  "top_k": 5,
  "response_type": "detailed",
  "filters": {
    "job_level": "Senior Level"
  }
}
```

Curl example:

```bash
curl -X POST http://localhost:8000/api/query \\
  -H "Content-Type: application/json" \\
  -d '{
    "query": "senior Python developer in San Francisco",
    "top_k": 5,
    "response_type": "detailed",
    "filters": {"job_level": "Senior Level"}
  }' | jq .
```

### Interactive testing (Swagger UI)
1. Open `http://localhost:8000/docs`
2. Expand `POST /api/query`
3. Click `Try it out`, paste JSON, `Execute`

---

## Under the hood (short)
- Query parsing: `src/query_parser.py` uses the Gemini LLM to extract:
  - `semantic_query` (the text to embed / semantically search)
  - `filters` (category, location, job_level, etc.)
- Embeddings: `src/embeddings.py` (local HF model or Gemini embedding-001 when enabled)
- Vector DB: `src/vector_store.py` (ChromaDB) stores per-chunk embeddings + metadata
- Retriever: `src/retriever.py` generates query embedding, calls `VectorStore.query_with_filters()` and deduplicates chunks into jobs
- Response generator: `src/llm_response.py` formats natural language responses using Gemini

---

## Configurable settings
See `src/config.py` for the main settings:
- `CHUNK_SIZE`, `CHUNK_OVERLAP`
- `TOP_K_RESULTS` & `RETRIEVAL_MULTIPLIER`
- `EMBEDDING_MODEL`, `EMBEDDING_DIMENSION` (Gemini defaults)
- `LLM_MODEL`, `LLM_TEMPERATURE`, `LLM_MAX_OUTPUT_TOKENS`

---

## Troubleshooting
- 404 / `TypeError: 'dict' object is not callable` in logs
  - This is caused by the ASGI/Starlette error handling when a middleware's error handler returns a raw dict. The server still works — check `/api/health` and `/docs`. We added robust handlers in `src/api.py` to return `JSONResponse`.

- Embedding dimension mismatch (0 results)
  - Ensure the same model is used for indexing and querying. Local model = 384D (all-MiniLM-L6-v2). If you indexed with Gemini (768D), queries using 384D embeddings will fail to match.

- Gemini quota exceeded
  - Use `--use-local` flag (already implemented). Re-embedding with Gemini is possible after quota resets.

- Server memory/killed issues
  - If uvicorn is killed due to memory, reduce parallelism or run without `--reload`.

---

## Files of interest
- `src/api.py` — FastAPI app and endpoints
- `src/preprocessing.py` — cleaning + chunking rules
- `src/embeddings.py` — wrapper for Gemini/local embeddings
- `src/vector_store.py` — ChromaDB wrapper (query_with_filters uses `$and` for multiple filters)
- `scripts/setup_database.py` — indexing script with checkpoint & rate-limiter
- `main.py` — quick entry-point to run the API

---

## Next steps / Checklist
1. (You) Review README and verify local instructions work on your machine
2. Run full indexing:

```bash
source venv/bin/activate
python scripts/setup_database.py --use-local --force
```

3. Run API and test via Swagger UI / `test_api.py` / curl
4. Add README to repo, finalize `.gitignore`, push to GitHub

---

## Contact
If anything fails, check `api.log` (if started with `nohup`) or the console running `uvicorn`. Share the last 50 lines of logs when asking for help.

---

Good luck — ready when you are to run full indexing and final tests! 🚀
