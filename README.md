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
- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

### Example API Request

```bash
curl -X POST http://localhost:8000/api/query \
  -H "Content-Type: application/json" \
  -d '{
    "query": "Senior Python developer jobs in New York",
    "top_k": 5,
    "include_llm_response": true
  }'
```

### Example Response

```json
{
  "query": "Senior Python developer jobs in New York",
  "filters_applied": {
    "level": "Senior Level",
    "location": "New York, NY"
  },
  "results": [
    {
      "job_id": "LF0001",
      "job_title": "Senior Python Developer",
      "company": "Tech Corp",
      "location": "New York, NY",
      "level": "Senior Level",
      "category": "Software Development",
      "relevance_score": 0.95,
      "snippet": "We are seeking an experienced Python developer..."
    }
  ],
  "llm_response": "I found 5 senior Python developer positions in New York...",
  "total_results": 5
}
```

## 📁 Project Structure

```
GenAI_Takeaway_assignment/
├── data/
│   └── lf_jobs.csv              # Job dataset (CSV)
├── src/
│   ├── __init__.py
│   ├── config.py                # Configuration
│   ├── data_loader.py           # Data loading
│   ├── preprocessing.py         # HTML cleaning & chunking
│   ├── embeddings.py            # Gemini embeddings
│   ├── vector_store.py          # ChromaDB operations
│   ├── query_parser.py          # Query parsing
│   ├── retriever.py             # Search & retrieval
│   ├── llm_response.py          # LLM response generation
│   └── api.py                   # FastAPI endpoints
├── scripts/
│   └── setup_database.py        # Database setup script
├── tests/
│   └── test_api.py              # Unit tests
├── notebooks/
│   └── exploration.ipynb        # Data exploration
├── chroma_db/                   # Vector database (auto-created)
├── .env                         # Environment variables
├── .gitignore
├── requirements.txt
├── main.py                      # Entry point
└── README.md
```

## 🧪 Testing

```bash
# Run all tests
pytest

# Run specific test
pytest tests/test_api.py -v
```

## 🔧 Configuration

Edit `src/config.py` to customize:
- Chunk size and overlap
- Number of results returned
- LLM temperature
- Embedding model settings

## 📝 Development Notes

### How It Works

1. **Data Processing**: Job descriptions are cleaned (HTML removed) and split into semantic chunks
2. **Embedding Creation**: Each chunk is converted to a 768-dimensional vector using Gemini
3. **Vector Storage**: Embeddings are stored in ChromaDB with metadata (filters)
4. **Query Processing**: User queries are parsed to extract filters and semantic intent
5. **Hybrid Search**: ChromaDB searches using both vector similarity and metadata filters
6. **Deduplication**: Results are deduplicated by job_id, keeping the best match per job
7. **LLM Enhancement**: Gemini Pro generates a natural language response

### Key Design Decisions

- **ChromaDB**: Chosen for local deployment, easy setup, and built-in filtering
- **Gemini**: Free tier, high quality embeddings (768D), integrated LLM
- **LangChain**: Industry-standard text splitting with smart separators
- **Hybrid Search**: Filters applied during search (not after) for better performance

## 🚧 Assumptions

- Job descriptions are in English
- CSV file follows the specified format (9 columns)
- Gemini API is accessible and within rate limits

## ⚠️ Limitations

- Currently supports only document-level search (not real-time updates)
- Rate limited by Gemini API (1500 requests/day on free tier)
- Local ChromaDB (not suitable for distributed deployment without changes)

## 🔮 Future Enhancements

1. **Reranker Model**: Add cross-encoder for better result ranking
2. **Caching**: Cache frequent queries for faster responses
3. **Advanced Filters**: Salary range, experience years, skills matching
4. **Multi-language**: Support for non-English job descriptions
5. **Real-time Updates**: Incremental updates to vector database
6. **Production Deployment**: Migration to cloud-based vector DB (Pinecone, Weaviate)
7. **User Feedback**: Learning from user clicks/selections

## 👥 Author

Samir Dahal - Leapfrog Technology Assignment

## 📄 License

This project is created as part of a technical assessment.

## 🙏 Acknowledgments

- Leapfrog Technology for the opportunity
- Google for Gemini API
- ChromaDB and LangChain communities
