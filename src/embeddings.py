"""
Embedding generation module using Google Gemini API or local models
Handles creating embeddings for both documents and queries
"""

import time
from typing import List, Optional

import google.generativeai as genai
from tqdm import tqdm

from src.config import config

# Try to import sentence-transformers for local fallback
try:
    from sentence_transformers import SentenceTransformer  # type: ignore

    HAS_SENTENCE_TRANSFORMERS = True
except ImportError:
    HAS_SENTENCE_TRANSFORMERS = False
    SentenceTransformer = None


class EmbeddingGenerator:
    """
    Handles embedding generation using Google Gemini API or local models
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        use_local: bool = False,
        local_model_name: str = "all-MiniLM-L6-v2",
    ):
        """
        Initialize embedding generator

        Args:
            api_key: Gemini API key (defaults to config.GEMINI_API_KEY)
            use_local: If True, use local sentence-transformers model instead of Gemini
            local_model_name: Name of the sentence-transformers model to use
        """
        self.use_local = use_local
        self.local_model = None
        self.local_model_name = local_model_name

        if use_local:
            if not HAS_SENTENCE_TRANSFORMERS:
                raise ImportError(
                    "sentence-transformers is required for local embeddings!\n"
                    "Install it with: pip install sentence-transformers torch"
                )
            print(f"🔄 Loading local embedding model: {local_model_name}")
            self.local_model = SentenceTransformer(local_model_name)
            self.dimension = self.local_model.get_sentence_embedding_dimension()
            print(f"✅ Local embedding model initialized")
            print(f"   Model: {local_model_name}")
            print(f"   Dimension: {self.dimension}")
        else:
            # Initialize Gemini
            self.api_key = api_key or config.GEMINI_API_KEY

            if not self.api_key or self.api_key == "your_gemini_api_key_here":
                raise ValueError(
                    "Gemini API key not configured!\n"
                    "Please set GEMINI_API_KEY in your .env file.\n"
                    "Get your key from: https://makersuite.google.com/app/apikey"
                )

            # Configure Gemini
            genai.configure(api_key=self.api_key)

            self.model = config.EMBEDDING_MODEL
            self.dimension = config.EMBEDDING_DIMENSION

            print("✅ Gemini Embedding API initialized")
            print(f"   Model: {self.model}")
            print(f"   Dimension: {self.dimension}")

    def create_embedding(
        self, text: str, task_type: str = "retrieval_document"
    ) -> Optional[List[float]]:
        """
        Create embedding for a single text

        Args:
            text: Text to embed
            task_type: "retrieval_document" for storing, "retrieval_query" for searching

        Returns:
            List of floats (embedding vector) or None if error
        """
        if not text or not text.strip():
            print("⚠️  Warning: Empty text provided for embedding")
            return None

        if self.use_local:
            return self._create_local_embedding(text)
        else:
            return self._create_gemini_embedding(text, task_type)

    def _create_local_embedding(self, text: str) -> Optional[List[float]]:
        """
        Create embedding using local sentence-transformers model

        Args:
            text: Text to embed

        Returns:
            Embedding vector or None if error
        """
        try:
            # Encode returns numpy array, convert to list
            embedding = self.local_model.encode(
                text, show_progress_bar=False, convert_to_numpy=True
            )
            return embedding.tolist()
        except Exception as e:
            print(f"❌ Error creating local embedding: {e}")
            return None

    def _create_gemini_embedding(
        self, text: str, task_type: str = "retrieval_document"
    ) -> Optional[List[float]]:
        """
        Create embedding using Gemini API

        Args:
            text: Text to embed
            task_type: "retrieval_document" or "retrieval_query"

        Returns:
            Embedding vector or None if error
        """

        try:
            result = genai.embed_content(
                model=self.model, content=text, task_type=task_type
            )

            embedding = result["embedding"]

            # Validate embedding dimension
            if len(embedding) != self.dimension:
                print(
                    f"⚠️  Warning: Expected {self.dimension}D embedding, got {len(embedding)}D"
                )

            return embedding

        except Exception as e:
            print(f"❌ Error creating embedding: {e}")
            return None

    def create_embeddings_batch(
        self,
        texts: List[str],
        task_type: str = "retrieval_document",
        show_progress: bool = True,
        delay: float = None,
    ) -> List[Optional[List[float]]]:
        """
        Create embeddings for multiple texts with rate limiting

        Args:
            texts: List of texts to embed
            task_type: "retrieval_document" or "retrieval_query"
            show_progress: Whether to show progress bar
            delay: Delay between requests (defaults to config.GEMINI_DELAY_BETWEEN_REQUESTS)

        Returns:
            List of embedding vectors (same length as input texts)
        """
        if self.use_local:
            # Use batch encoding for local model (much faster)
            return self._create_local_embeddings_batch(texts, show_progress)
        else:
            # Use Gemini API with rate limiting
            return self._create_gemini_embeddings_batch(
                texts, task_type, show_progress, delay
            )

    def _create_local_embeddings_batch(
        self, texts: List[str], show_progress: bool = True
    ) -> List[Optional[List[float]]]:
        """
        Create embeddings using local model in batch (fast)

        Args:
            texts: List of texts to embed
            show_progress: Whether to show progress bar

        Returns:
            List of embedding vectors
        """
        try:
            if show_progress:
                print(f"🔄 Creating {len(texts)} embeddings with local model...")

            # Batch encode is much faster than one-by-one
            embeddings_array = self.local_model.encode(
                texts,
                show_progress_bar=show_progress,
                convert_to_numpy=True,
                batch_size=32,  # Process in batches of 32
            )

            # Convert to list of lists
            embeddings = [emb.tolist() for emb in embeddings_array]

            if show_progress:
                print(
                    f"✅ Created {len(embeddings)}/{len(texts)} embeddings successfully"
                )

            return embeddings
        except Exception as e:
            print(f"❌ Error creating batch embeddings: {e}")
            # Fallback to one-by-one
            return [self._create_local_embedding(text) for text in texts]

    def _create_gemini_embeddings_batch(
        self,
        texts: List[str],
        task_type: str = "retrieval_document",
        show_progress: bool = True,
        delay: float = None,
    ) -> List[Optional[List[float]]]:
        """
        Create embeddings using Gemini API with rate limiting

        Args:
            texts: List of texts to embed
            task_type: "retrieval_document" or "retrieval_query"
            show_progress: Whether to show progress bar
            delay: Delay between requests

        Returns:
            List of embedding vectors
        """
        if delay is None:
            delay = config.GEMINI_DELAY_BETWEEN_REQUESTS

        embeddings = []

        # Use tqdm for progress bar if requested
        iterator = tqdm(texts, desc="Creating embeddings") if show_progress else texts

        for i, text in enumerate(iterator):
            # Create embedding
            embedding = self._create_gemini_embedding(text, task_type)
            embeddings.append(embedding)

            # Rate limiting (except for last item)
            if i < len(texts) - 1 and delay > 0:
                time.sleep(delay)

        # Count successful embeddings
        success_count = sum(1 for emb in embeddings if emb is not None)

        if show_progress:
            print(f"\n✅ Created {success_count}/{len(texts)} embeddings successfully")

        return embeddings

    def create_document_embedding(self, text: str) -> Optional[List[float]]:
        """
        Create embedding for storing a document

        Args:
            text: Document text

        Returns:
            Embedding vector
        """
        return self.create_embedding(
            text, task_type=config.EMBEDDING_TASK_TYPE_DOCUMENT
        )

    def create_query_embedding(self, text: str) -> Optional[List[float]]:
        """
        Create embedding for a search query

        Args:
            text: Query text

        Returns:
            Embedding vector
        """
        return self.create_embedding(text, task_type=config.EMBEDDING_TASK_TYPE_QUERY)

    def embed_job_chunks(
        self, chunks: List[dict], show_progress: bool = False
    ) -> List[dict]:
        """
        Create embeddings for job description chunks

        Args:
            chunks: List of chunk dictionaries (with 'text' key)
            show_progress: Whether to show progress

        Returns:
            List of chunks with added 'embedding' key
        """
        # Extract texts
        texts = [chunk["text"] for chunk in chunks]

        # Create embeddings
        embeddings = self.create_embeddings_batch(
            texts,
            task_type=config.EMBEDDING_TASK_TYPE_DOCUMENT,
            show_progress=show_progress,
        )

        # Add embeddings to chunks
        chunks_with_embeddings = []
        for chunk, embedding in zip(chunks, embeddings):
            chunk_copy = chunk.copy()
            chunk_copy["embedding"] = embedding
            chunks_with_embeddings.append(chunk_copy)

        return chunks_with_embeddings

    @staticmethod
    def validate_embedding(embedding: Optional[List[float]]) -> bool:
        """
        Validate that an embedding is valid

        Args:
            embedding: Embedding vector to validate

        Returns:
            True if valid, False otherwise
        """
        if embedding is None:
            return False

        if not isinstance(embedding, list):
            return False

        if len(embedding) != config.EMBEDDING_DIMENSION:
            return False

        # Check if all elements are numbers
        if not all(isinstance(x, (int, float)) for x in embedding):
            return False

        return True

    def test_connection(self) -> bool:
        """
        Test connection to Gemini API

        Returns:
            True if connection successful, False otherwise
        """
        print("\n🔍 Testing Gemini API connection...")

        test_text = "This is a test embedding."

        try:
            embedding = self.create_embedding(test_text)

            if embedding and len(embedding) == self.dimension:
                print("✅ Connection successful!")
                print(f"   Test embedding dimension: {len(embedding)}")
                print(f"   First 5 values: {embedding[:5]}")
                return True
            else:
                print("❌ Connection failed: Invalid embedding received")
                return False

        except Exception as e:
            print(f"❌ Connection failed: {e}")
            return False


# Convenience functions
def create_embedding(
    text: str, task_type: str = "retrieval_document"
) -> Optional[List[float]]:
    """
    Convenience function to create a single embedding

    Args:
        text: Text to embed
        task_type: Task type for embedding

    Returns:
        Embedding vector
    """
    generator = EmbeddingGenerator()
    return generator.create_embedding(text, task_type)


def create_document_embedding(text: str) -> Optional[List[float]]:
    """
    Convenience function to create document embedding

    Args:
        text: Document text

    Returns:
        Embedding vector
    """
    generator = EmbeddingGenerator()
    return generator.create_document_embedding(text)


def create_query_embedding(text: str) -> Optional[List[float]]:
    """
    Convenience function to create query embedding

    Args:
        text: Query text

    Returns:
        Embedding vector
    """
    generator = EmbeddingGenerator()
    return generator.create_query_embedding(text)


# Testing and demonstration
if __name__ == "__main__":
    print("Testing EmbeddingGenerator...\n")

    try:
        # Initialize
        print("=" * 60)
        print("TEST 1: Initialization")
        print("=" * 60)

        generator = EmbeddingGenerator()

        # Test connection
        print("\n" + "=" * 60)
        print("TEST 2: API Connection Test")
        print("=" * 60)

        connection_ok = generator.test_connection()

        if not connection_ok:
            print("\n⚠️  Cannot proceed with tests - API connection failed")
            exit(1)

        # Test single embedding
        print("\n" + "=" * 60)
        print("TEST 3: Single Text Embedding")
        print("=" * 60)

        test_text = (
            "Senior Python Developer with 5 years of experience in backend development"
        )
        print(f"\nText: '{test_text}'")

        embedding = generator.create_document_embedding(test_text)

        if embedding:
            print("✅ Embedding created successfully!")
            print(f"   Dimension: {len(embedding)}")
            print(f"   First 10 values: {embedding[:10]}")
            print(f"   Valid: {generator.validate_embedding(embedding)}")
        else:
            print("❌ Failed to create embedding")

        # Test batch embeddings
        print("\n" + "=" * 60)
        print("TEST 4: Batch Embeddings")
        print("=" * 60)

        test_texts = [
            "Data Scientist with machine learning expertise",
            "Frontend Developer skilled in React and TypeScript",
            "Project Manager with Agile experience",
        ]

        print(f"\nCreating embeddings for {len(test_texts)} texts...")

        embeddings = generator.create_embeddings_batch(
            test_texts, show_progress=True, delay=0.2
        )

        valid_count = sum(1 for emb in embeddings if generator.validate_embedding(emb))
        print(f"Valid embeddings: {valid_count}/{len(embeddings)}")

        # Test query vs document embeddings
        print("\n" + "=" * 60)
        print("TEST 5: Query vs Document Embeddings")
        print("=" * 60)

        text = "Python developer jobs"

        doc_emb = generator.create_document_embedding(text)
        query_emb = generator.create_query_embedding(text)

        print(f"\nText: '{text}'")
        print(f"Document embedding (first 5): {doc_emb[:5] if doc_emb else 'None'}")
        print(f"Query embedding (first 5): {query_emb[:5] if query_emb else 'None'}")

        if doc_emb and query_emb:
            # They should be different (different task types)
            are_same = doc_emb[:5] == query_emb[:5]
            print(f"Are they identical? {are_same}")

        # Test with job chunks
        print("\n" + "=" * 60)
        print("TEST 6: Job Chunks Embedding")
        print("=" * 60)

        sample_chunks = [
            {
                "text": "Senior Python Developer. Responsibilities include...",
                "type": "title_intro",
                "importance": "high",
            },
            {
                "text": "Requirements: 5+ years Python experience, Django, Flask...",
                "type": "requirements",
                "importance": "high",
            },
        ]

        print(f"\nEmbedding {len(sample_chunks)} job chunks...")

        chunks_with_embeddings = generator.embed_job_chunks(
            sample_chunks, show_progress=True
        )

        for i, chunk in enumerate(chunks_with_embeddings, 1):
            has_embedding = chunk.get("embedding") is not None
            print(
                f"  Chunk {i}: {chunk['type']} - Embedding: {'✅' if has_embedding else '❌'}"
            )

        print("\n✅ All EmbeddingGenerator tests complete!")

    except ValueError as e:
        print(f"\n❌ Error: {e}")
        print("\nPlease configure your Gemini API key in the .env file")
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
