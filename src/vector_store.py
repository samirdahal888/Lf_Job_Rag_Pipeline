"""
Vector store module using ChromaDB
Handles storing and retrieving job embeddings with metadata filtering
"""

from pathlib import Path
from typing import Any, Dict, List, Optional

import chromadb
from chromadb.config import Settings

from src.config import config


class VectorStore:
    """
    Manages ChromaDB vector store for job embeddings
    Handles adding, querying, and managing job chunks with metadata
    """

    def __init__(self, persist_directory: Optional[str] = None):
        """
        Initialize ChromaDB client and collection

        Args:
            persist_directory: Directory to persist ChromaDB data
                             (defaults to config.CHROMA_DB_PATH)
        """
        self.persist_directory = persist_directory or config.CHROMA_DB_PATH

        # Create directory if it doesn't exist
        Path(self.persist_directory).mkdir(parents=True, exist_ok=True)

        # Initialize ChromaDB client with persistence
        self.client = chromadb.PersistentClient(
            path=self.persist_directory,
            settings=Settings(anonymized_telemetry=False, allow_reset=True),
        )

        # Get or create collection
        self.collection = self._get_or_create_collection()

        print(f"✅ Vector store initialized")
        print(f"   Collection: {config.COLLECTION_NAME}")
        print(f"   Path: {self.persist_directory}")
        print(f"   Total chunks: {self.collection.count()}")

    def _get_or_create_collection(self):
        """
        Get existing collection or create new one

        Returns:
            ChromaDB collection
        """
        try:
            # Try to get existing collection
            collection = self.client.get_collection(name=config.COLLECTION_NAME)
            print(f"📂 Loaded existing collection: {config.COLLECTION_NAME}")

        except Exception:
            # Create new collection if it doesn't exist
            collection = self.client.create_collection(
                name=config.COLLECTION_NAME, metadata=config.COLLECTION_METADATA
            )
            print(f"📂 Created new collection: {config.COLLECTION_NAME}")

        return collection

    def add_job_chunks(
        self,
        job_id: str,
        job_title: str,
        company: str,
        category: str,
        location: str,
        job_level: str,
        publication_date: str,
        tags: str,
        chunks: List[Dict[str, Any]],
    ) -> int:
        """
        Add all chunks for a single job to the vector store

        Args:
            job_id: Unique job ID
            job_title: Job title
            company: Company name
            category: Job category
            location: Job location
            job_level: Job level (Senior, Mid, Entry, Internship)
            publication_date: Publication date
            tags: Job tags
            chunks: List of chunk dictionaries with 'text', 'embedding', 'type', 'importance'

        Returns:
            Number of chunks added
        """
        if not chunks:
            print(f"⚠️  No chunks to add for job {job_id}")
            return 0

        # Prepare data for ChromaDB
        ids = []
        embeddings = []
        documents = []
        metadatas = []

        for i, chunk in enumerate(chunks):
            # Skip chunks without embeddings
            if chunk.get("embedding") is None:
                continue

            # Generate unique ID for this chunk
            chunk_id = f"{job_id}_chunk_{i}"
            ids.append(chunk_id)

            # Add embedding
            embeddings.append(chunk["embedding"])

            # Add document text
            documents.append(chunk["text"])

            # Add metadata (all searchable fields)
            metadata = {
                "job_id": str(job_id),
                "job_title": job_title,
                "company": company,
                "category": category,
                "location": location,
                "job_level": job_level,
                "publication_date": publication_date,
                "tags": tags,
                "chunk_type": chunk.get("type", "general"),
                "chunk_importance": chunk.get("importance", "medium"),
                "chunk_index": i,
            }
            metadatas.append(metadata)

        # Add to ChromaDB
        if ids:
            self.collection.add(
                ids=ids, embeddings=embeddings, documents=documents, metadatas=metadatas
            )

        return len(ids)

    def query(
        self,
        query_embedding: List[float],
        n_results: int = None,
        where: Optional[Dict[str, Any]] = None,
        where_document: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Query the vector store with an embedding and optional filters

        Args:
            query_embedding: Query embedding vector
            n_results: Number of results to return (defaults to config.TOP_K_RESULTS * RETRIEVAL_MULTIPLIER)
            where: Metadata filters (e.g., {"category": "Software Engineering"})
            where_document: Document content filters

        Returns:
            Dictionary with 'ids', 'embeddings', 'documents', 'metadatas', 'distances'
        """
        if n_results is None:
            n_results = config.TOP_K_RESULTS * config.RETRIEVAL_MULTIPLIER

        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results,
            where=where,
            where_document=where_document,
            include=["embeddings", "documents", "metadatas", "distances"],
        )

        return results

    def query_with_filters(
        self,
        query_embedding: List[float],
        category: Optional[str] = None,
        company: Optional[str] = None,
        location: Optional[str] = None,
        job_level: Optional[str] = None,
        n_results: int = None,
    ) -> Dict[str, Any]:
        """
        Query with common metadata filters

        Args:
            query_embedding: Query embedding vector
            category: Filter by job category
            company: Filter by company name
            location: Filter by job location
            job_level: Filter by job level
            n_results: Number of results to return

        Returns:
            Query results
        """
        # Build where clause using ChromaDB operators
        where = None
        conditions = []

        if category:
            conditions.append({"category": category})
        if company:
            conditions.append({"company": company})
        if location:
            conditions.append({"location": location})
        if job_level:
            conditions.append({"job_level": job_level})

        # ChromaDB requires $and operator for multiple filters
        if len(conditions) == 0:
            where = None
        elif len(conditions) == 1:
            where = conditions[0]
        else:
            where = {"$and": conditions}

        # Query with filters
        return self.query(
            query_embedding=query_embedding,
            n_results=n_results,
            where=where,
        )

    def get_chunk_by_id(self, chunk_id: str) -> Optional[Dict[str, Any]]:
        """
        Get a specific chunk by its ID

        Args:
            chunk_id: Chunk ID

        Returns:
            Dictionary with chunk data or None if not found
        """
        try:
            result = self.collection.get(
                ids=[chunk_id], include=["embeddings", "documents", "metadatas"]
            )

            if result["ids"]:
                return {
                    "id": result["ids"][0],
                    "embedding": result["embeddings"][0],
                    "document": result["documents"][0],
                    "metadata": result["metadatas"][0],
                }
            else:
                return None

        except Exception as e:
            print(f"❌ Error getting chunk {chunk_id}: {e}")
            return None

    def get_job_chunks(self, job_id: str) -> List[Dict[str, Any]]:
        """
        Get all chunks for a specific job

        Args:
            job_id: Job ID

        Returns:
            List of chunk dictionaries
        """
        try:
            results = self.collection.get(
                where={"job_id": str(job_id)},
                include=["embeddings", "documents", "metadatas"],
            )

            chunks = []
            for i in range(len(results["ids"])):
                chunks.append(
                    {
                        "id": results["ids"][i],
                        "embedding": results["embeddings"][i],
                        "document": results["documents"][i],
                        "metadata": results["metadatas"][i],
                    }
                )

            return chunks

        except Exception as e:
            print(f"❌ Error getting chunks for job {job_id}: {e}")
            return []

    def delete_job(self, job_id: str) -> int:
        """
        Delete all chunks for a specific job

        Args:
            job_id: Job ID to delete

        Returns:
            Number of chunks deleted
        """
        try:
            # Get all chunk IDs for this job
            chunks = self.get_job_chunks(job_id)
            chunk_ids = [chunk["id"] for chunk in chunks]

            if chunk_ids:
                self.collection.delete(ids=chunk_ids)
                print(f"🗑️  Deleted {len(chunk_ids)} chunks for job {job_id}")
                return len(chunk_ids)
            else:
                print(f"⚠️  No chunks found for job {job_id}")
                return 0

        except Exception as e:
            print(f"❌ Error deleting job {job_id}: {e}")
            return 0

    def clear_collection(self) -> bool:
        """
        Delete all data from the collection

        Returns:
            True if successful
        """
        try:
            # Delete the collection
            self.client.delete_collection(name=config.COLLECTION_NAME)

            # Recreate empty collection
            self.collection = self._get_or_create_collection()

            print(f"🗑️  Collection cleared: {config.COLLECTION_NAME}")
            return True

        except Exception as e:
            print(f"❌ Error clearing collection: {e}")
            return False

    def get_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the vector store

        Returns:
            Dictionary with statistics
        """
        total_chunks = self.collection.count()

        # Get all metadata to analyze
        if total_chunks > 0:
            all_data = self.collection.get(include=["metadatas"])
            metadatas = all_data["metadatas"]

            # Count unique jobs
            unique_jobs = set(m["job_id"] for m in metadatas)

            # Count by category
            categories = {}
            for m in metadatas:
                cat = m.get("category", "Unknown")
                categories[cat] = categories.get(cat, 0) + 1

            # Count by job level
            job_levels = {}
            for m in metadatas:
                level = m.get("job_level", "Unknown")
                job_levels[level] = job_levels.get(level, 0) + 1

            # Count by chunk type
            chunk_types = {}
            for m in metadatas:
                ctype = m.get("chunk_type", "Unknown")
                chunk_types[ctype] = chunk_types.get(ctype, 0) + 1

            return {
                "total_chunks": total_chunks,
                "unique_jobs": len(unique_jobs),
                "avg_chunks_per_job": total_chunks / len(unique_jobs)
                if unique_jobs
                else 0,
                "categories": categories,
                "job_levels": job_levels,
                "chunk_types": chunk_types,
            }
        else:
            return {
                "total_chunks": 0,
                "unique_jobs": 0,
                "avg_chunks_per_job": 0,
                "categories": {},
                "job_levels": {},
                "chunk_types": {},
            }

    def print_stats(self):
        """Print formatted statistics"""
        stats = self.get_stats()

        print("\n" + "=" * 60)
        print("📊 VECTOR STORE STATISTICS")
        print("=" * 60)

        print(f"\n📦 Overall:")
        print(f"   Total chunks: {stats['total_chunks']:,}")
        print(f"   Unique jobs: {stats['unique_jobs']:,}")
        print(f"   Avg chunks/job: {stats['avg_chunks_per_job']:.1f}")

        if stats["categories"]:
            print(f"\n📂 By Category:")
            for cat, count in sorted(stats["categories"].items(), key=lambda x: -x[1])[
                :5
            ]:
                print(f"   {cat}: {count:,} chunks")

        if stats["job_levels"]:
            print(f"\n🎯 By Job Level:")
            for level, count in sorted(
                stats["job_levels"].items(), key=lambda x: -x[1]
            ):
                print(f"   {level}: {count:,} chunks")

        if stats["chunk_types"]:
            print(f"\n📝 By Chunk Type:")
            for ctype, count in sorted(
                stats["chunk_types"].items(), key=lambda x: -x[1]
            ):
                print(f"   {ctype}: {count:,} chunks")

        print("=" * 60)


# Convenience functions
def get_vector_store() -> VectorStore:
    """
    Get a VectorStore instance

    Returns:
        VectorStore instance
    """
    return VectorStore()


# Testing and demonstration
if __name__ == "__main__":
    print("Testing VectorStore...\n")

    print("=" * 70)
    print("TEST 1: Initialize Vector Store")
    print("=" * 70)

    # Initialize
    store = VectorStore()

    # Get initial stats
    print("\nInitial state:")
    store.print_stats()

    print("\n" + "=" * 70)
    print("TEST 2: Add Sample Job Chunks")
    print("=" * 70)

    # Create sample chunks with fake embeddings
    sample_chunks = [
        {
            "text": "Senior Python Developer. Join our team to build scalable backend services.",
            "embedding": [0.1] * 768,  # Fake 768D embedding
            "type": "title_intro",
            "importance": "high",
        },
        {
            "text": "Requirements: 5+ years Python, Django, FastAPI, PostgreSQL, Docker",
            "embedding": [0.2] * 768,
            "type": "requirements",
            "importance": "high",
        },
        {
            "text": "Responsibilities: Design APIs, optimize database queries, mentor junior developers",
            "embedding": [0.3] * 768,
            "type": "responsibilities",
            "importance": "high",
        },
    ]

    # Add job chunks
    print("\nAdding sample job...")
    chunks_added = store.add_job_chunks(
        job_id="TEST_001",
        job_title="Senior Python Developer",
        company="Tech Corp",
        category="Software Engineering",
        location="San Francisco, CA",
        job_level="Senior Level",
        publication_date="2025-11-13",
        tags="Python, Django, FastAPI",
        chunks=sample_chunks,
    )

    print(f"✅ Added {chunks_added} chunks")

    print("\n" + "=" * 70)
    print("TEST 3: Query Vector Store")
    print("=" * 70)

    # Create a query embedding (fake)
    query_embedding = [0.15] * 768

    print("\nQuerying with fake embedding...")
    results = store.query(query_embedding=query_embedding, n_results=3)

    print(f"✅ Found {len(results['ids'][0])} results")

    # Show results
    for i, (chunk_id, doc, metadata, distance) in enumerate(
        zip(
            results["ids"][0],
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0],
        )
    ):
        print(f"\n  Result {i + 1}:")
        print(f"    ID: {chunk_id}")
        print(f"    Job: {metadata['job_title']}")
        print(f"    Type: {metadata['chunk_type']}")
        print(f"    Distance: {distance:.4f}")
        print(f"    Text: {doc[:80]}...")

    print("\n" + "=" * 70)
    print("TEST 4: Query with Filters")
    print("=" * 70)

    print("\nQuerying with category filter...")
    results = store.query_with_filters(
        query_embedding=query_embedding, category="Software Engineering", n_results=2
    )

    print(f"✅ Found {len(results['ids'][0])} results for 'Software Engineering'")

    print("\n" + "=" * 70)
    print("TEST 5: Get Job Chunks")
    print("=" * 70)

    print("\nGetting all chunks for job TEST_001...")
    job_chunks = store.get_job_chunks("TEST_001")

    print(f"✅ Found {len(job_chunks)} chunks")
    for i, chunk in enumerate(job_chunks, 1):
        print(
            f"  Chunk {i}: {chunk['metadata']['chunk_type']} - {chunk['document'][:60]}..."
        )

    print("\n" + "=" * 70)
    print("TEST 6: Get Statistics")
    print("=" * 70)

    store.print_stats()

    print("\n" + "=" * 70)
    print("TEST 7: Delete Job")
    print("=" * 70)

    print("\nDeleting job TEST_001...")
    deleted = store.delete_job("TEST_001")
    print(f"✅ Deleted {deleted} chunks")

    # Verify deletion
    print("\nAfter deletion:")
    store.print_stats()

    print("\n✅ All VectorStore tests complete!")
    print("\nNote: This test uses fake embeddings for demonstration.")
    print("Real embeddings will be added by the setup_database.py script.")
