"""
Retriever module for hybrid search
Combines semantic search with metadata filtering and deduplication
"""

from collections import defaultdict
from typing import Any, Dict, List, Optional

from src.config import config
from src.embeddings import EmbeddingGenerator
from src.query_parser import QueryParser
from src.vector_store import VectorStore


class JobRetriever:
    """
    Retrieves relevant jobs using hybrid search:
    1. Parse query to extract filters and semantic intent
    2. Create embedding for semantic query
    3. Search vector store with filters
    4. Deduplicate chunks to unique jobs
    5. Rank and return top results
    """

    def __init__(
        self,
        vector_store: Optional[VectorStore] = None,
        embedding_generator: Optional[EmbeddingGenerator] = None,
        query_parser: Optional[QueryParser] = None,
        use_local_embeddings: bool = False,
    ):
        """
        Initialize the retriever with required components

        Args:
            vector_store: VectorStore instance (creates new if None)
            embedding_generator: EmbeddingGenerator instance (creates new if None)
            query_parser: QueryParser instance (creates new if None)
            use_local_embeddings: If True, use local model for query embeddings
        """
        self.vector_store = vector_store or VectorStore()
        self.embedding_generator = embedding_generator or EmbeddingGenerator(
            use_local=use_local_embeddings
        )
        self.query_parser = query_parser or QueryParser()

        print("✅ JobRetriever initialized")
        print(f"   Vector store: {self.vector_store.collection.count()} chunks")
        print(
            f"   Embedding mode: {'Local model' if use_local_embeddings else 'Gemini API'}"
        )
        print(f"   Top-K results: {config.TOP_K_RESULTS}")

    def retrieve(
        self,
        query: str,
        top_k: Optional[int] = None,
        filters: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Retrieve relevant jobs for a query

        Args:
            query: User's natural language query
            top_k: Number of jobs to return (defaults to config.TOP_K_RESULTS)
            filters: Optional manual filters to override/add to parsed filters

        Returns:
            Dictionary with:
                - jobs: List of job dictionaries
                - query_info: Parsed query information
                - search_info: Search metadata
        """
        if not query or not query.strip():
            return {
                "jobs": [],
                "query_info": {
                    "original_query": query,
                    "semantic_query": "",
                    "filters": {},
                },
                "search_info": {
                    "total_chunks": 0,
                    "unique_jobs": 0,
                    "filters_applied": {},
                },
            }

        top_k = top_k or config.TOP_K_RESULTS

        # Step 1: Parse query
        print(f'\n🔍 Parsing query: "{query}"')
        parsed = self.query_parser.parse_query(query)

        # Merge manual filters if provided
        if filters:
            parsed["filters"].update(filters)

        print(f'   Semantic query: "{parsed["semantic_query"]}"')
        print(f"   Filters: {parsed['filters']}")

        # Step 2: Create embedding for semantic search
        print("\n🧮 Creating query embedding...")
        query_embedding = self.embedding_generator.create_query_embedding(
            parsed["semantic_query"] or query
        )

        if not query_embedding:
            print("❌ Failed to create query embedding")
            return {
                "jobs": [],
                "query_info": parsed,
                "search_info": {"error": "Failed to create embedding"},
            }

        print(f"   Embedding dimension: {len(query_embedding)}")

        # Step 3: Search vector store
        print(f"\n🔎 Searching vector store...")
        n_results = top_k * config.RETRIEVAL_MULTIPLIER

        search_results = self.vector_store.query_with_filters(
            query_embedding=query_embedding,
            category=parsed["filters"].get("category"),
            company=parsed["filters"].get("company"),
            location=parsed["filters"].get("location"),
            job_level=parsed["filters"].get("job_level"),
            n_results=n_results,
        )

        total_chunks = len(search_results["ids"][0]) if search_results["ids"] else 0
        print(f"   Found {total_chunks} relevant chunks")

        # Step 4: Deduplicate and rank
        print("\n📊 Deduplicating to unique jobs...")
        jobs = self._deduplicate_and_rank(search_results, top_k=top_k)

        print(f"   Unique jobs: {len(jobs)}")

        # Step 5: Enrich job data
        jobs = self._enrich_jobs(jobs)

        # Build response
        return {
            "jobs": jobs,
            "query_info": {
                "original_query": query,
                "semantic_query": parsed["semantic_query"],
                "filters": parsed["filters"],
            },
            "search_info": {
                "total_chunks_found": total_chunks,
                "unique_jobs_found": len(jobs),
                "filters_applied": parsed["filters"],
                "requested_top_k": top_k,
            },
        }

    def _deduplicate_and_rank(
        self, search_results: Dict[str, Any], top_k: int
    ) -> List[Dict[str, Any]]:
        """
        Deduplicate chunks to unique jobs and rank by relevance

        Args:
            search_results: Raw search results from vector store
            top_k: Number of top jobs to return

        Returns:
            List of unique job dictionaries with scores
        """
        if not search_results["ids"] or not search_results["ids"][0]:
            return []

        # Group chunks by job_id
        jobs_dict = defaultdict(
            lambda: {
                "chunks": [],
                "min_distance": float("inf"),
                "avg_distance": 0.0,
                "chunk_count": 0,
                "metadata": None,
            }
        )

        for i, chunk_id in enumerate(search_results["ids"][0]):
            metadata = search_results["metadatas"][0][i]
            distance = search_results["distances"][0][i]
            document = search_results["documents"][0][i]

            job_id = metadata["job_id"]

            # Add chunk info
            jobs_dict[job_id]["chunks"].append(
                {
                    "text": document,
                    "distance": distance,
                    "type": metadata.get("chunk_type", "general"),
                    "importance": metadata.get("chunk_importance", "medium"),
                }
            )

            # Update distance metrics
            jobs_dict[job_id]["min_distance"] = min(
                jobs_dict[job_id]["min_distance"], distance
            )
            jobs_dict[job_id]["chunk_count"] += 1

            # Store metadata (same for all chunks of a job)
            if jobs_dict[job_id]["metadata"] is None:
                jobs_dict[job_id]["metadata"] = metadata

        # Calculate average distances and create job list
        jobs = []
        for job_id, job_data in jobs_dict.items():
            # Calculate average distance
            avg_distance = (
                sum(c["distance"] for c in job_data["chunks"]) / job_data["chunk_count"]
            )
            job_data["avg_distance"] = avg_distance

            # Calculate relevance score (lower distance = higher score)
            # Use weighted combination of min and avg distance
            relevance_score = 1 / (
                1 + (0.3 * job_data["min_distance"] + 0.7 * avg_distance)
            )

            job = {
                "job_id": job_id,
                "job_title": job_data["metadata"]["job_title"],
                "company": job_data["metadata"]["company"],
                "category": job_data["metadata"]["category"],
                "location": job_data["metadata"]["location"],
                "job_level": job_data["metadata"]["job_level"],
                "publication_date": job_data["metadata"]["publication_date"],
                "tags": job_data["metadata"]["tags"],
                "relevance_score": relevance_score,
                "min_distance": job_data["min_distance"],
                "avg_distance": avg_distance,
                "matched_chunks": job_data["chunk_count"],
                "top_chunks": sorted(job_data["chunks"], key=lambda x: x["distance"])[
                    :3
                ],  # Keep top 3 most relevant chunks
            }

            jobs.append(job)

        # Sort by relevance score (highest first)
        jobs.sort(key=lambda x: x["relevance_score"], reverse=True)

        # Return top K
        return jobs[:top_k]

    def _enrich_jobs(self, jobs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Enrich job data with additional computed fields

        Args:
            jobs: List of job dictionaries

        Returns:
            Enriched job list
        """
        for i, job in enumerate(jobs, 1):
            # Add rank
            job["rank"] = i

            # Add similarity percentage (inverse of distance)
            job["similarity_percentage"] = round(job["relevance_score"] * 100, 2)

            # Format relevance display
            if job["relevance_score"] >= 0.7:
                job["relevance_label"] = "Highly Relevant"
            elif job["relevance_score"] >= 0.5:
                job["relevance_label"] = "Relevant"
            elif job["relevance_score"] >= 0.3:
                job["relevance_label"] = "Moderately Relevant"
            else:
                job["relevance_label"] = "Somewhat Relevant"

            # Create snippet from top chunks
            chunk_texts = [c["text"] for c in job["top_chunks"]]
            job["snippet"] = " ... ".join(chunk_texts)[:500]

        return jobs

    def retrieve_by_id(self, job_id: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve complete information for a specific job

        Args:
            job_id: Job ID

        Returns:
            Job dictionary with all chunks or None if not found
        """
        chunks = self.vector_store.get_job_chunks(job_id)

        if not chunks:
            return None

        # Build job from chunks
        metadata = chunks[0]["metadata"]

        job = {
            "job_id": job_id,
            "job_title": metadata["job_title"],
            "company": metadata["company"],
            "category": metadata["category"],
            "location": metadata["location"],
            "job_level": metadata["job_level"],
            "publication_date": metadata["publication_date"],
            "tags": metadata["tags"],
            "total_chunks": len(chunks),
            "chunks": [
                {
                    "text": chunk["document"],
                    "type": chunk["metadata"]["chunk_type"],
                    "importance": chunk["metadata"]["chunk_importance"],
                }
                for chunk in chunks
            ],
        }

        # Reconstruct full description
        job["full_description"] = "\n\n".join(chunk["document"] for chunk in chunks)

        return job

    def get_statistics(self) -> Dict[str, Any]:
        """
        Get retriever statistics

        Returns:
            Statistics dictionary
        """
        vector_stats = self.vector_store.get_stats()

        return {
            "total_jobs_indexed": vector_stats["unique_jobs"],
            "total_chunks_indexed": vector_stats["total_chunks"],
            "avg_chunks_per_job": vector_stats["avg_chunks_per_job"],
            "categories": list(vector_stats["categories"].keys()),
            "job_levels": list(vector_stats["job_levels"].keys()),
            "top_k_results": config.TOP_K_RESULTS,
            "retrieval_multiplier": config.RETRIEVAL_MULTIPLIER,
        }

    def print_results(self, results: Dict[str, Any], verbose: bool = False):
        """
        Print search results in a formatted way

        Args:
            results: Results from retrieve()
            verbose: Whether to show detailed information
        """
        print("\n" + "=" * 70)
        print("🎯 SEARCH RESULTS")
        print("=" * 70)

        # Query info
        query_info = results["query_info"]
        print(f'\n📝 Original Query: "{query_info["original_query"]}"')
        print(f'🔍 Semantic Query: "{query_info["semantic_query"]}"')

        if query_info["filters"]:
            print("\n🔧 Filters Applied:")
            for key, value in query_info["filters"].items():
                print(f"   • {key}: {value}")

        # Search info
        search_info = results["search_info"]
        print(f"\n📊 Search Statistics:")
        print(f"   • Chunks found: {search_info['total_chunks_found']}")
        print(f"   • Unique jobs: {search_info['unique_jobs_found']}")
        print(
            f"   • Showing top: {min(search_info['requested_top_k'], len(results['jobs']))}"
        )

        # Results
        jobs = results["jobs"]

        if not jobs:
            print("\n❌ No jobs found matching your criteria.")
            print("Try:")
            print("   • Using different keywords")
            print("   • Removing some filters")
            print("   • Broadening your search")
            return

        print(f"\n{'=' * 70}")
        print(f"Found {len(jobs)} relevant job(s):")
        print("=" * 70)

        for job in jobs:
            print(f"\n#{job['rank']} - {job['job_title']}")
            print(f"{'─' * 70}")
            print(f"   🏢 Company:    {job['company']}")
            print(f"   📂 Category:   {job['category']}")
            print(f"   📍 Location:   {job['location']}")
            print(f"   🎯 Level:      {job['job_level']}")
            print(f"   📅 Posted:     {job['publication_date']}")
            print(f"   🏷️  Tags:       {job['tags']}")
            print(
                f"   ⭐ Relevance:  {job['relevance_label']} ({job['similarity_percentage']}%)"
            )
            print(f"   📄 Chunks:     {job['matched_chunks']} matched")

            if verbose:
                print(f"\n   📝 Snippet:")
                print(f"   {job['snippet'][:300]}...")

                print(f"\n   🔍 Top Matching Chunks:")
                for i, chunk in enumerate(job["top_chunks"], 1):
                    print(
                        f"      {i}. [{chunk['type']}] Distance: {chunk['distance']:.4f}"
                    )
                    print(f"         {chunk['text'][:100]}...")

        print("\n" + "=" * 70)


# Convenience function
def search_jobs(query: str, top_k: int = 5) -> Dict[str, Any]:
    """
    Convenience function to search for jobs

    Args:
        query: Search query
        top_k: Number of results

    Returns:
        Search results
    """
    retriever = JobRetriever()
    return retriever.retrieve(query, top_k=top_k)


# Testing and demonstration
if __name__ == "__main__":
    print("Testing JobRetriever...\n")

    print("=" * 70)
    print("TEST 1: Initialize Retriever")
    print("=" * 70)

    try:
        retriever = JobRetriever()

        # Get statistics
        stats = retriever.get_statistics()
        print(f"\n📊 Retriever Statistics:")
        print(f"   Jobs indexed: {stats['total_jobs_indexed']}")
        print(f"   Chunks indexed: {stats['total_chunks_indexed']}")
        print(f"   Avg chunks/job: {stats['avg_chunks_per_job']:.1f}")

        if stats["total_jobs_indexed"] == 0:
            print("\n⚠️  No jobs in vector store!")
            print(
                "Please run scripts/setup_database.py first to populate the database."
            )
            print("\nCannot run retrieval tests without data.")
            exit(0)

        print("\n" + "=" * 70)
        print("TEST 2: Simple Query")
        print("=" * 70)

        query = "Python developer"
        print(f'\nQuery: "{query}"')
        results = retriever.retrieve(query, top_k=3)
        retriever.print_results(results, verbose=False)

        print("\n" + "=" * 70)
        print("TEST 3: Query with Filters")
        print("=" * 70)

        query = "senior software engineer in San Francisco"
        print(f'\nQuery: "{query}"')
        results = retriever.retrieve(query, top_k=3)
        retriever.print_results(results, verbose=True)

        print("\n" + "=" * 70)
        print("TEST 4: Retrieve by ID")
        print("=" * 70)

        if results["jobs"]:
            job_id = results["jobs"][0]["job_id"]
            print(f"\nRetrieving job ID: {job_id}")
            job = retriever.retrieve_by_id(job_id)

            if job:
                print(f"\n✅ Job: {job['job_title']}")
                print(f"   Total chunks: {job['total_chunks']}")
                print(f"   Description length: {len(job['full_description'])} chars")

        print("\n✅ All JobRetriever tests complete!")

    except ValueError as e:
        print(f"\n❌ Error: {e}")
        print("\nPlease configure your Gemini API key in the .env file")
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        import traceback

        traceback.print_exc()
