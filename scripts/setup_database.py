#!/usr/bin/env python3
"""
Database Setup Script for LF Jobs RAG System

This script performs ONE-TIME database initialization:
1. Loads job data from CSV
2. Preprocesses all jobs
3. Generates embeddings for all text chunks
4. Stores embeddings and metadata in ChromaDB

Usage:
    python scripts/setup_database.py

    # With custom batch size
    python scripts/setup_database.py --batch-size 50

    # Force recreate database
    python scripts/setup_database.py --force

Author: Samir Dahal
Date: 2025-11-13
"""

import argparse
import json
import logging
import sys
import time
from collections import deque
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Try to import tqdm, but make it optional
try:
    from tqdm import tqdm

    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False

    # Fallback: simple progress function
    def tqdm(iterable, desc="", unit=""):
        """Fallback progress indicator when tqdm is not available"""
        total = len(iterable) if hasattr(iterable, "__len__") else None
        for i, item in enumerate(iterable, 1):
            if total:
                print(f"\r{desc}: {i}/{total} {unit}", end="", flush=True)
            yield item
        if total:
            print()  # New line after completion


# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Import after path is set (noqa comments suppress linting)
from src.config import Config  # noqa: E402
from src.data_loader import load_jobs_data  # noqa: E402
from src.embeddings import EmbeddingGenerator  # noqa: E402
from src.preprocessing import TextPreprocessor  # noqa: E402
from src.vector_store import VectorStore  # noqa: E402

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(), logging.FileHandler("database_setup.log")],
)
logger = logging.getLogger(__name__)


class RateLimiter:
    """Token bucket rate limiter for API calls."""

    def __init__(self, rpm: int = 100, tpm: int = 30000, rpd: int = 1000):
        """
        Initialize rate limiter.

        Args:
            rpm: Requests per minute limit
            tpm: Tokens per minute limit (not strictly enforced, just logged)
            rpd: Requests per day limit
        """
        self.rpm = rpm
        self.tpm = tpm
        self.rpd = rpd

        # Track recent requests (timestamp of each request)
        self.request_times = deque()
        self.daily_request_times = deque()

        # For logging
        self.total_requests = 0
        self.total_waits = 0
        self.total_wait_time = 0.0

    def wait_if_needed(self) -> None:
        """Wait if we're about to exceed rate limits."""
        now = time.time()

        # Remove requests older than 1 minute
        while self.request_times and now - self.request_times[0] > 60:
            self.request_times.popleft()

        # Remove requests older than 1 day
        while self.daily_request_times and now - self.daily_request_times[0] > 86400:
            self.daily_request_times.popleft()

        # Check daily limit
        if len(self.daily_request_times) >= self.rpd:
            wait_time = 86400 - (now - self.daily_request_times[0]) + 1
            logger.warning(
                f"⏳ Daily limit ({self.rpd} requests) reached. "
                f"Waiting {wait_time / 3600:.1f} hours..."
            )
            time.sleep(wait_time)
            now = time.time()
            # Clean up after wait
            while (
                self.daily_request_times and now - self.daily_request_times[0] > 86400
            ):
                self.daily_request_times.popleft()

        # Check per-minute limit
        if len(self.request_times) >= self.rpm:
            # Need to wait until oldest request is >60s old
            wait_time = 60 - (now - self.request_times[0]) + 0.1  # Small buffer
            if wait_time > 0:
                logger.info(
                    f"⏳ Rate limit: {len(self.request_times)}/{self.rpm} requests in last minute. "
                    f"Waiting {wait_time:.1f}s..."
                )
                self.total_waits += 1
                self.total_wait_time += wait_time
                time.sleep(wait_time)
                now = time.time()
                # Clean up after wait
                while self.request_times and now - self.request_times[0] > 60:
                    self.request_times.popleft()

        # Record this request
        self.request_times.append(now)
        self.daily_request_times.append(now)
        self.total_requests += 1

    def get_stats(self) -> Dict[str, Any]:
        """Get rate limiter statistics."""
        return {
            "total_requests": self.total_requests,
            "total_waits": self.total_waits,
            "total_wait_time": self.total_wait_time,
            "current_rpm_usage": len(self.request_times),
            "current_rpd_usage": len(self.daily_request_times),
        }


class DatabaseSetup:
    """Handles database initialization and population."""

    def __init__(
        self,
        batch_size: int = 100,
        force: bool = False,
        limit: int = None,
        checkpoint_file: str = "database_setup_checkpoint.json",
        use_local: bool = False,
    ):
        """
        Initialize database setup.

        Args:
            batch_size: Number of jobs to process in each batch
            force: If True, recreate database even if it exists
            limit: Limit number of jobs to process (None for all jobs)
            checkpoint_file: Path to checkpoint file for resume capability
            use_local: If True, use local sentence-transformers model instead of Gemini
        """
        self.batch_size = batch_size
        self.force = force
        self.limit = limit
        self.checkpoint_file = Path(checkpoint_file)
        self.use_local = use_local

        # Initialize components
        logger.info("Initializing components...")
        self.preprocessor = TextPreprocessor()
        self.embedding_generator = EmbeddingGenerator(use_local=use_local)
        self.vector_store = VectorStore()

        # Rate limiter for Gemini API (only if not using local)
        if not use_local:
            self.rate_limiter = RateLimiter(rpm=100, tpm=30000, rpd=1000)
        else:
            self.rate_limiter = None  # No rate limiting needed for local model

        # Checkpoint tracking
        self.processed_job_ids = set()
        self.load_checkpoint()

        self.config = Config()

        # Statistics
        self.stats = {
            "total_jobs": 0,
            "processed_jobs": 0,
            "failed_jobs": 0,
            "total_chunks": 0,
            "total_embeddings": 0,
            "start_time": None,
            "end_time": None,
            "errors": [],
        }

    def load_checkpoint(self) -> None:
        """Load checkpoint file to resume from previous run."""
        if self.checkpoint_file.exists():
            try:
                with open(self.checkpoint_file, "r") as f:
                    checkpoint_data = json.load(f)
                    self.processed_job_ids = set(
                        checkpoint_data.get("processed_job_ids", [])
                    )
                    logger.info(
                        f"📂 Loaded checkpoint: {len(self.processed_job_ids)} jobs already processed"
                    )
            except Exception as e:
                logger.warning(f"Could not load checkpoint: {e}. Starting fresh.")
                self.processed_job_ids = set()
        else:
            logger.info("No checkpoint found. Starting fresh.")

    def save_checkpoint(self) -> None:
        """Save current progress to checkpoint file."""
        try:
            checkpoint_data = {
                "processed_job_ids": list(self.processed_job_ids),
                "last_updated": datetime.now().isoformat(),
                "total_processed": len(self.processed_job_ids),
            }
            with open(self.checkpoint_file, "w") as f:
                json.dump(checkpoint_data, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save checkpoint: {e}")

    def clear_checkpoint(self) -> None:
        """Clear checkpoint file."""
        if self.checkpoint_file.exists():
            self.checkpoint_file.unlink()
            logger.info("Checkpoint file cleared")

    def check_existing_database(self) -> bool:
        """
        Check if database already exists and has data.

        Returns:
            True if database exists and has data
        """
        try:
            count = self.vector_store.collection.count()
            if count > 0:
                logger.info(f"Found existing database with {count} chunks")
                return True
            return False
        except Exception as e:
            logger.warning(f"Error checking existing database: {e}")
            return False

    def clear_database(self) -> None:
        """Clear existing database and checkpoint."""
        try:
            logger.info("Clearing existing database...")
            self.vector_store.clear_collection()
            logger.info("✅ Database cleared successfully")

            # Also clear checkpoint when forcing new run
            self.clear_checkpoint()
            self.processed_job_ids = set()

        except Exception as e:
            logger.error(f"❌ Error clearing database: {e}")
            raise

    def load_data(self) -> List[Dict[str, Any]]:
        """
        Load job data from CSV.

        Returns:
            List of job dictionaries
        """
        logger.info("Loading job data from CSV...")
        try:
            df = load_jobs_data()

            # Apply limit if specified
            if self.limit is not None:
                logger.info(f"⚠️  Limiting to first {self.limit} jobs for testing")
                df = df.head(self.limit)

            # Convert DataFrame to list of dictionaries
            jobs = df.to_dict("records")
            self.stats["total_jobs"] = len(jobs)
            logger.info(f"✅ Loaded {len(jobs)} jobs from CSV")
            return jobs
        except Exception as e:
            logger.error(f"❌ Error loading data: {e}")
            raise

    def preprocess_job(self, job: Dict[str, Any]) -> Tuple[str, Dict[str, Any]]:
        """
        Preprocess a single job.

        Args:
            job: Job dictionary

        Returns:
            Tuple of (cleaned_description, metadata)
        """
        try:
            # Clean HTML from job description
            job_description = str(job.get("Job Description", ""))
            cleaned_text = self.preprocessor.clean_html(job_description)

            # Extract metadata - convert DataFrame row to dict if needed
            metadata = {
                "job_id": str(job.get("ID", "")),
                "job_title": str(job.get("Job Title", "")),
                "company": str(job.get("Company Name", "")),
                "location": str(job.get("Job Location", "")),
                "job_type": str(job.get("Job Category", "")),
                "seniority_level": str(job.get("Job Level", "")),
                "employment_type": "",
                "industries": "",
                "job_functions": "",
            }

            return cleaned_text, metadata

        except Exception as e:
            logger.error(f"Error preprocessing job {job.get('Job Id', 'unknown')}: {e}")
            raise

    def process_batch(
        self, jobs_batch: List[Dict[str, Any]], batch_num: int
    ) -> Tuple[int, int]:
        """
        Process a batch of jobs with rate limiting and checkpointing.

        Args:
            jobs_batch: List of jobs to process
            batch_num: Batch number for logging

        Returns:
            Tuple of (successful_count, failed_count)
        """
        successful = 0
        failed = 0

        logger.info(f"\n{'=' * 70}")
        logger.info(f"Processing Batch {batch_num} ({len(jobs_batch)} jobs)")
        logger.info(f"{'=' * 70}")

        for job in tqdm(jobs_batch, desc=f"Batch {batch_num}", unit="job"):
            try:
                job_id = str(job.get("ID", ""))

                if not job_id:
                    logger.warning(f"Skipping job without ID: {job}")
                    failed += 1
                    continue

                # Skip if already processed (checkpoint resume)
                if job_id in self.processed_job_ids:
                    logger.info(
                        f"✓ Job {job_id} already processed (from checkpoint), skipping"
                    )
                    successful += 1
                    continue

                # Preprocess job
                cleaned_description, metadata = self.preprocess_job(job)
                job_title = str(job.get("Job Title", ""))

                # Create chunks using TextPreprocessor
                chunks = self.preprocessor.create_chunks(job_title, cleaned_description)

                if not chunks:
                    logger.warning(f"No chunks created for job {job_id}")
                    failed += 1
                    continue

                # Generate embeddings for all chunks
                texts = [chunk["text"] for chunk in chunks]

                if self.use_local:
                    # Use batch embedding for local model (fast, no rate limiting needed)
                    embeddings = self.embedding_generator.create_embeddings_batch(
                        texts, task_type="retrieval_document", show_progress=False
                    )
                else:
                    # Use Gemini API with rate limiting and retry logic
                    embeddings = []

                    for i, text in enumerate(texts):
                        # Wait if needed to respect rate limits
                        self.rate_limiter.wait_if_needed()

                        # Retry logic with exponential backoff
                        max_retries = 3
                        retry_delay = 1.0

                        for attempt in range(max_retries):
                            embedding_error = None
                            try:
                                embedding = self.embedding_generator.create_embedding(
                                    text, task_type="retrieval_document"
                                )
                                embeddings.append(embedding)
                                break  # Success, exit retry loop
                            except Exception as e:
                                embedding_error = e
                                if attempt < max_retries - 1:
                                    wait_time = retry_delay * (2**attempt)
                                    logger.warning(
                                        f"Embedding failed for chunk {i + 1}/{len(texts)} of job {job_id}, "
                                        f"attempt {attempt + 1}/{max_retries}. Retrying in {wait_time}s... Error: {e}"
                                    )
                                    time.sleep(wait_time)
                                else:
                                    logger.error(
                                        f"Failed to generate embedding for chunk {i + 1}/{len(texts)} "
                                        f"of job {job_id} after {max_retries} attempts: {e}"
                                    )
                                    embeddings.append(
                                        None
                                    )  # Add None for failed embedding

                if not embeddings or len(embeddings) != len(chunks):
                    logger.warning(
                        f"Embedding generation failed for job {job_id}. "
                        f"Expected {len(chunks)}, got {len(embeddings)}"
                    )
                    failed += 1
                    continue

                # Check how many embeddings are None
                valid_embeddings = [e for e in embeddings if e is not None]
                if len(valid_embeddings) < len(embeddings):
                    logger.warning(
                        f"Job {job_id}: Only {len(valid_embeddings)}/{len(embeddings)} embeddings succeeded"
                    )

                # Add embeddings to chunks
                for chunk, embedding in zip(chunks, embeddings):
                    chunk["embedding"] = embedding

                # Store in vector database (only chunks with valid embeddings will be stored)
                num_added = self.vector_store.add_job_chunks(
                    job_id=job_id,
                    job_title=metadata["job_title"],
                    company=metadata["company"],
                    category=metadata["job_type"],
                    location=metadata["location"],
                    job_level=metadata["seniority_level"],
                    publication_date=str(job.get("Publication Date", "")),
                    tags=str(job.get("Tags", "")),
                    chunks=chunks,
                )

                if num_added > 0:
                    logger.info(
                        f"✅ Job {job_id}: Added {num_added} chunks to database"
                    )

                    # Mark as processed and save checkpoint
                    self.processed_job_ids.add(job_id)
                    self.save_checkpoint()

                    # Update statistics
                    self.stats["total_chunks"] += len(chunks)
                    self.stats["total_embeddings"] += len(valid_embeddings)
                    successful += 1
                else:
                    logger.warning(f"❌ Job {job_id}: No chunks added to database")
                    failed += 1

            except Exception as e:
                failed += 1
                error_msg = (
                    f"Error processing job {job.get('Job Id', 'unknown')}: {str(e)}"
                )
                logger.error(error_msg)
                self.stats["errors"].append(error_msg)

        return successful, failed

    def process_all_jobs(self, jobs: List[Dict[str, Any]]) -> None:
        """
        Process all jobs in batches.

        Args:
            jobs: List of all jobs to process
        """
        total_jobs = len(jobs)
        num_batches = (total_jobs + self.batch_size - 1) // self.batch_size

        logger.info(f"\n{'=' * 70}")
        logger.info("Starting Database Population")
        logger.info(f"{'=' * 70}")
        logger.info(f"Total jobs: {total_jobs}")
        logger.info(f"Batch size: {self.batch_size}")
        logger.info(f"Number of batches: {num_batches}")
        logger.info(f"{'=' * 70}\n")

        self.stats["start_time"] = datetime.now()

        total_successful = 0
        total_failed = 0

        # Process jobs in batches
        for i in range(0, total_jobs, self.batch_size):
            batch_num = (i // self.batch_size) + 1
            batch = jobs[i : i + self.batch_size]

            successful, failed = self.process_batch(batch, batch_num)
            total_successful += successful
            total_failed += failed

            # Log batch progress
            logger.info(f"\nBatch {batch_num} Summary:")
            logger.info(f"  ✅ Successful: {successful}")
            logger.info(f"  ❌ Failed: {failed}")
            logger.info(f"  📊 Overall Progress: {total_successful}/{total_jobs} jobs")

        self.stats["processed_jobs"] = total_successful
        self.stats["failed_jobs"] = total_failed
        self.stats["end_time"] = datetime.now()

    def print_final_summary(self) -> None:
        """Print final summary of database setup."""
        duration = (self.stats["end_time"] - self.stats["start_time"]).total_seconds()

        print("\n" + "=" * 70)
        print("DATABASE SETUP COMPLETE!")
        print("=" * 70)
        print("\n📊 Final Statistics:")
        print(f"  • Total jobs in CSV: {self.stats['total_jobs']}")
        print(f"  • Successfully processed: {self.stats['processed_jobs']}")
        print(f"  • Failed: {self.stats['failed_jobs']}")
        print(
            f"  • Success rate: {(self.stats['processed_jobs'] / max(self.stats['total_jobs'], 1) * 100):.2f}%"
        )
        print(f"\n  • Total text chunks: {self.stats['total_chunks']}")
        print(f"  • Total embeddings: {self.stats['total_embeddings']}")
        print(
            f"  • Average chunks per job: {self.stats['total_chunks'] / max(self.stats['processed_jobs'], 1):.2f}"
        )
        print(f"\n  • Duration: {duration:.2f} seconds ({duration / 60:.2f} minutes)")
        if self.stats["processed_jobs"] > 0:
            print(
                f"  • Processing speed: {self.stats['processed_jobs'] / duration:.2f} jobs/second"
            )

        # Rate limiter stats (only if using Gemini)
        if self.rate_limiter is not None:
            rate_stats = self.rate_limiter.get_stats()
            print("\n⏱️  Rate Limiter Statistics:")
            print(f"  • Total API requests: {rate_stats['total_requests']}")
            print(f"  • Total rate limit waits: {rate_stats['total_waits']}")
            print(
                f"  • Total wait time: {rate_stats['total_wait_time']:.2f} seconds ({rate_stats['total_wait_time'] / 60:.2f} minutes)"
            )
            print(f"  • Current RPM usage: {rate_stats['current_rpm_usage']}/100")
            print(f"  • Current RPD usage: {rate_stats['current_rpd_usage']}/1000")
        else:
            print("\n🚀 Used local embedding model (no rate limiting)")

        if self.stats["errors"]:
            print(f"\n⚠️  Errors encountered: {len(self.stats['errors'])}")
            print("   (See database_setup.log for details)")

        # Verify final database state
        try:
            final_count = self.vector_store.collection.count()
            print("\n✅ Database verification:")
            print(f"  • Total chunks in database: {final_count}")
            print(f"  • Database location: {self.config.CHROMA_DB_PATH}")
            print(f"  • Collection name: {self.config.COLLECTION_NAME}")
        except Exception as e:
            print(f"\n❌ Error verifying database: {e}")

        print("\n" + "=" * 70)
        print("Next Steps:")
        print("  1. Start API server: python -m src.api")
        print("  2. Or run main: python main.py")
        print("  3. Test queries: POST http://localhost:8000/api/query")
        print("  4. View docs: http://localhost:8000/docs")
        print("=" * 70 + "\n")

    def run(self) -> None:
        """Run the complete database setup process."""
        try:
            # Check if database exists
            if not self.force and self.check_existing_database():
                response = input(
                    "\n⚠️  Database already exists with data. "
                    "Do you want to recreate it? (yes/no): "
                )
                if response.lower() not in ["yes", "y"]:
                    logger.info("Aborting database setup.")
                    return

            # Clear database if exists or force flag is set
            if self.force or self.check_existing_database():
                self.clear_database()

            # Load data
            jobs = self.load_data()

            if len(jobs) == 0:
                logger.error("No jobs loaded. Aborting setup.")
                return

            # Process all jobs
            self.process_all_jobs(jobs)

            # Print final summary
            self.print_final_summary()

            # Write detailed error log if errors occurred
            if self.stats["errors"]:
                error_log_path = "database_setup_errors.log"
                with open(error_log_path, "w") as f:
                    f.write(f"Database Setup Errors - {datetime.now()}\n")
                    f.write("=" * 70 + "\n\n")
                    for error in self.stats["errors"]:
                        f.write(f"{error}\n")
                logger.info(f"Detailed error log written to {error_log_path}")

        except KeyboardInterrupt:
            logger.warning("\n\n⚠️  Database setup interrupted by user")
            sys.exit(1)
        except Exception as e:
            logger.error(f"\n\n❌ Fatal error during database setup: {e}")
            raise


def main():
    """Main entry point for database setup script."""
    parser = argparse.ArgumentParser(
        description="Setup and populate ChromaDB with job data",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Default setup
  python scripts/setup_database.py
  
  # Custom batch size
  python scripts/setup_database.py --batch-size 50
  
  # Force recreate database
  python scripts/setup_database.py --force
  
  # Small batch with force recreate
  python scripts/setup_database.py --batch-size 25 --force
        """,
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=100,
        help="Number of jobs to process in each batch (default: 100)",
    )

    parser.add_argument(
        "--force", action="store_true", help="Force recreate database even if it exists"
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit number of jobs to process (useful for testing, e.g., --limit 2)",
    )

    parser.add_argument(
        "--use-local",
        action="store_true",
        help="Use local sentence-transformers model instead of Gemini API (faster, no quota limits)",
    )

    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")

    args = parser.parse_args()

    # Set logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Print banner
    print("\n" + "=" * 70)
    print("LF Jobs RAG System - Database Setup")
    print("=" * 70)
    print(f"Batch size: {args.batch_size}")
    print(f"Force recreate: {args.force}")
    print(f"Limit: {args.limit if args.limit else 'All jobs'}")
    print(
        f"Embedding mode: {'Local (sentence-transformers)' if args.use_local else 'Gemini API'}"
    )
    print(f"Verbose mode: {args.verbose}")
    print("=" * 70 + "\n")

    # Run setup
    setup = DatabaseSetup(
        batch_size=args.batch_size,
        force=args.force,
        limit=args.limit,
        use_local=args.use_local,
    )
    setup.run()


if __name__ == "__main__":
    main()
