"""
Text preprocessing module
Handles HTML cleaning and intelligent text chunking for job descriptions
"""

import re
from typing import Dict, List

import pandas as pd
from bs4 import BeautifulSoup
from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.config import config


class TextPreprocessor:
    """
    Handles text cleaning and chunking operations
    """

    def __init__(self):
        """Initialize text preprocessor with chunking settings from config"""
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=config.CHUNK_SIZE,
            chunk_overlap=config.CHUNK_OVERLAP,
            separators=config.CHUNK_SEPARATORS,
            length_function=len,
        )

    @staticmethod
    def clean_html(html_text: str) -> str:
        """
        Remove HTML tags and clean text

        Args:
            html_text: Raw HTML text from job description

        Returns:
            Clean text without HTML tags
        """
        if not html_text or pd.isna(html_text):
            return ""

        # Parse HTML with BeautifulSoup
        soup = BeautifulSoup(html_text, "html.parser")

        # Get text content
        text = soup.get_text(separator=" ")

        # Remove extra whitespace (multiple spaces, tabs, newlines)
        text = re.sub(r"\s+", " ", text)

        # Strip leading/trailing whitespace
        text = text.strip()

        return text

    def create_chunks(self, job_title: str, description: str) -> List[Dict[str, str]]:
        """
        Create intelligent chunks from job description

        Strategy:
        1. First chunk: Title + Introduction (most important for matching)
        2. Remaining chunks: Use LangChain's RecursiveCharacterTextSplitter
        3. Detect chunk types (responsibilities, requirements, benefits)

        Args:
            job_title: Job title
            description: Cleaned job description text

        Returns:
            List of dictionaries containing chunk information
        """
        chunks = []

        if not description:
            # If no description, just return title
            return [{"text": job_title, "type": "title_only", "importance": "high"}]

        # ==================== CHUNK 1: Title + Introduction ====================
        # This is the most important chunk for semantic matching
        intro_length = min(300, len(description))
        intro_text = f"{job_title}. {description[:intro_length]}"

        chunks.append(
            {"text": intro_text.strip(), "type": "title_intro", "importance": "high"}
        )

        # ==================== REMAINING CHUNKS ====================
        # Use LangChain to split the remaining description
        remaining_text = description[intro_length:].strip()

        if remaining_text:
            # Split text into chunks
            text_chunks = self.text_splitter.split_text(remaining_text)

            # Process each chunk and detect its type
            for i, chunk_text in enumerate(text_chunks):
                chunk_type = self._detect_chunk_type(chunk_text)
                importance = self._determine_importance(chunk_type)

                chunks.append(
                    {
                        "text": chunk_text.strip(),
                        "type": chunk_type,
                        "importance": importance,
                    }
                )

        return chunks

    @staticmethod
    def _detect_chunk_type(text: str) -> str:
        """
        Detect the type of content in a chunk based on keywords

        Args:
            text: Chunk text

        Returns:
            Chunk type (responsibilities, requirements, benefits, or general)
        """
        text_lower = text.lower()

        # Define keyword patterns for each section type
        responsibility_keywords = [
            "responsibilities",
            "duties",
            "what you will do",
            "role",
            "day-to-day",
            "tasks",
            "you will",
            "responsible for",
        ]

        requirement_keywords = [
            "requirements",
            "qualifications",
            "skills",
            "experience",
            "education",
            "must have",
            "should have",
            "you have",
            "required",
            "preferred",
            "ideal candidate",
        ]

        benefit_keywords = [
            "benefits",
            "we offer",
            "compensation",
            "salary",
            "perks",
            "package",
            "health",
            "insurance",
            "vacation",
            "pto",
            "pay range",
            "equity",
            "bonus",
        ]

        # Check for each type (in order of priority)
        for keyword in responsibility_keywords:
            if keyword in text_lower:
                return "responsibilities"

        for keyword in requirement_keywords:
            if keyword in text_lower:
                return "requirements"

        for keyword in benefit_keywords:
            if keyword in text_lower:
                return "benefits"

        # Default type
        return "general"

    @staticmethod
    def _determine_importance(chunk_type: str) -> str:
        """
        Determine importance level based on chunk type

        Args:
            chunk_type: Type of chunk

        Returns:
            Importance level (high, medium, or low)
        """
        importance_map = {
            "title_intro": "high",
            "title_only": "high",
            "responsibilities": "high",
            "requirements": "high",
            "benefits": "medium",
            "general": "medium",
        }

        return importance_map.get(chunk_type, "medium")

    def process_job(self, job_row: pd.Series) -> Dict:
        """
        Process a single job: clean HTML and create chunks

        Args:
            job_row: Pandas Series containing job data

        Returns:
            Dictionary with cleaned data and chunks
        """
        # Clean HTML from job description
        cleaned_description = self.clean_html(job_row["Job Description"])

        # Create chunks
        chunks = self.create_chunks(
            job_title=job_row["Job Title"], description=cleaned_description
        )

        return {
            "job_id": job_row["ID"],
            "job_title": job_row["Job Title"],
            "cleaned_description": cleaned_description,
            "chunks": chunks,
            "chunk_count": len(chunks),
        }

    def process_dataframe(self, df: pd.DataFrame, verbose: bool = True) -> pd.DataFrame:
        """
        Process entire DataFrame: clean all descriptions and create chunks

        Args:
            df: DataFrame containing job data
            verbose: Whether to print progress

        Returns:
            DataFrame with added 'cleaned_description' and 'chunks' columns
        """
        if verbose:
            print(f"\n🔧 Processing {len(df)} jobs...")
            print("   • Cleaning HTML")
            print("   • Creating intelligent chunks")

        # Create a copy to avoid modifying original
        df_processed = df.copy()

        # Clean HTML descriptions
        df_processed["cleaned_description"] = df_processed["Job Description"].apply(
            self.clean_html
        )

        # Create chunks for each job
        df_processed["chunks"] = df_processed.apply(
            lambda row: self.create_chunks(
                row["Job Title"], row["cleaned_description"]
            ),
            axis=1,
        )

        # Add chunk count for analysis
        df_processed["chunk_count"] = df_processed["chunks"].apply(len)

        if verbose:
            print(f"\n✅ Processing complete!")
            print(f"   • Total chunks created: {df_processed['chunk_count'].sum()}")
            print(
                f"   • Average chunks per job: {df_processed['chunk_count'].mean():.2f}"
            )
            print(f"   • Min chunks: {df_processed['chunk_count'].min()}")
            print(f"   • Max chunks: {df_processed['chunk_count'].max()}")

            # Show chunk type distribution
            all_chunk_types = []
            for chunks_list in df_processed["chunks"]:
                all_chunk_types.extend([chunk["type"] for chunk in chunks_list])

            print(f"\n📊 Chunk Type Distribution:")
            chunk_type_counts = pd.Series(all_chunk_types).value_counts()
            for chunk_type, count in chunk_type_counts.items():
                print(f"   • {chunk_type}: {count}")

        return df_processed


# Convenience functions
def clean_html_text(html_text: str) -> str:
    """
    Convenience function to clean HTML text

    Args:
        html_text: HTML text to clean

    Returns:
        Cleaned text
    """
    preprocessor = TextPreprocessor()
    return preprocessor.clean_html(html_text)


def create_job_chunks(job_title: str, description: str) -> List[Dict[str, str]]:
    """
    Convenience function to create chunks for a job

    Args:
        job_title: Job title
        description: Job description (cleaned text)

    Returns:
        List of chunk dictionaries
    """
    preprocessor = TextPreprocessor()
    return preprocessor.create_chunks(job_title, description)


# Testing and demonstration
if __name__ == "__main__":
    print("Testing TextPreprocessor...\n")

    # Test HTML cleaning
    print("=" * 60)
    print("TEST 1: HTML Cleaning")
    print("=" * 60)

    sample_html = """
    <p><b>Job Description:</b><br><br>
    At Bank of America, we are guided by a common purpose to help make 
    financial lives better.<br><br>
    <b>Responsibilities:</b><br>
    <ul>
        <li>Lead the development of new models</li>
        <li>Create technical documentation</li>
    </ul>
    </p>
    """

    preprocessor = TextPreprocessor()
    cleaned = preprocessor.clean_html(sample_html)

    print(f"Original length: {len(sample_html)} chars")
    print(f"Cleaned length: {len(cleaned)} chars")
    print(f"\nCleaned text:\n{cleaned[:200]}...")

    # Test chunking
    print("\n" + "=" * 60)
    print("TEST 2: Intelligent Chunking")
    print("=" * 60)

    sample_title = "Senior Python Developer"
    chunks = preprocessor.create_chunks(sample_title, cleaned)

    print(f"\nCreated {len(chunks)} chunks:")
    for i, chunk in enumerate(chunks, 1):
        print(f"\nChunk {i}:")
        print(f"  Type: {chunk['type']}")
        print(f"  Importance: {chunk['importance']}")
        print(f"  Length: {len(chunk['text'])} chars")
        print(f"  Preview: {chunk['text'][:100]}...")

    # Test with real data (if available)
    print("\n" + "=" * 60)
    print("TEST 3: Processing Real Data")
    print("=" * 60)

    try:
        from src.data_loader import load_jobs_data

        # Load just first 5 jobs for testing
        df = load_jobs_data()
        df_sample = df.head(5)

        # Process them
        df_processed = preprocessor.process_dataframe(df_sample, verbose=True)

        # Show example
        print(f"\n📄 Example Job Processing:")
        first_job = df_processed.iloc[0]
        print(f"   Job: {first_job['Job Title']}")
        print(
            f"   Original description length: {len(first_job['Job Description'])} chars"
        )
        print(
            f"   Cleaned description length: {len(first_job['cleaned_description'])} chars"
        )
        print(f"   Number of chunks: {first_job['chunk_count']}")

        print("\n✅ TextPreprocessor test complete!")

    except Exception as e:
        print(f"⚠️  Could not test with real data: {e}")
        print("   This is normal if data hasn't been loaded yet.")
