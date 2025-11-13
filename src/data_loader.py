"""
Data loading and validation module
Loads job data from CSV and provides utility functions for data access
"""

from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

from src.config import config


class DataLoader:
    """
    Handles loading and validation of job data from CSV
    """

    def __init__(self, data_path: Optional[Path] = None):
        """
        Initialize DataLoader

        Args:
            data_path: Path to CSV file (defaults to config.DATA_PATH)
        """
        self.data_path = data_path or config.DATA_PATH
        self.df = None
        self._unique_values_cache = None

    def load_data(self) -> pd.DataFrame:
        """
        Load job data from CSV file

        Returns:
            DataFrame containing job data

        Raises:
            FileNotFoundError: If CSV file doesn't exist
            ValueError: If required columns are missing
        """
        # Check if file exists
        if not self.data_path.exists():
            raise FileNotFoundError(
                f"Data file not found: {self.data_path}\n"
                f"Please ensure lf_jobs.csv is in the data/ directory"
            )

        print(f"📂 Loading data from: {self.data_path}")

        # Load CSV
        try:
            self.df = pd.read_csv(self.data_path)
            print(f"✅ Loaded {len(self.df)} jobs successfully")
        except Exception as e:
            raise ValueError(f"Error reading CSV file: {e}")

        # Validate columns
        self._validate_columns()

        # Clean data
        self._clean_data()

        # Display summary
        self._display_summary()

        return self.df

    def _validate_columns(self):
        """Validate that all required columns exist"""
        missing_columns = [
            col for col in config.REQUIRED_COLUMNS if col not in self.df.columns
        ]

        if missing_columns:
            raise ValueError(
                f"Missing required columns: {missing_columns}\n"
                f"Expected columns: {config.REQUIRED_COLUMNS}\n"
                f"Found columns: {list(self.df.columns)}"
            )

        print(
            f"✅ All required columns present: {len(config.REQUIRED_COLUMNS)} columns"
        )

    def _clean_data(self):
        """Clean and prepare data"""
        print("🧹 Cleaning data...")

        # Fill NaN values in Tags column
        self.df["Tags"] = self.df["Tags"].fillna("")

        # Fill NaN values in Job Description
        self.df["Job Description"] = self.df["Job Description"].fillna("")

        # Remove any completely empty rows
        initial_count = len(self.df)
        self.df = self.df.dropna(subset=["ID", "Job Title", "Company Name"])
        removed_count = initial_count - len(self.df)

        if removed_count > 0:
            print(f"   ⚠️  Removed {removed_count} rows with missing critical data")

        # Convert Publication Date to datetime (with error handling)
        try:
            self.df["Publication Date"] = pd.to_datetime(
                self.df["Publication Date"], errors="coerce"
            )
        except Exception as e:
            print(f"   ⚠️  Warning: Could not parse some dates: {e}")

        # Strip whitespace from string columns
        string_columns = [
            "Job Title",
            "Company Name",
            "Job Location",
            "Job Level",
            "Job Category",
        ]
        for col in string_columns:
            if col in self.df.columns:
                self.df[col] = self.df[col].astype(str).str.strip()

        print(f"✅ Data cleaning complete")

    def _display_summary(self):
        """Display data summary statistics"""
        print("\n" + "=" * 60)
        print("📊 DATA SUMMARY")
        print("=" * 60)
        print(f"Total Jobs: {len(self.df)}")
        print(f"Unique Companies: {self.df['Company Name'].nunique()}")
        print(f"Unique Categories: {self.df['Job Category'].nunique()}")
        print(f"Unique Locations: {self.df['Job Location'].nunique()}")
        print(f"Unique Job Levels: {self.df['Job Level'].nunique()}")

        print("\n📈 Job Categories:")
        category_counts = self.df["Job Category"].value_counts()
        for category, count in category_counts.items():
            print(f"   • {category}: {count} jobs")

        print("\n📍 Top 10 Locations:")
        location_counts = self.df["Job Location"].value_counts().head(10)
        for location, count in location_counts.items():
            print(f"   • {location}: {count} jobs")

        print("\n🏢 Job Levels:")
        level_counts = self.df["Job Level"].value_counts()
        for level, count in level_counts.items():
            print(f"   • {level}: {count} jobs")

        print("=" * 60 + "\n")

    def get_unique_values(self) -> Dict[str, List[str]]:
        """
        Get unique values for filter fields (for query parsing)

        Returns:
            Dictionary with unique values for each filterable field
        """
        if self._unique_values_cache is not None:
            return self._unique_values_cache

        if self.df is None:
            raise ValueError("Data not loaded. Call load_data() first.")

        self._unique_values_cache = {
            "categories": sorted(self.df["Job Category"].dropna().unique().tolist()),
            "companies": sorted(self.df["Company Name"].dropna().unique().tolist()),
            "locations": sorted(self.df["Job Location"].dropna().unique().tolist()),
            "levels": sorted(self.df["Job Level"].dropna().unique().tolist()),
        }

        return self._unique_values_cache

    def get_job_by_id(self, job_id: str) -> Optional[Dict]:
        """
        Get a specific job by ID

        Args:
            job_id: Job ID (e.g., "LF0001")

        Returns:
            Dictionary containing job data, or None if not found
        """
        if self.df is None:
            raise ValueError("Data not loaded. Call load_data() first.")

        job = self.df[self.df["ID"] == job_id]

        if len(job) == 0:
            return None

        return job.iloc[0].to_dict()

    def filter_jobs(
        self,
        category: Optional[str] = None,
        company: Optional[str] = None,
        location: Optional[str] = None,
        level: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        Filter jobs by multiple criteria

        Args:
            category: Job category
            company: Company name
            location: Job location
            level: Job level

        Returns:
            Filtered DataFrame
        """
        if self.df is None:
            raise ValueError("Data not loaded. Call load_data() first.")

        filtered_df = self.df.copy()

        if category:
            filtered_df = filtered_df[filtered_df["Job Category"] == category]

        if company:
            filtered_df = filtered_df[filtered_df["Company Name"] == company]

        if location:
            # Case-insensitive partial match
            filtered_df = filtered_df[
                filtered_df["Job Location"].str.contains(location, case=False, na=False)
            ]

        if level:
            filtered_df = filtered_df[filtered_df["Job Level"] == level]

        return filtered_df

    def get_random_sample(self, n: int = 5) -> pd.DataFrame:
        """
        Get random sample of jobs (useful for testing)

        Args:
            n: Number of jobs to sample

        Returns:
            DataFrame with random sample
        """
        if self.df is None:
            raise ValueError("Data not loaded. Call load_data() first.")

        sample_size = min(n, len(self.df))
        return self.df.sample(n=sample_size)

    def get_dataframe(self) -> pd.DataFrame:
        """
        Get the full DataFrame

        Returns:
            Complete DataFrame
        """
        if self.df is None:
            raise ValueError("Data not loaded. Call load_data() first.")

        return self.df


# Convenience function for quick data loading
def load_jobs_data(data_path: Optional[Path] = None) -> pd.DataFrame:
    """
    Convenience function to load job data

    Args:
        data_path: Path to CSV file (optional)

    Returns:
        DataFrame containing job data
    """
    loader = DataLoader(data_path)
    return loader.load_data()


# Example usage and testing
if __name__ == "__main__":
    print("Testing DataLoader...\n")

    # Load data
    loader = DataLoader()
    df = loader.load_data()

    # Get unique values
    print("\n📋 Unique Values:")
    unique_vals = loader.get_unique_values()
    print(f"   Categories: {unique_vals['categories']}")
    print(f"   Levels: {unique_vals['levels']}")
    print(f"   Total Companies: {len(unique_vals['companies'])}")
    print(f"   Total Locations: {len(unique_vals['locations'])}")

    # Test filtering
    print("\n🔍 Testing Filters:")
    if len(unique_vals["levels"]) > 0:
        test_level = unique_vals["levels"][0]
        filtered = loader.filter_jobs(level=test_level)
        print(f"   Jobs with level '{test_level}': {len(filtered)}")

    # Get random sample
    print("\n🎲 Random Sample (3 jobs):")
    sample = loader.get_random_sample(3)
    for idx, job in sample.iterrows():
        print(f"   • {job['Job Title']} at {job['Company Name']}")

    # Get specific job
    print("\n🔎 Testing get_job_by_id:")
    if len(df) > 0:
        first_id = df.iloc[0]["ID"]
        job = loader.get_job_by_id(first_id)
        if job:
            print(f"   Found: {job['Job Title']} ({job['ID']})")

    print("\n✅ DataLoader test complete!")
