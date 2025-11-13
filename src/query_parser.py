"""
Query parser module using Gemini LLM
Parses natural language queries to extract filters and search intent
"""

import json
from datetime import datetime, timedelta
from typing import Any, Dict, Optional

import google.generativeai as genai

from src.config import config


class QueryParser:
    """
    Parses natural language job search queries using Gemini LLM
    Extracts filters (category, company, location, job_level) and semantic intent
    """

    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize Gemini API for query parsing

        Args:
            api_key: Gemini API key (defaults to config.GEMINI_API_KEY)
        """
        self.api_key = api_key or config.GEMINI_API_KEY

        if not self.api_key or self.api_key == "your_gemini_api_key_here":
            raise ValueError(
                "Gemini API key not configured!\n"
                "Please set GEMINI_API_KEY in your .env file.\n"
                "Get your key from: https://makersuite.google.com/app/apikey"
            )

        # Configure Gemini
        genai.configure(api_key=self.api_key)

        # Initialize model for parsing
        self.model = genai.GenerativeModel(config.LLM_MODEL)

        # Load available values from config
        self.job_levels = config.JOB_LEVELS
        self.job_level_keywords = config.JOB_LEVEL_KEYWORDS
        self.date_keywords = config.DATE_FILTER_KEYWORDS

        print("✅ Query Parser initialized")
        print(f"   Model: {config.LLM_MODEL}")

    def parse_query(self, query: str) -> Dict[str, Any]:
        """
        Parse a natural language query into structured filters and search intent

        Args:
            query: User's natural language query

        Returns:
            Dictionary with:
                - semantic_query: cleaned query for semantic search
                - filters: dict of metadata filters
                - original_query: the original query
        """
        if not query or not query.strip():
            return {"semantic_query": "", "filters": {}, "original_query": query}

        # First, try rule-based extraction for simple patterns
        filters = self._extract_filters_rule_based(query)

        # Then use LLM for more complex parsing
        llm_result = self._parse_with_llm(query)

        # Merge results (LLM takes precedence)
        if llm_result:
            filters.update(llm_result.get("filters", {}))
            semantic_query = llm_result.get("semantic_query", query)
        else:
            semantic_query = self._clean_query_for_search(query, filters)

        return {
            "semantic_query": semantic_query,
            "filters": filters,
            "original_query": query,
        }

    def _extract_filters_rule_based(self, query: str) -> Dict[str, Any]:
        """
        Extract filters using rule-based pattern matching

        Args:
            query: User query

        Returns:
            Dictionary of filters
        """
        filters = {}
        query_lower = query.lower()

        # Extract job level
        for keyword, level in self.job_level_keywords.items():
            if keyword in query_lower:
                filters["job_level"] = level
                break

        # Extract date filters
        for keyword, days in self.date_keywords.items():
            if keyword in query_lower:
                # Calculate cutoff date
                cutoff_date = datetime.now() - timedelta(days=days)
                filters["publication_date_after"] = cutoff_date.strftime("%Y-%m-%d")
                break

        return filters

    def _parse_with_llm(self, query: str) -> Optional[Dict[str, Any]]:
        """
        Use Gemini LLM to parse the query

        Args:
            query: User query

        Returns:
            Parsed result or None if parsing fails
        """
        prompt = self._build_parsing_prompt(query)

        try:
            response = self.model.generate_content(
                prompt,
                generation_config=genai.types.GenerationConfig(
                    temperature=0.1,  # Low temperature for consistent parsing
                    max_output_tokens=500,
                ),
            )

            # Extract JSON from response
            result = self._extract_json_from_response(response.text)

            return result

        except Exception as e:
            print(f"⚠️  LLM parsing failed: {e}")
            return None

    def _build_parsing_prompt(self, query: str) -> str:
        """
        Build the prompt for LLM query parsing

        Args:
            query: User query

        Returns:
            Formatted prompt
        """
        prompt = f"""You are a query parser for a job search system. Parse the user's natural language query into structured filters and semantic search intent.

User Query: "{query}"

Available Job Levels:
- Senior Level
- Mid Level
- Entry Level
- Internship

Available Job Categories (examples):
- Software Engineering
- Data and Analytics
- Design and UX
- Sales
- Project Management
- Advertising and Marketing
- General

Instructions:
1. Extract the SEMANTIC SEARCH INTENT (what skills, roles, technologies the user wants)
2. Extract FILTERS (job level, category, company, location if mentioned)
3. Only include filters that are explicitly mentioned or strongly implied
4. Keep semantic query focused on skills, roles, and technologies

Return ONLY a JSON object in this exact format:
{{
    "semantic_query": "core search terms focusing on skills and role",
    "filters": {{
        "category": "category name if mentioned or null",
        "job_level": "job level if mentioned or null",
        "company": "company name if mentioned or null",
        "location": "location if mentioned or null"
    }}
}}

Examples:

Query: "Senior Python developer jobs in San Francisco"
{{
    "semantic_query": "Python developer",
    "filters": {{
        "job_level": "Senior Level",
        "location": "San Francisco"
    }}
}}

Query: "Looking for data scientist positions at Google"
{{
    "semantic_query": "data scientist",
    "filters": {{
        "category": "Data and Analytics",
        "company": "Google"
    }}
}}

Query: "Entry level frontend developer with React experience"
{{
    "semantic_query": "frontend developer React",
    "filters": {{
        "job_level": "Entry Level",
        "category": "Software Engineering"
    }}
}}

Query: "UX designer roles"
{{
    "semantic_query": "UX designer",
    "filters": {{
        "category": "Design and UX"
    }}
}}

Now parse the user's query. Return ONLY the JSON object, no other text."""

        return prompt

    def _extract_json_from_response(
        self, response_text: str
    ) -> Optional[Dict[str, Any]]:
        """
        Extract JSON object from LLM response

        Args:
            response_text: LLM response text

        Returns:
            Parsed JSON dict or None
        """
        try:
            # Try to find JSON object in response
            # Look for content between first { and last }
            start = response_text.find("{")
            end = response_text.rfind("}") + 1

            if start != -1 and end > start:
                json_str = response_text[start:end]
                result = json.loads(json_str)

                # Clean up filters (remove null values)
                if "filters" in result:
                    result["filters"] = {
                        k: v
                        for k, v in result["filters"].items()
                        if v is not None and v != "" and v != "null"
                    }

                return result
            else:
                print("⚠️  No JSON found in LLM response")
                return None

        except json.JSONDecodeError as e:
            print(f"⚠️  JSON parsing error: {e}")
            print(f"Response: {response_text[:200]}")
            return None

    def _clean_query_for_search(self, query: str, filters: Dict[str, Any]) -> str:
        """
        Clean query by removing filter keywords to get semantic search query

        Args:
            query: Original query
            filters: Extracted filters

        Returns:
            Cleaned query for semantic search
        """
        cleaned = query.lower()

        # Remove job level keywords
        for keyword in self.job_level_keywords.keys():
            cleaned = cleaned.replace(keyword, "")

        # Remove date keywords
        for keyword in self.date_keywords.keys():
            cleaned = cleaned.replace(keyword, "")

        # Remove location patterns (in, at, near)
        if "location" in filters:
            location = filters["location"].lower()
            cleaned = cleaned.replace(f"in {location}", "")
            cleaned = cleaned.replace(f"at {location}", "")
            cleaned = cleaned.replace(f"near {location}", "")
            cleaned = cleaned.replace(location, "")

        # Remove company patterns
        if "company" in filters:
            company = filters["company"].lower()
            cleaned = cleaned.replace(f"at {company}", "")
            cleaned = cleaned.replace(company, "")

        # Remove common filler words
        filler_words = [
            "jobs",
            "job",
            "position",
            "positions",
            "role",
            "roles",
            "looking for",
            "find",
            "search for",
            "want",
            "need",
        ]
        for word in filler_words:
            cleaned = cleaned.replace(word, "")

        # Clean up whitespace
        cleaned = " ".join(cleaned.split())

        return cleaned.strip() or query

    def validate_filters(self, filters: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate and normalize filter values

        Args:
            filters: Raw filters

        Returns:
            Validated filters
        """
        validated = {}

        # Validate job_level
        if "job_level" in filters:
            level = filters["job_level"]
            if level in self.job_levels:
                validated["job_level"] = level
            else:
                # Try to map it
                level_lower = level.lower()
                for keyword, mapped_level in self.job_level_keywords.items():
                    if keyword in level_lower:
                        validated["job_level"] = mapped_level
                        break

        # Pass through other filters (they'll be validated against data)
        for key in ["category", "company", "location", "publication_date_after"]:
            if key in filters and filters[key]:
                validated[key] = filters[key]

        return validated

    def get_filter_summary(self, parsed: Dict[str, Any]) -> str:
        """
        Generate human-readable summary of parsed query

        Args:
            parsed: Parsed query result

        Returns:
            Summary string
        """
        parts = []

        if parsed["semantic_query"]:
            parts.append(f"Searching for: {parsed['semantic_query']}")

        filters = parsed.get("filters", {})

        if filters:
            filter_parts = []

            if "category" in filters:
                filter_parts.append(f"Category: {filters['category']}")

            if "job_level" in filters:
                filter_parts.append(f"Level: {filters['level']}")

            if "company" in filters:
                filter_parts.append(f"Company: {filters['company']}")

            if "location" in filters:
                filter_parts.append(f"Location: {filters['location']}")

            if "publication_date_after" in filters:
                filter_parts.append(
                    f"Posted after: {filters['publication_date_after']}"
                )

            if filter_parts:
                parts.append("Filters: " + ", ".join(filter_parts))

        return " | ".join(parts) if parts else "No filters applied"


# Convenience function
def parse_query(query: str) -> Dict[str, Any]:
    """
    Convenience function to parse a query

    Args:
        query: User query

    Returns:
        Parsed result
    """
    parser = QueryParser()
    return parser.parse_query(query)


# Testing and demonstration
if __name__ == "__main__":
    print("Testing QueryParser...\n")

    # Check if API key is configured
    if not config.GEMINI_API_KEY or config.GEMINI_API_KEY == "your_gemini_api_key_here":
        print("=" * 70)
        print("⚠️  GEMINI API KEY NOT CONFIGURED")
        print("=" * 70)
        print("\nPlease add your Gemini API key to .env file:")
        print("GEMINI_API_KEY=your_actual_key_here")
        print("\nGet your key from: https://makersuite.google.com/app/apikey")
        print("\nCannot run tests without API key.")
        exit(1)

    try:
        print("=" * 70)
        print("TEST 1: Initialize QueryParser")
        print("=" * 70)

        parser = QueryParser()

        print("\n" + "=" * 70)
        print("TEST 2: Simple Queries (Rule-based)")
        print("=" * 70)

        simple_queries = [
            "senior Python developer",
            "entry level data scientist",
            "internship frontend developer",
            "mid level UX designer",
        ]

        for query in simple_queries:
            print(f'\n📝 Query: "{query}"')
            result = parser.parse_query(query)
            print(f"   Semantic: {result['semantic_query']}")
            print(f"   Filters: {result['filters']}")

        print("\n" + "=" * 70)
        print("TEST 3: Complex Queries (LLM-based)")
        print("=" * 70)

        complex_queries = [
            "Looking for senior Python developer jobs in San Francisco",
            "Entry level data scientist positions at Google",
            "Frontend developer with React experience, remote",
            "UX designer roles at tech companies in New York",
            "Software engineering internships",
            "Recent project manager jobs",
        ]

        for query in complex_queries:
            print(f'\n📝 Query: "{query}"')
            result = parser.parse_query(query)
            print(f"   Semantic: {result['semantic_query']}")
            print(f"   Filters: {result['filters']}")
            print(f"   Summary: {parser.get_filter_summary(result)}")

        print("\n" + "=" * 70)
        print("TEST 4: Queries with Locations")
        print("=" * 70)

        location_queries = [
            "Python developer in Seattle",
            "Data analyst jobs near Boston",
            "Software engineer at Microsoft in Redmond",
        ]

        for query in location_queries:
            print(f'\n📝 Query: "{query}"')
            result = parser.parse_query(query)
            print(f"   Semantic: {result['semantic_query']}")
            print(f"   Location filter: {result['filters'].get('location', 'None')}")

        print("\n" + "=" * 70)
        print("TEST 5: Queries with Companies")
        print("=" * 70)

        company_queries = [
            "Jobs at Apple",
            "Software engineer positions at Google",
            "Data scientist roles at Microsoft",
        ]

        for query in company_queries:
            print(f'\n📝 Query: "{query}"')
            result = parser.parse_query(query)
            print(f"   Semantic: {result['semantic_query']}")
            print(f"   Company filter: {result['filters'].get('company', 'None')}")

        print("\n" + "=" * 70)
        print("TEST 6: Category Detection")
        print("=" * 70)

        category_queries = [
            "Machine learning engineer",
            "UX/UI designer",
            "Sales representative",
            "Project manager with Agile experience",
        ]

        for query in category_queries:
            print(f'\n📝 Query: "{query}"')
            result = parser.parse_query(query)
            print(f"   Semantic: {result['semantic_query']}")
            print(f"   Category: {result['filters'].get('category', 'None')}")

        print("\n✅ All QueryParser tests complete!")

    except ValueError as e:
        print(f"\n❌ Error: {e}")
        print("\nPlease configure your Gemini API key in the .env file")
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        import traceback

        traceback.print_exc()
