"""
LLM Response Generator using Gemini Pro
Formats search results into natural language responses
"""

from typing import Any, Dict, List, Optional

import google.generativeai as genai

from src.config import config


class ResponseGenerator:
    """
    Generates natural language responses using Gemini Pro
    Takes search results and creates helpful, conversational output
    """

    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize Gemini Pro for response generation

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

        # Initialize Gemini Pro for text generation
        self.model = genai.GenerativeModel(
            config.LLM_MODEL,
            generation_config=genai.types.GenerationConfig(
                temperature=config.LLM_TEMPERATURE,
                max_output_tokens=config.LLM_MAX_OUTPUT_TOKENS,
            ),
        )

        print("✅ Response Generator initialized")
        print(f"   Model: {config.LLM_MODEL}")
        print(f"   Temperature: {config.LLM_TEMPERATURE}")

    def generate_response(
        self, results: Dict[str, Any], response_type: str = "brief"
    ) -> str:
        """
        Generate a natural language response from search results

        Args:
            results: Results from JobRetriever.retrieve()
            response_type: "detailed", "summary", or "brief"

        Returns:
            Natural language response
        """
        jobs = results.get("jobs", [])
        query_info = results.get("query_info", {})
        search_info = results.get("search_info", {})

        if not jobs:
            return self._generate_no_results_response(query_info)

        if response_type == "brief":
            return self._generate_brief_response(jobs, query_info)
        elif response_type == "summary":
            return self._generate_summary_response(jobs, query_info, search_info)
        else:  # detailed
            return self._generate_detailed_response(jobs, query_info, search_info)

    def _generate_detailed_response(
        self,
        jobs: List[Dict[str, Any]],
        query_info: Dict[str, Any],
        search_info: Dict[str, Any],
    ) -> str:
        """Generate detailed response with job descriptions"""

        prompt = self._build_detailed_prompt(jobs, query_info, search_info)

        try:
            response = self.model.generate_content(prompt)
            return response.text.strip()
        except Exception as e:
            print(f"⚠️  LLM response generation failed: {e}")
            return self._generate_fallback_response(jobs, query_info)

    def _generate_summary_response(
        self,
        jobs: List[Dict[str, Any]],
        query_info: Dict[str, Any],
        search_info: Dict[str, Any],
    ) -> str:
        """Generate summary response with highlights"""

        prompt = self._build_summary_prompt(jobs, query_info, search_info)

        try:
            response = self.model.generate_content(prompt)
            return response.text.strip()
        except Exception as e:
            print(f"⚠️  LLM response generation failed: {e}")
            return self._generate_fallback_response(jobs, query_info)

    def _generate_brief_response(
        self, jobs: List[Dict[str, Any]], query_info: Dict[str, Any]
    ) -> str:
        """Generate brief list response"""

        lines = [f'Found {len(jobs)} job(s) for "{query_info["original_query"]}":\n']

        for job in jobs:
            lines.append(
                f"{job['rank']}. {job['job_title']} at {job['company']} "
                f"({job['location']}) - {job['relevance_label']}"
            )

        return "\n".join(lines)

    def _generate_no_results_response(self, query_info: Dict[str, Any]) -> str:
        """Generate response when no results found"""

        prompt = f"""You are a helpful job search assistant. No jobs were found for the user's query.

User Query: "{query_info["original_query"]}"
Filters Applied: {query_info.get("filters", {})}

Generate a helpful response that:
1. Politely acknowledges no results were found
2. Suggests what the user could try (broader search, different keywords, fewer filters)
3. Remains encouraging and professional
4. Keep it brief (2-3 sentences)

Response:"""

        try:
            response = self.model.generate_content(prompt)
            return response.text.strip()
        except Exception as e:
            print(f"⚠️  LLM response generation failed: {e}")
            return (
                f'I couldn\'t find any jobs matching "{query_info["original_query"]}". '
                "Try using different keywords or removing some filters for broader results."
            )

    def _build_detailed_prompt(
        self,
        jobs: List[Dict[str, Any]],
        query_info: Dict[str, Any],
        search_info: Dict[str, Any],
    ) -> str:
        """Build prompt for detailed response"""

        # Format jobs data
        jobs_text = self._format_jobs_for_prompt(jobs, include_snippets=True)

        prompt = f"""You are an expert job search assistant. A user searched for jobs and you found {len(jobs)} relevant matches.

USER QUERY: "{query_info["original_query"]}"

SEARCH DETAILS:
- Semantic search: "{query_info["semantic_query"]}"
- Filters applied: {query_info.get("filters", "None")}
- Total results found: {len(jobs)}

TOP JOB MATCHES:
{jobs_text}

TASK:
Generate a helpful, conversational response that:

1. **Introduction** (1-2 sentences):
   - Acknowledge the search and number of results found
   - Mention any filters that were applied

2. **Job Highlights** (for each job):
   - Job title and company
   - Key reasons why it's a good match
   - Notable requirements or benefits (if mentioned in snippet)
   - Relevance indicator

3. **Recommendations** (1-2 sentences):
   - Suggest next steps (review details, apply, etc.)
   - Mention if they want to refine search

STYLE:
- Professional but friendly and conversational
- Focus on what makes each job relevant to their search
- Use bullet points for clarity
- Keep descriptions concise but informative
- Highlight variety in results if applicable (different companies, locations, etc.)

Response:"""

        return prompt

    def _build_summary_prompt(
        self,
        jobs: List[Dict[str, Any]],
        query_info: Dict[str, Any],
        search_info: Dict[str, Any],
    ) -> str:
        """Build prompt for summary response"""

        jobs_text = self._format_jobs_for_prompt(jobs, include_snippets=False)

        prompt = f"""You are a job search assistant. Provide a summary of search results.

USER QUERY: "{query_info["original_query"]}"
RESULTS: {len(jobs)} jobs found

JOBS:
{jobs_text}

Generate a concise summary (3-4 sentences) that:
1. States how many relevant jobs were found
2. Highlights the most relevant match
3. Mentions variety/commonalities (companies, locations, levels)
4. Suggests reviewing the details

Keep it brief and actionable.

Summary:"""

        return prompt

    def _format_jobs_for_prompt(
        self, jobs: List[Dict[str, Any]], include_snippets: bool = True
    ) -> str:
        """Format jobs list for inclusion in prompt"""

        lines = []

        for job in jobs:
            lines.append(f"\n{job['rank']}. {job['job_title']}")
            lines.append(f"   Company: {job['company']}")
            lines.append(f"   Location: {job['location']}")
            lines.append(f"   Level: {job['job_level']}")
            lines.append(f"   Category: {job['category']}")
            lines.append(
                f"   Relevance: {job['relevance_label']} ({job['similarity_percentage']}%)"
            )

            if include_snippets and job.get("snippet"):
                snippet = job["snippet"][:300]  # Limit snippet length
                lines.append(f"   Key Details: {snippet}...")

        return "\n".join(lines)

    def _generate_fallback_response(
        self, jobs: List[Dict[str, Any]], query_info: Dict[str, Any]
    ) -> str:
        """Generate fallback response if LLM fails"""

        lines = [
            f'Found {len(jobs)} job(s) matching "{query_info["original_query"]}":',
            "",
        ]

        for job in jobs[:5]:  # Limit to top 5
            lines.append(f"{job['rank']}. **{job['job_title']}** at {job['company']}")
            lines.append(f"   📍 {job['location']} | 🎯 {job['job_level']}")
            lines.append(
                f"   ⭐ {job['relevance_label']} ({job['similarity_percentage']}%)"
            )
            lines.append("")

        if len(jobs) > 5:
            lines.append(f"... and {len(jobs) - 5} more results")

        return "\n".join(lines)

    def generate_job_detail_response(self, job: Dict[str, Any]) -> str:
        """
        Generate detailed response for a single job

        Args:
            job: Job dictionary with full details

        Returns:
            Natural language description
        """
        prompt = f"""You are a job search assistant. Provide a comprehensive overview of this job posting.

JOB DETAILS:
Title: {job["job_title"]}
Company: {job["company"]}
Location: {job["location"]}
Level: {job["job_level"]}
Category: {job["category"]}
Posted: {job.get("publication_date", "Recently")}
Tags: {job.get("tags", "N/A")}

FULL DESCRIPTION:
{job.get("full_description", job.get("snippet", "No description available"))}

Generate a well-structured overview that includes:
1. **Summary** (2-3 sentences): What the role is about
2. **Key Responsibilities**: Main duties (bullet points)
3. **Requirements**: Skills and qualifications needed (bullet points)
4. **What Makes This Role Attractive**: Highlights/benefits (if mentioned)

Keep it professional, clear, and well-organized.

Overview:"""

        try:
            response = self.model.generate_content(prompt)
            return response.text.strip()
        except Exception as e:
            print(f"⚠️  LLM response generation failed: {e}")
            return f"""**{job["job_title"]}** at {job["company"]}

📍 Location: {job["location"]}
🎯 Level: {job["job_level"]}
📂 Category: {job["category"]}
🏷️ Tags: {job.get("tags", "N/A")}

{job.get("snippet", "No description available")}"""


# Convenience function
def generate_response(results: Dict[str, Any], response_type: str = "detailed") -> str:
    """
    Convenience function to generate response

    Args:
        results: Search results from JobRetriever
        response_type: "detailed", "summary", or "brief"

    Returns:
        Natural language response
    """
    generator = ResponseGenerator()
    return generator.generate_response(results, response_type)


# Testing and demonstration
if __name__ == "__main__":
    print("Testing ResponseGenerator...\n")

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
        print("TEST 1: Initialize ResponseGenerator")
        print("=" * 70)

        generator = ResponseGenerator()

        print("\n" + "=" * 70)
        print("TEST 2: Generate Response with Mock Results")
        print("=" * 70)

        # Create mock search results
        mock_results = {
            "jobs": [
                {
                    "rank": 1,
                    "job_id": "1",
                    "job_title": "Senior Python Developer",
                    "company": "Tech Corp",
                    "location": "San Francisco, CA",
                    "job_level": "Senior Level",
                    "category": "Software Engineering",
                    "publication_date": "2025-11-10",
                    "tags": "Python, Django, AWS, Docker",
                    "relevance_score": 0.87,
                    "relevance_label": "Highly Relevant",
                    "similarity_percentage": 87.0,
                    "matched_chunks": 15,
                    "snippet": "We are seeking a Senior Python Developer with expertise in Django and AWS. "
                    "You will design scalable backend services, mentor junior developers, and "
                    "optimize system performance. Requirements include 5+ years Python experience.",
                },
                {
                    "rank": 2,
                    "job_id": "2",
                    "job_title": "Python Backend Engineer",
                    "company": "Startup Inc",
                    "location": "Remote",
                    "job_level": "Mid Level",
                    "category": "Software Engineering",
                    "publication_date": "2025-11-12",
                    "tags": "Python, FastAPI, PostgreSQL",
                    "relevance_score": 0.75,
                    "relevance_label": "Relevant",
                    "similarity_percentage": 75.0,
                    "matched_chunks": 12,
                    "snippet": "Join our fast-growing startup as a Python Backend Engineer. Build APIs with "
                    "FastAPI, work with PostgreSQL databases, and collaborate with a remote team. "
                    "3+ years experience required.",
                },
            ],
            "query_info": {
                "original_query": "Python developer jobs",
                "semantic_query": "Python developer",
                "filters": {},
            },
            "search_info": {
                "total_chunks_found": 27,
                "unique_jobs_found": 2,
                "filters_applied": {},
                "requested_top_k": 5,
            },
        }

        print("\nGenerating DETAILED response...")
        print("-" * 70)
        detailed = generator.generate_response(mock_results, response_type="detailed")
        print(detailed)

        print("\n" + "=" * 70)
        print("TEST 3: Generate Summary Response")
        print("=" * 70)

        print("\nGenerating SUMMARY response...")
        print("-" * 70)
        summary = generator.generate_response(mock_results, response_type="summary")
        print(summary)

        print("\n" + "=" * 70)
        print("TEST 4: Generate Brief Response")
        print("=" * 70)

        print("\nGenerating BRIEF response...")
        print("-" * 70)
        brief = generator.generate_response(mock_results, response_type="brief")
        print(brief)

        print("\n" + "=" * 70)
        print("TEST 5: No Results Response")
        print("=" * 70)

        no_results = {
            "jobs": [],
            "query_info": {
                "original_query": "quantum computing engineer",
                "semantic_query": "quantum computing engineer",
                "filters": {"category": "Software Engineering"},
            },
            "search_info": {"total_chunks_found": 0, "unique_jobs_found": 0},
        }

        print("\nGenerating NO RESULTS response...")
        print("-" * 70)
        no_results_resp = generator.generate_response(no_results)
        print(no_results_resp)

        print("\n" + "=" * 70)
        print("TEST 6: Job Detail Response")
        print("=" * 70)

        mock_job = {
            "job_id": "1",
            "job_title": "Senior Python Developer",
            "company": "Tech Corp",
            "location": "San Francisco, CA",
            "job_level": "Senior Level",
            "category": "Software Engineering",
            "publication_date": "2025-11-10",
            "tags": "Python, Django, AWS",
            "full_description": """
Senior Python Developer - Tech Corp

We are seeking an experienced Python developer to join our backend team.

Responsibilities:
- Design and implement scalable backend services using Django
- Optimize database queries and system performance
- Mentor junior developers and conduct code reviews
- Collaborate with frontend team on API design

Requirements:
- 5+ years of Python development experience
- Strong knowledge of Django and REST APIs
- Experience with AWS services (EC2, S3, Lambda)
- PostgreSQL database expertise
- Docker and Kubernetes knowledge

Benefits:
- Competitive salary and equity
- Remote work flexibility
- Health insurance and 401k
- Professional development budget
            """,
        }

        print("\nGenerating JOB DETAIL response...")
        print("-" * 70)
        detail = generator.generate_job_detail_response(mock_job)
        print(detail)

        print("\n" + "=" * 70)
        print("✅ All ResponseGenerator tests complete!")
        print("=" * 70)

    except ValueError as e:
        print(f"\n❌ Error: {e}")
        print("\nPlease configure your Gemini API key in the .env file")
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        import traceback

        traceback.print_exc()
