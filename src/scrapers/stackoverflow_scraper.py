"""
Stack Overflow Scraper for VBScript and Selenium Q&A Pairs
Collects high-quality question-answer pairs for training
"""

import os
import json
import time
import html
import re
from pathlib import Path
from typing import Generator, Dict, Any, List, Optional
from dataclasses import dataclass, asdict
from datetime import datetime

import requests
from bs4 import BeautifulSoup
from loguru import logger
from tqdm import tqdm
from dotenv import load_dotenv

load_dotenv()


@dataclass
class QAPair:
    """Represents a question-answer pair."""
    source: str
    question_id: int
    title: str
    question: str
    answer: str
    code_blocks: List[str]
    tags: List[str]
    score: int
    answer_score: int
    url: str
    collected_at: str


class StackOverflowScraper:
    """Scrapes VBScript and Selenium Q&A from Stack Overflow."""

    BASE_URL = "https://api.stackexchange.com/2.3"

    # Tags to search for
    TAGS = {
        "vbscript": [
            "vbscript",
            "qtp",
            "uft",
            "hp-uft",
            "microfocus-uft",
            "vbscript-automation",
        ],
        "selenium": [
            "selenium",
            "selenium-webdriver",
            "selenium-python",
            "selenium-java",
            "selenium-chromedriver",
            "webdriver",
            "python-selenium",
        ],
    }

    def __init__(self, api_key: str = None, output_dir: str = "./data/raw"):
        """
        Initialize Stack Overflow scraper.

        Args:
            api_key: Stack Exchange API key (optional, increases quota)
            output_dir: Directory to save scraped data
        """
        self.api_key = api_key or os.getenv("STACKEXCHANGE_API_KEY")
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.qa_pairs: List[QAPair] = []
        self.collected_ids = set()

        logger.info(f"Stack Overflow Scraper initialized. Output: {self.output_dir}")

    def _make_request(self, endpoint: str, params: Dict[str, Any]) -> Dict:
        """Make API request with rate limiting."""
        url = f"{self.BASE_URL}/{endpoint}"

        params["site"] = "stackoverflow"
        # Use the actual filter ID that includes body content
        params["filter"] = "!nNPvSNdWme"  # Filter that includes body

        if self.api_key:
            params["key"] = self.api_key

        try:
            response = requests.get(url, params=params, timeout=30)
            response.raise_for_status()

            data = response.json()

            # Log quota for debugging
            if "quota_remaining" in data:
                logger.info(f"API quota remaining: {data['quota_remaining']}")
                if data["quota_remaining"] < 100:
                    logger.warning(f"API quota low: {data['quota_remaining']} remaining")

            # Handle backoff
            if "backoff" in data:
                logger.info(f"Backing off for {data['backoff']} seconds")
                time.sleep(data["backoff"])

            return data

        except requests.exceptions.RequestException as e:
            logger.error(f"API request failed: {e}")
            return {"items": []}

    def _clean_html(self, html_content: str) -> str:
        """Clean HTML content and extract text."""
        if not html_content:
            return ""

        # Parse HTML
        soup = BeautifulSoup(html_content, 'html.parser')

        # Extract code blocks separately
        code_blocks = []
        for code in soup.find_all(['code', 'pre']):
            code_blocks.append(code.get_text())
            code.replace_with(f"[CODE_BLOCK_{len(code_blocks)-1}]")

        # Get text
        text = soup.get_text(separator='\n')

        # Restore code blocks
        for i, code in enumerate(code_blocks):
            text = text.replace(f"[CODE_BLOCK_{i}]", f"\n```\n{code}\n```\n")

        # Clean up
        text = html.unescape(text)
        text = re.sub(r'\n{3,}', '\n\n', text)
        text = text.strip()

        return text

    def _extract_code_blocks(self, html_content: str) -> List[str]:
        """Extract all code blocks from HTML content."""
        if not html_content:
            return []

        soup = BeautifulSoup(html_content, 'html.parser')
        code_blocks = []

        for code in soup.find_all(['pre', 'code']):
            code_text = code.get_text().strip()
            if len(code_text) > 20:  # Filter out tiny snippets
                code_blocks.append(code_text)

        return code_blocks

    def search_questions(
        self,
        tags: List[str],
        min_score: int = 0,
        min_answers: int = 1,
        page_size: int = 100,
        max_pages: int = 5
    ) -> Generator[Dict[str, Any], None, None]:
        """
        Search for questions with specified tags.

        Args:
            tags: List of tags to search
            min_score: Minimum question score
            min_answers: Minimum number of answers
            page_size: Results per page
            max_pages: Maximum pages to fetch
        """
        collected_ids = set()

        for tag in tags:
            logger.info(f"Searching tag: {tag}")

            for page in range(1, max_pages + 1):
                params = {
                    "tagged": tag,
                    "sort": "votes",
                    "order": "desc",
                    "pagesize": page_size,
                    "page": page,
                }

                data = self._make_request("questions", params)

                if not data.get("items"):
                    logger.warning(f"No results for tag {tag} page {page}")
                    break

                logger.info(f"Found {len(data['items'])} questions for {tag} page {page}")

                for question in data["items"]:
                    qid = question["question_id"]
                    if qid in collected_ids:
                        continue
                    collected_ids.add(qid)

                    if question["score"] >= min_score and question.get("answer_count", 0) >= min_answers:
                        yield question

                time.sleep(0.5)  # Rate limiting

                # Check if more pages available
                if not data.get("has_more", False):
                    break

    def search_questions_old(
        self,
        tags: List[str],
        min_score: int = 5,
        min_answers: int = 1,
        page_size: int = 100,
        max_pages: int = 10
    ) -> Generator[Dict[str, Any], None, None]:
        """Old version - kept for reference."""
        for page in range(1, max_pages + 1):
            params = {
                "tagged": ";".join(tags),
                "sort": "votes",
                "order": "desc",
                "pagesize": page_size,
                "page": page,
            }

            data = self._make_request("questions", params)

            if not data.get("items"):
                break

            for question in data["items"]:
                if question["score"] >= min_score and question.get("answer_count", 0) >= min_answers:
                    yield question

            # Check if more pages available
            if not data.get("has_more", False):
                break

            time.sleep(0.5)  # Rate limiting

    def get_answers(self, question_id: int) -> List[Dict[str, Any]]:
        """Get answers for a specific question."""
        params = {
            "order": "desc",
            "sort": "votes",
            "filter": "withbody",
        }

        data = self._make_request(f"questions/{question_id}/answers", params)
        return data.get("items", [])

    def collect_qa_pairs(
        self,
        category: str,
        tags: List[str],
        max_questions: int = 500,
        min_score: int = 0,
        min_answer_score: int = 1
    ):
        """
        Collect Q&A pairs for a category.

        Args:
            category: Category name (vbscript, selenium)
            tags: Tags to search
            max_questions: Maximum questions to collect
            min_score: Minimum question score
            min_answer_score: Minimum accepted answer score
        """
        logger.info(f"Collecting {category} Q&A pairs...")

        collected = 0
        questions_checked = 0

        for question in tqdm(
            self.search_questions(tags, min_score=min_score),
            desc=f"{category} questions",
            total=max_questions
        ):
            if collected >= max_questions:
                break

            questions_checked += 1
            question_id = question["question_id"]

            if question_id in self.collected_ids:
                continue

            self.collected_ids.add(question_id)

            # Get the best answer
            answers = self.get_answers(question_id)

            if not answers:
                continue

            # Prefer accepted answer, then highest scored
            best_answer = None
            for answer in answers:
                if answer.get("is_accepted", False):
                    best_answer = answer
                    break
                elif best_answer is None or answer.get("score", 0) > best_answer.get("score", 0):
                    best_answer = answer

            if not best_answer or best_answer.get("score", 0) < min_answer_score:
                continue

            # Extract content - handle missing body gracefully
            question_body = question.get("body", "")
            answer_body = best_answer.get("body", "")

            if not answer_body:
                continue

            question_text = self._clean_html(question_body)
            answer_text = self._clean_html(answer_body)
            code_blocks = self._extract_code_blocks(answer_body)

            # Include even if no explicit code blocks - some answers have inline code
            qa_pair = QAPair(
                source="stackoverflow",
                question_id=question_id,
                title=html.unescape(question.get("title", "")),
                question=question_text if question_text else html.unescape(question.get("title", "")),
                answer=answer_text,
                code_blocks=code_blocks,
                tags=question.get("tags", []),
                score=question.get("score", 0),
                answer_score=best_answer.get("score", 0),
                url=question.get("link", f"https://stackoverflow.com/q/{question_id}"),
                collected_at=datetime.now().isoformat(),
            )

            self.qa_pairs.append(qa_pair)
            collected += 1

            if collected % 10 == 0:
                logger.info(f"Collected {collected} {category} Q&A pairs (checked {questions_checked})")

            time.sleep(0.3)  # Rate limiting

        logger.info(f"Collected {collected} {category} Q&A pairs (total checked: {questions_checked})")

    def collect_all(self, max_per_category: int = 500):
        """Collect Q&A pairs for all categories."""
        for category, tags in self.TAGS.items():
            self.collect_qa_pairs(
                category=category,
                tags=tags,
                max_questions=max_per_category
            )

    def save_qa_pairs(self, filename: str = None) -> Path:
        """Save collected Q&A pairs to JSONL file."""
        if not self.qa_pairs:
            logger.warning("No Q&A pairs to save")
            return None

        if filename is None:
            filename = f"stackoverflow_qa_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jsonl"

        output_path = self.output_dir / "stackoverflow" / filename
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, 'w', encoding='utf-8') as f:
            for qa in self.qa_pairs:
                f.write(json.dumps(asdict(qa), ensure_ascii=False) + '\n')

        logger.info(f"Saved {len(self.qa_pairs)} Q&A pairs to {output_path}")

        # Summary by tags
        tag_counts = {}
        for qa in self.qa_pairs:
            for tag in qa.tags:
                tag_counts[tag] = tag_counts.get(tag, 0) + 1

        top_tags = sorted(tag_counts.items(), key=lambda x: x[1], reverse=True)[:10]
        logger.info(f"Top tags: {top_tags}")

        return output_path

    def run_full_collection(self, max_per_category: int = 500) -> Path:
        """Run full collection pipeline."""
        logger.info("Starting Stack Overflow collection...")

        self.collect_all(max_per_category=max_per_category)
        output_path = self.save_qa_pairs()

        logger.info(f"Collection complete! Total Q&A pairs: {len(self.qa_pairs)}")
        return output_path


def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(description="Scrape VBScript and Selenium Q&A from Stack Overflow")
    parser.add_argument("--api-key", help="Stack Exchange API key")
    parser.add_argument("--output", default="./data/raw", help="Output directory")
    parser.add_argument("--max-per-category", type=int, default=500, help="Max Q&A pairs per category")
    args = parser.parse_args()

    scraper = StackOverflowScraper(api_key=args.api_key, output_dir=args.output)
    scraper.run_full_collection(max_per_category=args.max_per_category)


if __name__ == "__main__":
    main()
