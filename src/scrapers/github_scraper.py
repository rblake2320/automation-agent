"""
GitHub Scraper for VBScript and Selenium Code Examples
Collects training data from public repositories
"""

import os
import json
import time
import base64
from pathlib import Path
from typing import Generator, Dict, Any, List
from dataclasses import dataclass, asdict
from datetime import datetime

import requests
from github import Github, RateLimitExceededException
from loguru import logger
from tqdm import tqdm
from dotenv import load_dotenv

load_dotenv()


@dataclass
class CodeSample:
    """Represents a collected code sample."""
    source: str
    repo_name: str
    file_path: str
    language: str
    code: str
    description: str
    url: str
    stars: int
    collected_at: str


class GitHubScraper:
    """Scrapes VBScript and Selenium code from GitHub repositories."""

    # Search queries for different code types
    SEARCH_QUERIES = {
        "vbscript_uft": [
            "UFT VBScript automation",
            "QTP VBScript test",
            "UFT One script",
            "Micro Focus UFT",
            "QuickTest Professional VBScript",
            "UFT DataTable",
            "UFT Reporter",
            "UFT SystemUtil",
            "Browser Page WebEdit UFT",
            "UFT Object Repository",
            "UFT descriptive programming",
            "UFT checkpoint verification",
            "UFT recovery scenario",
            "UFT action template",
            "UFT function library",
        ],
        "vbscript_general": [
            "VBScript automation",
            "VBScript WScript",
            "VBScript FileSystemObject",
            "VBScript COM automation",
            "VBScript Excel automation",
            "VBScript SAP GUI",
            "VBScript WMI query",
            "VBScript ADODB connection",
            "VBScript regex pattern",
            "VBScript error handling",
        ],
        "selenium_python": [
            "selenium webdriver python",
            "selenium pytest automation",
            "selenium page object model",
            "selenium explicit wait",
            "selenium action chains",
            "selenium screenshot",
            "selenium headless chrome",
            "selenium firefox geckodriver",
            "selenium select dropdown",
            "selenium handle alert",
            "selenium iframe switch",
            "selenium multiple windows",
            "selenium data driven testing",
            "selenium parallel execution",
        ],
        "selenium_java": [
            "selenium java testng",
            "selenium java junit",
            "selenium java page factory",
            "selenium java webdriverwait",
            "selenium java cucumber bdd",
        ],
        "selenium_javascript": [
            "selenium webdriver javascript",
            "selenium mocha chai",
            "webdriverio automation",
            "nightwatch selenium",
        ],
    }

    # File extensions to look for
    FILE_EXTENSIONS = {
        "vbscript": [".vbs", ".qfl", ".mts"],
        "python": [".py"],
        "java": [".java"],
        "javascript": [".js", ".ts"],
    }

    def __init__(self, token: str = None, output_dir: str = "./data/raw"):
        """
        Initialize the GitHub scraper.

        Args:
            token: GitHub personal access token (increases rate limits)
            output_dir: Directory to save scraped data
        """
        self.token = token or os.getenv("GITHUB_TOKEN")
        if self.token:
            from github import Auth
            self.github = Github(auth=Auth.Token(self.token))
        else:
            self.github = Github()
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Track collected repos to avoid duplicates
        self.collected_repos = set()
        self.samples: List[CodeSample] = []

        logger.info(f"GitHub Scraper initialized. Output: {self.output_dir}")

        # Check rate limit
        self._check_rate_limit()

    def _check_rate_limit(self):
        """Check and log GitHub API rate limit status."""
        try:
            rate_limit = self.github.get_rate_limit()
            # Handle both old and new API versions
            if hasattr(rate_limit, 'core'):
                core = rate_limit.core
                search = rate_limit.search
            else:
                core = rate_limit.rate
                search = rate_limit.rate

            logger.info(f"Rate Limit - Core: {core.remaining}/{core.limit}")

            if hasattr(search, 'remaining') and search.remaining < 5:
                reset_time = search.reset.timestamp() - time.time()
                logger.warning(f"Search rate limit low. Resets in {reset_time:.0f} seconds")
        except Exception as e:
            logger.warning(f"Could not check rate limit: {e}")

    def _wait_for_rate_limit(self):
        """Wait for rate limit to reset."""
        try:
            rate_limit = self.github.get_rate_limit()
            if hasattr(rate_limit, 'search'):
                reset_time = rate_limit.search.reset.timestamp() - time.time()
            else:
                reset_time = rate_limit.rate.reset.timestamp() - time.time()
        except:
            reset_time = 60  # Default wait

        if reset_time > 0:
            logger.info(f"Waiting {reset_time:.0f} seconds for rate limit reset...")
            time.sleep(reset_time + 5)

    def search_repositories(
        self,
        query: str,
        language: str = None,
        min_stars: int = 5,
        max_results: int = 100
    ) -> Generator[Dict[str, Any], None, None]:
        """
        Search for repositories matching the query.

        Args:
            query: Search query
            language: Filter by programming language
            min_stars: Minimum star count
            max_results: Maximum repositories to return
        """
        search_query = f"{query} stars:>={min_stars}"
        if language:
            search_query += f" language:{language}"

        logger.info(f"Searching: {search_query}")

        try:
            repos = self.github.search_repositories(
                query=search_query,
                sort="stars",
                order="desc"
            )

            count = 0
            for repo in repos:
                if count >= max_results:
                    break

                if repo.full_name in self.collected_repos:
                    continue

                self.collected_repos.add(repo.full_name)
                count += 1

                yield {
                    "name": repo.full_name,
                    "description": repo.description or "",
                    "url": repo.html_url,
                    "stars": repo.stargazers_count,
                    "language": repo.language,
                    "default_branch": repo.default_branch,
                }

        except RateLimitExceededException:
            logger.warning("Rate limit exceeded, waiting...")
            self._wait_for_rate_limit()

    def get_file_contents(
        self,
        repo_name: str,
        extensions: List[str]
    ) -> Generator[Dict[str, Any], None, None]:
        """
        Get contents of files with specified extensions from a repository.

        Args:
            repo_name: Full repository name (owner/repo)
            extensions: List of file extensions to collect
        """
        try:
            repo = self.github.get_repo(repo_name)

            # Get all files recursively
            contents = repo.get_contents("")

            while contents:
                file_content = contents.pop(0)

                if file_content.type == "dir":
                    try:
                        contents.extend(repo.get_contents(file_content.path))
                    except Exception as e:
                        logger.debug(f"Could not access {file_content.path}: {e}")
                        continue
                else:
                    # Check if file has target extension
                    if any(file_content.name.lower().endswith(ext) for ext in extensions):
                        try:
                            # Get file content
                            if file_content.size < 100000:  # Skip files > 100KB
                                content = base64.b64decode(file_content.content).decode('utf-8', errors='ignore')

                                yield {
                                    "path": file_content.path,
                                    "name": file_content.name,
                                    "content": content,
                                    "url": file_content.html_url,
                                    "size": file_content.size,
                                }
                        except Exception as e:
                            logger.debug(f"Could not decode {file_content.path}: {e}")

        except RateLimitExceededException:
            logger.warning("Rate limit exceeded, waiting...")
            self._wait_for_rate_limit()
        except Exception as e:
            logger.error(f"Error accessing repo {repo_name}: {e}")

    def collect_vbscript_samples(self, max_repos: int = 50):
        """Collect VBScript code samples for UFT/QTP automation."""
        logger.info("Collecting VBScript samples...")

        all_queries = self.SEARCH_QUERIES["vbscript_uft"] + self.SEARCH_QUERIES["vbscript_general"]

        for query in tqdm(all_queries, desc="VBScript queries"):
            for repo in self.search_repositories(query, max_results=max_repos // len(all_queries) + 1):
                for file_data in self.get_file_contents(repo["name"], self.FILE_EXTENSIONS["vbscript"]):
                    sample = CodeSample(
                        source="github",
                        repo_name=repo["name"],
                        file_path=file_data["path"],
                        language="vbscript",
                        code=file_data["content"],
                        description=repo["description"],
                        url=file_data["url"],
                        stars=repo["stars"],
                        collected_at=datetime.now().isoformat(),
                    )
                    self.samples.append(sample)

                # Respect rate limits
                time.sleep(0.5)

        logger.info(f"Collected {len([s for s in self.samples if s.language == 'vbscript'])} VBScript samples")

    def collect_selenium_samples(self, max_repos: int = 50):
        """Collect Selenium code samples in Python, Java, and JavaScript."""
        logger.info("Collecting Selenium samples...")

        language_mapping = {
            "selenium_python": ("python", self.FILE_EXTENSIONS["python"]),
            "selenium_java": ("java", self.FILE_EXTENSIONS["java"]),
            "selenium_javascript": ("javascript", self.FILE_EXTENSIONS["javascript"]),
        }

        for query_key, (lang, extensions) in language_mapping.items():
            queries = self.SEARCH_QUERIES[query_key]

            for query in tqdm(queries, desc=f"Selenium {lang}"):
                for repo in self.search_repositories(query, language=lang.capitalize() if lang != "javascript" else "JavaScript", max_results=max_repos // len(queries) + 1):
                    for file_data in self.get_file_contents(repo["name"], extensions):
                        # Filter for files that actually contain Selenium code
                        if "selenium" in file_data["content"].lower() or "webdriver" in file_data["content"].lower():
                            sample = CodeSample(
                                source="github",
                                repo_name=repo["name"],
                                file_path=file_data["path"],
                                language=f"selenium_{lang}",
                                code=file_data["content"],
                                description=repo["description"],
                                url=file_data["url"],
                                stars=repo["stars"],
                                collected_at=datetime.now().isoformat(),
                            )
                            self.samples.append(sample)

                    time.sleep(0.5)

        selenium_count = len([s for s in self.samples if s.language.startswith("selenium")])
        logger.info(f"Collected {selenium_count} Selenium samples")

    def save_samples(self, filename: str = None):
        """Save collected samples to JSONL file."""
        if not self.samples:
            logger.warning("No samples to save")
            return

        if filename is None:
            filename = f"github_samples_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jsonl"

        output_path = self.output_dir / filename

        with open(output_path, 'w', encoding='utf-8') as f:
            for sample in self.samples:
                f.write(json.dumps(asdict(sample), ensure_ascii=False) + '\n')

        logger.info(f"Saved {len(self.samples)} samples to {output_path}")

        # Summary by language
        by_lang = {}
        for s in self.samples:
            by_lang[s.language] = by_lang.get(s.language, 0) + 1

        logger.info(f"Samples by language: {by_lang}")

        return output_path

    def run_full_collection(self, max_repos_per_category: int = 30):
        """Run full data collection pipeline."""
        logger.info("Starting full GitHub collection...")

        self.collect_vbscript_samples(max_repos=max_repos_per_category)
        self.collect_selenium_samples(max_repos=max_repos_per_category)

        output_path = self.save_samples()

        logger.info(f"Collection complete! Total samples: {len(self.samples)}")
        return output_path


def main():
    """Main entry point for GitHub scraping."""
    import argparse

    parser = argparse.ArgumentParser(description="Scrape VBScript and Selenium code from GitHub")
    parser.add_argument("--token", help="GitHub personal access token")
    parser.add_argument("--output", default="./data/raw/github", help="Output directory")
    parser.add_argument("--max-repos", type=int, default=30, help="Max repos per category")
    args = parser.parse_args()

    scraper = GitHubScraper(token=args.token, output_dir=args.output)
    scraper.run_full_collection(max_repos_per_category=args.max_repos)


if __name__ == "__main__":
    main()
