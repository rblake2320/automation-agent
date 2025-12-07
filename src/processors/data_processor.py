"""
Data Processor
Transforms raw collected data into training format for the automation agent
"""

import os
import re
import json
import hashlib
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime

import polars as pl
from loguru import logger
from tqdm import tqdm


@dataclass
class TrainingExample:
    """A single training example in Alpaca format."""
    instruction: str
    input: str
    output: str


class DataProcessor:
    """
    Processes raw scraped data into training format.

    Transforms:
    - GitHub code samples -> instruction-output pairs
    - Stack Overflow Q&A -> instruction-output pairs
    - UFT scripts -> instruction-output pairs
    """

    # Templates for generating instructions
    INSTRUCTION_TEMPLATES = {
        "vbscript": [
            "Write VBScript code to {task}",
            "Create a UFT VBScript function that {task}",
            "Implement VBScript automation for {task}",
            "Generate UFT script to {task}",
            "Write QTP VBScript code that {task}",
        ],
        "selenium_python": [
            "Write Selenium Python code to {task}",
            "Create a Selenium WebDriver script that {task}",
            "Implement browser automation in Python to {task}",
            "Generate Selenium pytest code to {task}",
            "Write Python automation code that {task}",
        ],
        "selenium_java": [
            "Write Selenium Java code to {task}",
            "Create a Selenium WebDriver test in Java that {task}",
            "Implement browser automation in Java to {task}",
        ],
        "selenium_javascript": [
            "Write Selenium JavaScript code to {task}",
            "Create a WebDriverIO script that {task}",
            "Implement browser automation in Node.js to {task}",
        ],
    }

    # Common automation tasks to identify
    TASK_PATTERNS = {
        # VBScript/UFT patterns
        r'click.*button': "click a button",
        r'enter.*text|input.*value|set.*field': "enter text into a field",
        r'select.*dropdown|choose.*option': "select from a dropdown",
        r'verify.*element|check.*exist': "verify an element exists",
        r'wait.*element|wait.*load': "wait for an element",
        r'login|sign.?in': "handle user login",
        r'logout|sign.?out': "handle user logout",
        r'table.*row|iterate.*table': "work with table data",
        r'screenshot|capture': "take a screenshot",
        r'file.*upload|upload.*file': "upload a file",
        r'download|save.*file': "download a file",
        r'navigate|goto|open.*url': "navigate to a URL",
        r'frame|iframe|switch.*frame': "handle iframes",
        r'window|tab|switch.*window': "handle multiple windows",
        r'alert|popup|dialog': "handle alerts/dialogs",
        r'drag.*drop': "perform drag and drop",
        r'scroll': "scroll the page",
        r'hover|mouse.*over': "hover over an element",
        r'right.?click|context.*menu': "perform right-click",
        r'double.?click': "perform double-click",
        r'keyboard|key.*press|send.*keys': "simulate keyboard input",
        r'cookie': "handle cookies",
        r'excel|spreadsheet': "work with Excel data",
        r'database|sql|query': "interact with database",
        r'api|rest|http': "make API calls",
        r'json|parse': "handle JSON data",
        r'xml': "handle XML data",
        r'regex|pattern': "use regular expressions",
        r'error.*handling|try.*catch': "handle errors",
        r'log|report': "log or report results",
        r'parallel|concurrent': "run tests in parallel",
        r'data.*driven|parameterize': "implement data-driven testing",
    }

    def __init__(self, raw_data_dir: str = "./data/raw", output_dir: str = "./data/processed"):
        """
        Initialize the data processor.

        Args:
            raw_data_dir: Directory containing raw scraped data
            output_dir: Directory to save processed training data
        """
        self.raw_data_dir = Path(raw_data_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.examples: List[TrainingExample] = []
        self.seen_hashes = set()  # For deduplication

        logger.info(f"Data Processor initialized. Raw: {raw_data_dir}, Output: {output_dir}")

    def _compute_hash(self, text: str) -> str:
        """Compute hash for deduplication."""
        return hashlib.md5(text.encode()).hexdigest()

    def _is_duplicate(self, output: str) -> bool:
        """Check if output is a duplicate."""
        h = self._compute_hash(output.strip().lower())
        if h in self.seen_hashes:
            return True
        self.seen_hashes.add(h)
        return False

    def _detect_task(self, code: str, description: str = "") -> str:
        """Detect the task being performed by the code."""
        combined = f"{code} {description}".lower()

        for pattern, task in self.TASK_PATTERNS.items():
            if re.search(pattern, combined, re.IGNORECASE):
                return task

        # Default based on common keywords
        if "test" in combined:
            return "automate testing"
        elif "function" in combined:
            return "create a reusable function"
        else:
            return "automate a task"

    def _clean_code(self, code: str) -> str:
        """Clean and normalize code."""
        # Remove excessive blank lines
        code = re.sub(r'\n{3,}', '\n\n', code)

        # Remove trailing whitespace
        code = '\n'.join(line.rstrip() for line in code.split('\n'))

        # Remove common boilerplate comments
        code = re.sub(r'^\'.*copyright.*$', '', code, flags=re.MULTILINE | re.IGNORECASE)
        code = re.sub(r'^#.*copyright.*$', '', code, flags=re.MULTILINE | re.IGNORECASE)

        return code.strip()

    def _validate_code(self, code: str, language: str) -> bool:
        """Basic validation that code is meaningful."""
        # Minimum length
        if len(code) < 50:
            return False

        # Maximum length (avoid too long examples)
        if len(code) > 10000:
            return False

        # Must have some structure
        if language == "vbscript":
            # Should have some VBScript keywords
            keywords = ['function', 'sub', 'dim', 'set', 'if', 'for', 'while', 'browser', 'window']
            if not any(kw in code.lower() for kw in keywords):
                return False

        elif language.startswith("selenium"):
            # Should have Selenium imports or usage
            selenium_indicators = ['selenium', 'webdriver', 'driver', 'find_element', 'By.']
            if not any(ind in code for ind in selenium_indicators):
                return False

        return True

    def _generate_instruction(self, code: str, language: str, description: str = "") -> str:
        """Generate an instruction for the code."""
        import random

        task = self._detect_task(code, description)
        templates = self.INSTRUCTION_TEMPLATES.get(language, self.INSTRUCTION_TEMPLATES["vbscript"])

        template = random.choice(templates)
        return template.format(task=task)

    def process_github_samples(self):
        """Process GitHub code samples into training examples."""
        logger.info("Processing GitHub samples...")

        github_dir = self.raw_data_dir / "github"
        if not github_dir.exists():
            logger.warning(f"GitHub data directory not found: {github_dir}")
            return

        for jsonl_file in github_dir.glob("*.jsonl"):
            logger.info(f"Processing {jsonl_file.name}...")

            with open(jsonl_file, 'r', encoding='utf-8') as f:
                for line in tqdm(f, desc=jsonl_file.name):
                    try:
                        sample = json.loads(line)

                        code = self._clean_code(sample.get("code", ""))
                        language = sample.get("language", "vbscript")

                        if not self._validate_code(code, language):
                            continue

                        if self._is_duplicate(code):
                            continue

                        instruction = self._generate_instruction(
                            code,
                            language,
                            sample.get("description", "")
                        )

                        # Create training example
                        example = TrainingExample(
                            instruction=instruction,
                            input="",  # Could add context here
                            output=code,
                        )

                        self.examples.append(example)

                    except json.JSONDecodeError:
                        continue

        logger.info(f"Processed GitHub samples: {len(self.examples)} examples")

    def process_stackoverflow_qa(self):
        """Process Stack Overflow Q&A into training examples."""
        logger.info("Processing Stack Overflow Q&A...")

        so_dir = self.raw_data_dir / "stackoverflow"
        if not so_dir.exists():
            logger.warning(f"Stack Overflow data directory not found: {so_dir}")
            return

        initial_count = len(self.examples)

        for jsonl_file in so_dir.glob("*.jsonl"):
            logger.info(f"Processing {jsonl_file.name}...")

            with open(jsonl_file, 'r', encoding='utf-8') as f:
                for line in tqdm(f, desc=jsonl_file.name):
                    try:
                        qa = json.loads(line)

                        # Use question as instruction
                        title = qa.get("title", "")
                        question = qa.get("question", "")

                        # Use code blocks from answer as output
                        code_blocks = qa.get("code_blocks", [])
                        if not code_blocks:
                            continue

                        # Take the most substantial code block
                        best_code = max(code_blocks, key=len)
                        best_code = self._clean_code(best_code)

                        if len(best_code) < 50:
                            continue

                        if self._is_duplicate(best_code):
                            continue

                        # Determine language from tags
                        tags = qa.get("tags", [])
                        if any("vbscript" in t or "uft" in t or "qtp" in t for t in tags):
                            language = "vbscript"
                        elif any("python" in t for t in tags):
                            language = "selenium_python"
                        elif any("java" in t for t in tags):
                            language = "selenium_java"
                        else:
                            language = "selenium_python"  # Default

                        # Clean up instruction from title
                        instruction = title
                        if not instruction.endswith("?"):
                            instruction = f"How to {title.lower()}"

                        example = TrainingExample(
                            instruction=instruction,
                            input=question[:500] if question else "",  # Truncate long questions
                            output=best_code,
                        )

                        self.examples.append(example)

                    except json.JSONDecodeError:
                        continue

        new_count = len(self.examples) - initial_count
        logger.info(f"Processed Stack Overflow Q&A: {new_count} new examples")

    def process_uft_scripts(self):
        """Process UFT script samples into training examples."""
        logger.info("Processing UFT scripts...")

        uft_dir = self.raw_data_dir / "your_scripts"
        if not uft_dir.exists():
            logger.warning(f"UFT scripts directory not found: {uft_dir}")
            return

        initial_count = len(self.examples)

        for jsonl_file in uft_dir.glob("*training*.jsonl"):
            logger.info(f"Processing {jsonl_file.name}...")

            with open(jsonl_file, 'r', encoding='utf-8') as f:
                for line in tqdm(f, desc=jsonl_file.name):
                    try:
                        sample = json.loads(line)

                        instruction = sample.get("instruction", "")
                        input_text = sample.get("input", "")
                        output = sample.get("output", "")

                        if not instruction or not output:
                            continue

                        output = self._clean_code(output)

                        if len(output) < 50:
                            continue

                        if self._is_duplicate(output):
                            continue

                        example = TrainingExample(
                            instruction=instruction,
                            input=input_text,
                            output=output,
                        )

                        self.examples.append(example)

                    except json.JSONDecodeError:
                        continue

        new_count = len(self.examples) - initial_count
        logger.info(f"Processed UFT scripts: {new_count} new examples")

    def augment_examples(self):
        """Augment training examples with variations."""
        logger.info("Augmenting training examples...")

        augmented = []

        for example in tqdm(self.examples, desc="Augmenting"):
            augmented.append(example)

            # Add variations of instructions
            instruction_variations = [
                example.instruction,
                example.instruction.replace("Write", "Create"),
                example.instruction.replace("Write", "Generate"),
                example.instruction.replace("code to", "script that"),
            ]

            # Only add 1-2 variations to avoid too much duplication
            for var in instruction_variations[1:2]:
                if var != example.instruction:
                    augmented.append(TrainingExample(
                        instruction=var,
                        input=example.input,
                        output=example.output,
                    ))

        self.examples = augmented
        logger.info(f"Augmented to {len(self.examples)} examples")

    def split_train_val(self, val_ratio: float = 0.05) -> Tuple[List[TrainingExample], List[TrainingExample]]:
        """Split examples into training and validation sets."""
        import random

        random.shuffle(self.examples)

        val_size = int(len(self.examples) * val_ratio)
        val_examples = self.examples[:val_size]
        train_examples = self.examples[val_size:]

        logger.info(f"Split: {len(train_examples)} train, {len(val_examples)} validation")
        return train_examples, val_examples

    def save_training_data(self, filename: str = "training_data.jsonl"):
        """Save processed training data to JSONL file."""
        output_path = self.output_dir / filename

        with open(output_path, 'w', encoding='utf-8') as f:
            for example in self.examples:
                f.write(json.dumps(asdict(example), ensure_ascii=False) + '\n')

        logger.info(f"Saved {len(self.examples)} examples to {output_path}")

        # Also save train/val split
        train, val = self.split_train_val()

        train_path = self.output_dir / "train.jsonl"
        val_path = self.output_dir / "val.jsonl"

        with open(train_path, 'w', encoding='utf-8') as f:
            for example in train:
                f.write(json.dumps(asdict(example), ensure_ascii=False) + '\n')

        with open(val_path, 'w', encoding='utf-8') as f:
            for example in val:
                f.write(json.dumps(asdict(example), ensure_ascii=False) + '\n')

        logger.info(f"Saved train/val split to {self.output_dir}")

        return output_path

    def process_all(self):
        """Run the complete processing pipeline."""
        logger.info("Starting data processing pipeline...")

        self.process_github_samples()
        self.process_stackoverflow_qa()
        self.process_uft_scripts()

        logger.info(f"Total raw examples: {len(self.examples)}")

        # Augment
        self.augment_examples()

        # Save
        self.save_training_data()

        logger.info("Data processing complete!")

        # Print summary
        print("\n" + "="*50)
        print("PROCESSING SUMMARY")
        print("="*50)
        print(f"Total training examples: {len(self.examples)}")
        print(f"Output directory: {self.output_dir}")
        print(f"Files created:")
        print(f"  - training_data.jsonl (all examples)")
        print(f"  - train.jsonl (95% for training)")
        print(f"  - val.jsonl (5% for validation)")
        print("="*50)


def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(description="Process raw data into training format")
    parser.add_argument("--raw-dir", default="./data/raw", help="Raw data directory")
    parser.add_argument("--output-dir", default="./data/processed", help="Output directory")
    args = parser.parse_args()

    processor = DataProcessor(raw_data_dir=args.raw_dir, output_dir=args.output_dir)
    processor.process_all()


if __name__ == "__main__":
    main()
