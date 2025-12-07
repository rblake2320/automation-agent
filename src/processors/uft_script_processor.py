"""
UFT Script Processor
Extracts and processes VBScript from UFT/QTP projects for training data
"""

import os
import re
import json
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, List, Any, Optional, Generator
from dataclasses import dataclass, asdict, field
from datetime import datetime
from zipfile import ZipFile, BadZipFile

from loguru import logger
from tqdm import tqdm


@dataclass
class UFTAction:
    """Represents a UFT action/test script."""
    name: str
    script_path: str
    code: str
    functions: List[Dict[str, str]]
    object_references: List[str]
    description: str = ""
    parameters: List[str] = field(default_factory=list)


@dataclass
class UFTProject:
    """Represents a UFT project with all its components."""
    name: str
    path: str
    actions: List[UFTAction]
    function_libraries: List[Dict[str, Any]]
    object_repository: Dict[str, Any]
    data_tables: List[str]
    recovery_scenarios: List[str]
    collected_at: str


@dataclass
class TrainingSample:
    """A training sample in instruction-output format."""
    instruction: str
    input: str
    output: str
    source: str
    category: str
    metadata: Dict[str, Any]


class UFTScriptProcessor:
    """
    Processes UFT/QTP projects to extract training data.

    Supports:
    - Action scripts (.mts files)
    - Function libraries (.qfl, .vbs)
    - Object repositories (.tsr, .bdb)
    - Test configurations
    """

    # Patterns for extracting VBScript components
    FUNCTION_PATTERN = re.compile(
        r'(?:Public\s+|Private\s+)?(?:Function|Sub)\s+(\w+)\s*\((.*?)\)(.*?)End\s+(?:Function|Sub)',
        re.IGNORECASE | re.DOTALL
    )

    OBJECT_PATTERN = re.compile(
        r'(Browser|Window|Dialog|Page|Frame|WebButton|WebEdit|WebList|WebTable|WebElement|WinButton|WinEdit|WinList|WinObject|SwfWindow|SwfButton|SwfEdit)\s*\(\s*"([^"]+)"\s*\)',
        re.IGNORECASE
    )

    CHECKPOINT_PATTERN = re.compile(
        r'\.Check\s*\(\s*Checkpoint\s*\(\s*"([^"]+)"\s*\)\s*\)',
        re.IGNORECASE
    )

    DATATABLE_PATTERN = re.compile(
        r'DataTable(?:\.Value)?\s*\(\s*"([^"]+)"(?:\s*,\s*(?:dtGlobalSheet|dtLocalSheet|"[^"]+"))?\s*\)',
        re.IGNORECASE
    )

    REPORTER_PATTERN = re.compile(
        r'Reporter\.ReportEvent\s+(\w+)\s*,\s*"([^"]+)"\s*,\s*"([^"]+)"',
        re.IGNORECASE
    )

    def __init__(self, output_dir: str = "./data/raw/your_scripts"):
        """
        Initialize UFT Script Processor.

        Args:
            output_dir: Directory to save processed data
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.projects: List[UFTProject] = []
        self.training_samples: List[TrainingSample] = []

        logger.info(f"UFT Script Processor initialized. Output: {self.output_dir}")

    def _read_file_content(self, file_path: Path) -> str:
        """Read file content with multiple encoding attempts."""
        encodings = ['utf-8', 'utf-16', 'cp1252', 'iso-8859-1']

        for encoding in encodings:
            try:
                return file_path.read_text(encoding=encoding)
            except (UnicodeDecodeError, UnicodeError):
                continue

        # Binary fallback
        try:
            return file_path.read_bytes().decode('utf-8', errors='ignore')
        except Exception as e:
            logger.warning(f"Could not read {file_path}: {e}")
            return ""

    def _extract_functions(self, code: str) -> List[Dict[str, str]]:
        """Extract function definitions from VBScript code."""
        functions = []

        for match in self.FUNCTION_PATTERN.finditer(code):
            func_name = match.group(1)
            params = match.group(2).strip()
            body = match.group(3).strip()

            # Get full function text
            full_match = match.group(0)

            functions.append({
                "name": func_name,
                "parameters": params,
                "body": body,
                "full_code": full_match,
            })

        return functions

    def _extract_object_references(self, code: str) -> List[str]:
        """Extract UFT object references from code."""
        objects = set()

        for match in self.OBJECT_PATTERN.finditer(code):
            obj_type = match.group(1)
            obj_name = match.group(2)
            objects.add(f"{obj_type}(\"{obj_name}\")")

        return list(objects)

    def _extract_data_table_refs(self, code: str) -> List[str]:
        """Extract DataTable column references."""
        refs = set()

        for match in self.DATATABLE_PATTERN.finditer(code):
            refs.add(match.group(1))

        return list(refs)

    def process_action_script(self, script_path: Path) -> Optional[UFTAction]:
        """
        Process a single UFT action script (.mts file).

        Args:
            script_path: Path to the .mts or Script.mts file
        """
        try:
            code = self._read_file_content(script_path)

            if not code.strip():
                return None

            # Extract components
            functions = self._extract_functions(code)
            objects = self._extract_object_references(code)

            # Get action name from path
            action_name = script_path.parent.name
            if action_name == "Action1" or action_name.startswith("Action"):
                pass  # Keep default action names

            return UFTAction(
                name=action_name,
                script_path=str(script_path),
                code=code,
                functions=functions,
                object_references=objects,
                parameters=self._extract_data_table_refs(code),
            )

        except Exception as e:
            logger.error(f"Error processing {script_path}: {e}")
            return None

    def process_function_library(self, lib_path: Path) -> Optional[Dict[str, Any]]:
        """
        Process a UFT function library (.qfl or .vbs file).

        Args:
            lib_path: Path to the function library file
        """
        try:
            code = self._read_file_content(lib_path)

            if not code.strip():
                return None

            functions = self._extract_functions(code)

            return {
                "name": lib_path.stem,
                "path": str(lib_path),
                "code": code,
                "functions": functions,
                "function_count": len(functions),
            }

        except Exception as e:
            logger.error(f"Error processing library {lib_path}: {e}")
            return None

    def process_object_repository(self, tsr_path: Path) -> Dict[str, Any]:
        """
        Process a UFT object repository (.tsr file).

        Note: .tsr files are binary, this extracts what we can.
        For full parsing, UFT API would be needed.
        """
        objects = {}

        try:
            # Try to read as XML (some .tsr versions)
            try:
                tree = ET.parse(tsr_path)
                root = tree.getroot()

                for obj in root.iter():
                    if 'Name' in obj.attrib:
                        objects[obj.attrib['Name']] = {
                            "type": obj.tag,
                            "properties": dict(obj.attrib),
                        }
            except ET.ParseError:
                # Binary format - extract strings
                content = tsr_path.read_bytes()

                # Extract readable strings (object names, properties)
                string_pattern = re.compile(rb'[\x20-\x7e]{4,}')
                strings = string_pattern.findall(content)

                for s in strings:
                    decoded = s.decode('ascii', errors='ignore')
                    if any(keyword in decoded.lower() for keyword in ['browser', 'window', 'page', 'button', 'edit', 'list']):
                        objects[decoded] = {"type": "extracted", "raw": True}

        except Exception as e:
            logger.warning(f"Could not fully parse object repository {tsr_path}: {e}")

        return objects

    def scan_uft_project(self, project_path: Path) -> Optional[UFTProject]:
        """
        Scan a UFT project directory and extract all components.

        Args:
            project_path: Path to UFT project folder
        """
        project_path = Path(project_path)

        if not project_path.is_dir():
            logger.warning(f"Not a directory: {project_path}")
            return None

        logger.info(f"Scanning UFT project: {project_path}")

        actions = []
        function_libraries = []
        object_repository = {}
        data_tables = []
        recovery_scenarios = []

        # Find all action scripts
        for script_file in project_path.rglob("Script.mts"):
            action = self.process_action_script(script_file)
            if action:
                actions.append(action)

        # Also check for .mts files directly
        for script_file in project_path.rglob("*.mts"):
            if script_file.name != "Script.mts":
                action = self.process_action_script(script_file)
                if action:
                    actions.append(action)

        # Find function libraries
        for lib_pattern in ["*.qfl", "*.vbs"]:
            for lib_file in project_path.rglob(lib_pattern):
                lib = self.process_function_library(lib_file)
                if lib:
                    function_libraries.append(lib)

        # Find object repositories
        for tsr_file in project_path.rglob("*.tsr"):
            objects = self.process_object_repository(tsr_file)
            object_repository.update(objects)

        # Find data tables (Excel files)
        for excel_file in project_path.rglob("*.xls*"):
            data_tables.append(str(excel_file))

        if not actions and not function_libraries:
            logger.warning(f"No scripts found in {project_path}")
            return None

        project = UFTProject(
            name=project_path.name,
            path=str(project_path),
            actions=actions,
            function_libraries=function_libraries,
            object_repository=object_repository,
            data_tables=data_tables,
            recovery_scenarios=recovery_scenarios,
            collected_at=datetime.now().isoformat(),
        )

        logger.info(f"Found {len(actions)} actions, {len(function_libraries)} libraries")
        return project

    def scan_directory(self, root_path: str) -> List[UFTProject]:
        """
        Recursively scan a directory for UFT projects.

        Args:
            root_path: Root directory to scan
        """
        root = Path(root_path)
        projects = []

        logger.info(f"Scanning for UFT projects in: {root}")

        # Look for Test.tsp files (UFT project markers)
        for tsp_file in root.rglob("*.tsp"):
            project = self.scan_uft_project(tsp_file.parent)
            if project:
                projects.append(project)
                self.projects.append(project)

        # Also look for standalone action folders
        for mts_file in root.rglob("Script.mts"):
            # Check if already processed
            project_path = mts_file.parent.parent
            if not any(p.path == str(project_path) for p in projects):
                project = self.scan_uft_project(project_path)
                if project:
                    projects.append(project)
                    self.projects.append(project)

        logger.info(f"Found {len(projects)} UFT projects")
        return projects

    def generate_training_samples(self) -> List[TrainingSample]:
        """
        Generate training samples from collected UFT projects.
        Creates instruction-output pairs for fine-tuning.
        """
        samples = []

        for project in self.projects:
            # Generate samples from actions
            for action in project.actions:
                # Full script sample
                if len(action.code) > 100:
                    samples.append(TrainingSample(
                        instruction=f"Write a UFT VBScript action that performs automation tasks",
                        input=f"Action name: {action.name}. Objects used: {', '.join(action.object_references[:5])}",
                        output=action.code,
                        source="uft_project",
                        category="full_action",
                        metadata={"project": project.name, "action": action.name},
                    ))

                # Individual function samples
                for func in action.functions:
                    if len(func["full_code"]) > 50:
                        # Infer purpose from function name
                        purpose = self._infer_function_purpose(func["name"])

                        samples.append(TrainingSample(
                            instruction=f"Write a VBScript function to {purpose}",
                            input=f"Function name: {func['name']}, Parameters: {func['parameters']}",
                            output=func["full_code"],
                            source="uft_project",
                            category="function",
                            metadata={"project": project.name, "function": func["name"]},
                        ))

            # Generate samples from function libraries
            for lib in project.function_libraries:
                for func in lib["functions"]:
                    if len(func["full_code"]) > 50:
                        purpose = self._infer_function_purpose(func["name"])

                        samples.append(TrainingSample(
                            instruction=f"Write a reusable VBScript function to {purpose}",
                            input=f"Function: {func['name']}({func['parameters']})",
                            output=func["full_code"],
                            source="uft_library",
                            category="library_function",
                            metadata={"library": lib["name"], "function": func["name"]},
                        ))

        self.training_samples = samples
        logger.info(f"Generated {len(samples)} training samples")
        return samples

    def _infer_function_purpose(self, func_name: str) -> str:
        """Infer the purpose of a function from its name."""
        name_lower = func_name.lower()

        # Common patterns
        patterns = {
            "login": "handle user login",
            "logout": "handle user logout",
            "click": "click an element",
            "enter": "enter data into a field",
            "input": "input data",
            "select": "select from a dropdown",
            "verify": "verify an element or value",
            "check": "check a condition",
            "validate": "validate data",
            "wait": "wait for an element or condition",
            "get": "get a value or element",
            "set": "set a value",
            "read": "read data",
            "write": "write data",
            "export": "export data",
            "import": "import data",
            "navigate": "navigate to a page",
            "open": "open an application or page",
            "close": "close an application or window",
            "search": "search for an item",
            "filter": "filter data",
            "sort": "sort data",
            "save": "save data",
            "delete": "delete an item",
            "update": "update data",
            "create": "create a new item",
            "table": "handle table operations",
            "row": "process table rows",
            "cell": "handle table cells",
            "error": "handle errors",
            "log": "log information",
            "report": "report results",
        }

        for pattern, purpose in patterns.items():
            if pattern in name_lower:
                return purpose

        # Default
        return f"perform {func_name} operation"

    def save_projects(self, filename: str = None) -> Path:
        """Save collected projects to JSON file."""
        if not self.projects:
            logger.warning("No projects to save")
            return None

        if filename is None:
            filename = f"uft_projects_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

        output_path = self.output_dir / filename

        # Convert to serializable format
        data = []
        for project in self.projects:
            project_dict = {
                "name": project.name,
                "path": project.path,
                "actions": [
                    {
                        "name": a.name,
                        "script_path": a.script_path,
                        "code": a.code,
                        "functions": a.functions,
                        "object_references": a.object_references,
                    }
                    for a in project.actions
                ],
                "function_libraries": project.function_libraries,
                "object_repository": project.object_repository,
                "data_tables": project.data_tables,
                "collected_at": project.collected_at,
            }
            data.append(project_dict)

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

        logger.info(f"Saved {len(self.projects)} projects to {output_path}")
        return output_path

    def save_training_samples(self, filename: str = None) -> Path:
        """Save training samples to JSONL file."""
        if not self.training_samples:
            self.generate_training_samples()

        if not self.training_samples:
            logger.warning("No training samples to save")
            return None

        if filename is None:
            filename = f"uft_training_samples_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jsonl"

        output_path = self.output_dir / filename

        with open(output_path, 'w', encoding='utf-8') as f:
            for sample in self.training_samples:
                f.write(json.dumps(asdict(sample), ensure_ascii=False) + '\n')

        logger.info(f"Saved {len(self.training_samples)} training samples to {output_path}")

        # Summary
        categories = {}
        for s in self.training_samples:
            categories[s.category] = categories.get(s.category, 0) + 1
        logger.info(f"Samples by category: {categories}")

        return output_path


def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(description="Process UFT/QTP projects for training data")
    parser.add_argument("--scan-dir", required=True, help="Directory to scan for UFT projects")
    parser.add_argument("--output", default="./data/raw/your_scripts", help="Output directory")
    args = parser.parse_args()

    processor = UFTScriptProcessor(output_dir=args.output)
    processor.scan_directory(args.scan_dir)
    processor.save_projects()
    processor.save_training_samples()


if __name__ == "__main__":
    main()
