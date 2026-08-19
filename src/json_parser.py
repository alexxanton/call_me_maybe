from __future__ import annotations
import json
from pathlib import Path
from pydantic import BaseModel
from json.decoder import JSONDecodeError
from typing import List, Dict, Optional, TypeVar
from .validation import Function, Prompt


T = TypeVar("T", bound=BaseModel)


class JsonParser(BaseModel):
    """Loads and validates function and prompt definitions."""

    functions: List[Function]
    prompts: List[Prompt]
    output_file: str

    def __init__(
        self, funcs_file: str, prompts_file: str, output_file: str
    ) -> None:
        """Loads function and prompt definitions from JSON files."""
        functions = self._load_file(funcs_file, Function)
        prompts = self._load_file(prompts_file, Prompt)
        super().__init__(
            functions=functions, prompts=prompts, output_file=output_file
        )
        self._path: Optional[Path] = None
        self._outputs = []

    def _load_file(self, file: str, model: type[T]) -> List[T]:
        """Reads a JSON file and validates it."""
        try:
            with open(file, "r") as f:
                data = json.load(f)
            return [model.model_validate(item) for item in data]
        except (OSError, JSONDecodeError) as e:
            raise RuntimeError(f"Failed to load {file}: {e}") from e

    def write_to_output_file(self) -> None:
        """Write the generated output from the LLM to the output file."""
        if self._path is None:
            self._path = Path(self.output_file)
            self._path.parent.mkdir(parents=True, exist_ok=True)
        self._path.write_text(json.dumps(self._outputs, indent=2))

    def append_output(self, output: str) -> None:
        """Adds a function object to the outputs list."""
        try:
            self._outputs.append(json.loads(output))
        except JSONDecodeError:
            print("Invalid JSON")
