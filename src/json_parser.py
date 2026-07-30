from __future__ import annotations
import json
from json.decoder import JSONDecodeError
from typing import List, TypeVar
from pydantic import BaseModel
from .validation import Function, Prompt


T = TypeVar("T", bound=BaseModel)


class JsonParser(BaseModel):
    """Loads and validates function and prompt definitions."""

    functions: List[Function]
    prompts: List[Prompt]

    @classmethod
    def load_files(cls, funcs_file: str, prompts_file: str) -> JsonParser:
        """Loads function and prompt definitions from JSON files."""
        functions = cls._load_file(funcs_file, Function)
        prompts = cls._load_file(prompts_file, Prompt)

        return cls(
            functions=functions,
            prompts=prompts
        )

    @staticmethod
    def _load_file(file: str, model: type[T]) -> List[T]:
        """Reads a JSON file and validates it."""
        try:
            with open(file, "r") as f:
                data = json.load(f)
            return [model.model_validate(item) for item in data]
        except (OSError, JSONDecodeError) as e:
            raise RuntimeError(f"Failed to load {file}: {e}") from e
