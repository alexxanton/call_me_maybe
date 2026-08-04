import json
from typing import List, Dict
from pydantic import BaseModel
from llm_sdk.llm_sdk import Small_LLM_Model
from .validation import Function
from .base_prompt import BasePrompt


class FunctionCallingEngine(BaseModel):
    """
    Uses constrained decoding to generate a structured JSON output
    for a function calling system from a natural language prompt.
    """

    functions: List[Function]

    def __init__(self, functions: List[Function]) -> None:
        """Initialize the function calling engine."""
        super().__init__(functions=functions)
        self._model = Small_LLM_Model()
        self._vocab: Dict[str, int] = {}
        self._load_vocab()
        self._base_prompt = BasePrompt(functions)

    def _load_vocab(self) -> None:
        """Load vocabulary."""
        path = self._model.get_path_to_vocab_file()
        with open(path, "r", encoding="utf-8") as f:
            self._vocab = json.load(f)

    def generate(self, prompt: str) -> str:
        """Generate the output."""
        output = f'{{"prompt": "{prompt}",'
        self._model.encode(f"{self._base_prompt}\n{output}")
        return ""
