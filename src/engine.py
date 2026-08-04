import json
from typing import List, Dict
from pydantic import BaseModel
from llm_sdk.llm_sdk import Small_LLM_Model
from .validation import Function


class FunctionCallingEngine(BaseModel):
    """
    Uses constrained decoding to generate a structured JSON output
    for a function calling system from a natural language prompt.
    """

    functions: List[Function]
    _raw_base_prompt = """
        All generated output must be in valid JSON format.
        Take one function from the list and pick
        the most appropiate for the request.
        The parameters must be the same type as
        in the definition and must make sense,
        for example, a string can't be on a number parameter.
    """

    def __init__(self, functions: List[Function]) -> None:
        """Initialize the function calling engine."""
        super().__init__(functions=functions)
        self._model = Small_LLM_Model()
        self._vocab: Dict[str, int] = {}
        self._load_vocab()
        self._base_prompt = self._format_base_prompt()

    def _format_base_prompt(self) -> str:
        """Format the base prompt with the functions."""
        funcs = [func.model_dump(mode="json") for func in self.functions]
        dump = json.dumps(funcs, indent=2)
        prompt = " ".join(self._raw_base_prompt.split())
        prompt += f"\nFunctions:\n{dump}"
        print(prompt)
        return prompt

    def _load_vocab(self) -> None:
        """Load vocabulary."""
        path = self._model.get_path_to_vocab_file()
        with open(path, "r", encoding="utf-8") as f:
            self._vocab = json.load(f)

    def generate(self, prompt: str) -> str:
        """Generate the output."""
        output = f'{{"prompt": "{prompt}",'
        return ""
