import json
import numpy as np
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
        prompt = f'{{\n  "prompt": "{prompt}",\n  "name": "'
        output = f'{self._base_prompt}\n{prompt}'
        print(prompt, end="")
        while not output.endswith("}\n}\n"):
            input_ids = self._model.encode(output)[0].tolist()
            logits = self._model.get_logits_from_input_ids(input_ids)
            logits_arr = np.array(logits)
            next_id = np.argmax(logits_arr)
            token = self._model.decode(next_id)
            output += token
            print(token, end="", flush=True)
        return output
