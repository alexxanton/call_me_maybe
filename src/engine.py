import json
import numpy as np
from typing import List
from pydantic import BaseModel
from .validation import Function
from .base_prompt import BasePrompt
from .decoder import ConstrainedDecoder


class FunctionCallingEngine(BaseModel):
    """
    Uses constrained decoding to generate a structured JSON output
    for a function calling system from a natural language prompt.
    """

    functions: List[Function]

    def __init__(self, functions: List[Function]) -> None:
        """Initialize the function calling engine."""
        super().__init__(functions=functions)
        self._base_prompt = BasePrompt(functions)

        from llm_sdk.llm_sdk import Small_LLM_Model
        self._model = Small_LLM_Model()
        path = self._model.get_path_to_vocab_file()
        with open(path, "r", encoding="utf-8") as f:
            self._vocab = json.load(f)
        self._id_values = {v: k for k, v in self._vocab.items()}

    def _get_input_ids(self, output: str) -> List[int]:
        """Encode a string and return as a list of ints."""
        return self._model.encode(output)[0].tolist()

    def generate(self, prompt: str) -> str:
        """Generate the output."""
        decoder = ConstrainedDecoder(prompt, self.functions)
        input_ids = self._get_input_ids(f'{self._base_prompt}\n')
        output_start_idx = len(self._model.decode(input_ids))
        input_ids += self._get_input_ids(f"{decoder.state}")
        name_complete = False
        text_idx = output_start_idx
        param_complete = True

        while not decoder.finished:
            if not name_complete:
                allowed_tokens = decoder.get_name_tokens(
                    self._model.decode(input_ids), self._vocab, self._id_values
                )
            elif not decoder.func_name:
                decoder.retrieve_func_name(self._model.decode(input_ids))
                input_ids += self._get_input_ids(decoder.state)

            if name_complete and param_complete:
                input_ids += self._get_input_ids(decoder.inject_next_param())
                param_complete = False

            logits = self._model.get_logits_from_input_ids(input_ids)
            np_logits = np.array(logits)
            if allowed_tokens:
                mask = np.ones_like(np_logits, dtype=bool)
                for allowed in allowed_tokens:
                    if 0 <= allowed < len(mask):
                        mask[allowed] = False
                np_logits[mask] = -float("inf")
                allowed_tokens = None

            next_id = int(np.argmax(np_logits))
            if next_id == self._vocab.get('"'):
                name_complete = True
            input_ids.append(next_id)

            text = self._model.decode(input_ids)
            print(text[text_idx:], end="", flush=True)
            text_idx = len(text)

        return self._model.decode(input_ids)
