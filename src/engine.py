import json
import numpy as np
from time import sleep
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
        self._text_idx = 0
        with open(path, "r", encoding="utf-8") as f:
            self._vocab = json.load(f)

    def _typewrite(self, input_ids: List[int]) -> None:
        """Print the output with a typewriter effect."""
        text = self._model.decode(input_ids[self._text_idx:])
        for c in text:
            print(c, end="", flush=True)
            sleep(.005)
        self._text_idx = len(input_ids)

    def _get_input_ids(self, output: str) -> List[int]:
        """Encode a string and return as a list of ints."""
        return self._model.encode(output)[0].tolist()

    def _print_top(self, np_logits) -> None:
        top_ids = np.argsort(np_logits)[-10:][::-1]

        for token_id in top_ids:
            print(
                token_id,
                repr(self._model.decode(int(token_id))),
                np_logits[token_id],
            )

    def generate(self, prompt: str) -> str:
        """Generate the output."""
        decoder = ConstrainedDecoder(prompt, self.functions, self._vocab)
        input_ids = self._get_input_ids(f'{self._base_prompt}\n')
        output_start_idx = len(input_ids)
        input_ids += self._get_input_ids(decoder.state)
        self._text_idx = output_start_idx

        while not decoder.finished:
            if not decoder.name_complete:
                allowed_tokens = decoder.get_name_tokens(
                    self._model.decode(input_ids)
                )
            elif not decoder.func_name:
                decoder.retrieve_func_name(self._model.decode(input_ids))
                input_ids += self._get_input_ids(decoder.state)

            if decoder.name_complete and decoder.param_complete:
                input_ids += self._get_input_ids(decoder.inject_next_param())
                decoder.param_complete = False

            logits = self._model.get_logits_from_input_ids(input_ids)
            np_logits = np.array(logits)

            if allowed_tokens:
                mask = np.ones_like(np_logits, dtype=bool)
                for allowed in allowed_tokens:
                    if 0 <= allowed < len(mask):
                        mask[allowed] = False

                np_logits[mask] = -float("inf")
                allowed_tokens = set()

            next_id = int(np.argmax(np_logits))
            if next_id == self._vocab.get('"'):
                decoder.name_complete = True
            if "\n" in self._model.decode([next_id]):
                decoder.param_complete = True
                decoder.close_param()
            input_ids.append(next_id)

            self._typewrite(input_ids)
            #self._typewrite(f"({self._model.decode(next_id)})"+text[text_idx:])
        input_ids += self._get_input_ids(decoder.state)
        self._typewrite(input_ids)

        return self._model.decode(input_ids[output_start_idx:])
