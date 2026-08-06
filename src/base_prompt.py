import json
from typing import List
from pydantic import BaseModel
from .validation import Function


class BasePrompt(BaseModel):
    """Represents the base prompt for the LLM and formats it appropiately."""

    functions: List[Function]
    _raw_base_prompt = """
        All generated output must be in valid JSON format.
        Take one function from the list and pick
        the most appropriate for the request.
        The parameters must be the same type as
        in the definition and must make sense,
        for example, a string can't be on a number parameter.
    """

    def __str__(self) -> str:
        """Defines how the class must be interpreted as a string."""
        return self._prompt

    def __init__(self, functions: List[Function]) -> None:
        """Initialize the base prompt."""
        super().__init__(functions=functions)
        self._prompt = self._format_base_prompt()

    def _format_base_prompt(self) -> str:
        """Format the base prompt with the functions."""
        funcs = [func.model_dump(mode="json") for func in self.functions]
        dump = json.dumps(funcs, indent=2)
        prompt = " ".join(self._raw_base_prompt.split())
        prompt += f"\nFunctions:\n{dump}"
        return prompt

    def _alt_formatting(self) -> str:
        """Unused due to worse LLM output."""
        funcs = "\n\n".join(
            "\n".join([
                f"Name: {func.name}",
                f"Description: {func.description}",
                f"Parameters:",
                *[
                    f"  {name}: {type.type}"
                    for name, type in func.parameters.items()
                ],
                f"Returns: {func.returns.type}"
            ])
            for func in self.functions
        )
        prompt = " ".join(self._raw_base_prompt.split())
        prompt += f"\nFunctions:\n{funcs}"
        return prompt
