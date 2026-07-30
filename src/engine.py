#import llm_sdk.llm_sdk
from pydantic import BaseModel
from typing import List
from .validation import Function


class FunctionCallingEngine(BaseModel):
    """
    Uses constrained decoding to generate a structured JSON output
    for a function calling system from a natural language prompt.
    """

    functions: List[Function]

    def generate(self, prompt: str) -> str:
        """Generate the output."""
        return ""
