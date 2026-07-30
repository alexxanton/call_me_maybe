#import llm_sdk.llm_sdk
from typing import Dict, Any


class FunctionCallingEngine:
    """
    Uses constrained decoding to generate a structured JSON output
    for a function calling system from a natural language prompt.
    """

    def __init__(self, functions: Dict[Any, Any]) -> None:
        """Initialize engine."""
        self._functions = functions

    def generate(self, prompt: str) -> str:
        """Generate the output."""
        return ""
