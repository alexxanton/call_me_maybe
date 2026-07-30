#import llm_sdk.llm_sdk

class FunctionCallingEngine:
    """
    Uses constrained decoding to generate a structured JSON output
    for a function calling system from a natural language prompt.
    """
    def __init__(self, functions):
        """Initialize engine"""
        self._functions = functions
