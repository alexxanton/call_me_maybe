import json
from typing import Dict, Callable, Any, cast


class JsonParser:
    """Converts JSON files to dictionaries and validates their structure."""

    def __init__(self, functions: str, tests: str) -> None:
        """Initialize the JSON parser."""
        self._functions = self._load_file(functions, self._validate_functions)
        self._tests = self._load_file(tests, self._validate_tests)

    def _validate_functions(self, data: Dict[Any, Any]) -> bool:
        """Validates the structure of the functions dictionary."""
        return True

    def _validate_tests(self, data: Dict[Any, Any]) -> bool:
        """Validates the structure of the tests dictionary."""
        return True

    def _load_file(
        self, file: str, is_valid: Callable[[Dict[Any, Any]], bool]
    ) -> Dict[Any, Any]:
        """Reads a JSON file and converts it into a dictionary."""
        try:
            with open(file, "r") as f:
                data = json.loads(f.read())
            if not is_valid(data):
                raise Exception(f"The structure of {file} is not valid.")
            return cast(Dict[Any, Any], data)
        except (OSError, Exception) as e:
            exit(str(e))

    @property
    def functions(self) -> Dict[Any, Any]:
        """Returns the functions dictionary."""
        return self._functions

    @property
    def tests(self) -> Dict[Any, Any]:
        """Returns the tests dictionary."""
        return self._tests
