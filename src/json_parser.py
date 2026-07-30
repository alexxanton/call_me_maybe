import json
from jsonschema import validate, ValidationError
from typing import Dict, Any, cast


class JsonParser:
    """Converts JSON files to dictionaries and validates their structure."""

    _functions_schema = {
        "type": "array",
        "items": {
            "type": "object",
            "required": ["name", "description", "parameters", "returns"],
            "properties": {
                "name": {"type": "string"},
                "description": {"type": "string"},
                "parameters": {
                    "type": "object",
                    "additionalProperties": {
                        "required": ["type"],
                        "properties": {
                            "type": {
                                "type": "string",
                                "enum": ["number", "string"]
                            }
                        }
                    }
                },
                "returns": {
                    "type": "object",
                    "required": ["type"],
                    "properties": {
                        "type": {
                            "type": "string",
                            "enum": ["number", "string"]
                        }
                    }
                }
            }
        }
    }

    _prompts_schema = {
        "type": "array",
        "items": {
            "type": "object",
            "required": ["prompt"],
            "properties": {"prompt": {"type": "string"}}
        }
    }

    def __init__(self, functions: str, tests: str) -> None:
        """Initialize the JSON parser."""
        self._functions = self._load_file(functions, self._functions_schema)
        self._tests = self._load_file(tests, self._prompts_schema)

    def _load_file(self, file: str, schema: Dict[Any, Any]) -> Dict[Any, Any]:
        """Reads a JSON file and converts it into a dictionary."""
        try:
            with open(file, "r") as f:
                data = json.loads(f.read())
            validate(data, schema)
            return cast(Dict[Any, Any], data)
        except OSError as e:
            exit(str(e))
        except ValidationError as e:
            print("Error found from the following object:")
            print(json.dumps(e.instance, indent=2))
            print(f"Expected type: {e.validator_value!r}")
            if e.validator_value == "array":
                print("JSON file must contain one list holding all objects")
            exit()

    @property
    def functions(self) -> Dict[Any, Any]:
        """Returns the functions dictionary."""
        return self._functions

    @property
    def tests(self) -> Dict[Any, Any]:
        """Returns the tests dictionary."""
        return self._tests
