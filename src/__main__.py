import argparse
from pydantic import ValidationError
from src.engine import FunctionCallingEngine
from src.json_parser import JsonParser


def main() -> None:
    """Parse the args and execute the function calling tests."""
    parser = argparse.ArgumentParser(description="LLM Function Caller")
    parser.add_argument(
        "--functions_definition",
        default="data/input/functions_definition.json"
    )
    parser.add_argument(
        "--input",
        default="data/input/function_calling_tests.json"
    )
    parser.add_argument(
        "--output",
        default="data/output/function_calling_results.json"
    )

    args = parser.parse_args()
    try:
        json_parser = JsonParser.load_files(
            args.functions_definition, args.input
        )
    except (ValidationError, RuntimeError) as e:
        exit(str(e))
    engine = FunctionCallingEngine(functions=json_parser.functions)

    for test in json_parser.prompts:
        engine.generate(test.prompt)


if __name__ == "__main__":
    main()
