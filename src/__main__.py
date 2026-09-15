import sys
import argparse
from dotenv import load_dotenv
from pydantic import ValidationError
from src.json_parser import JsonParser
from src.engine import FunctionCallingEngine


def main() -> None:
    """Parse the args and execute the function calling tests."""
    load_dotenv()
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
        json_parser = JsonParser(
            args.functions_definition, args.input, args.output
        )
    except (ValidationError, RuntimeError) as e:
        sys.exit(str(e))

    try:
        engine = FunctionCallingEngine(json_parser.functions)
    except Exception as e:
        sys.exit(f"Error while initializating the engine: {e}")

    for test in json_parser.prompts:
        output = engine.generate(test.prompt)
        json_parser.append_output(output)
        print("\n")

    try:
        json_parser.write_to_output_file()
    except OSError as e:
        print(e)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nProgram stopped.")
