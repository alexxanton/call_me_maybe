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
        exit(str(e))
    engine = FunctionCallingEngine(json_parser.functions)

    for test in json_parser.prompts:
        print(test)
        output = engine.generate(test.prompt)
        print()
        try:
            json_parser.write_to_output_file(output)
        except OSError as e:
            print(e)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nProgram stopped.")
