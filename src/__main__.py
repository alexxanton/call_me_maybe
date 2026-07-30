import argparse
from src.engine import FunctionCallingEngine
from src.json_parser import JsonParser


def main() -> None:
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
    json_parser = JsonParser(args.functions_definition, args.input)
    engine = FunctionCallingEngine(json_parser.functions)

    for test in json_parser.tests:
        engine.generate(test["prompt"])


if __name__ == "__main__":
    main()
