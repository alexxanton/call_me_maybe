import json
import argparse
#from src.utils import load_inputs
from src.engine import FunctionCallingEngine

def main():
    parser = argparse.ArgumentParser(description="Call Me Maybe: LLM Function Caller")
    parser.add_argument("--functions_definition", default="data/input/functions_definition.json")
    parser.add_argument("--input", default="data/input/function_calling_tests.json")
    parser.add_argument("--output", default="data/output/function_calling_results.json")
    args = parser.parse_args()

    with open(args.functions_definition, "r") as f:
        funcs = json.loads(f.read())
    with open(args.input, "r") as f:
        tests = json.loads(f.read())
    engine = FunctionCallingEngine(funcs)

    for test in tests:
        print(test["prompt"])
        #engine.generate(test)

if __name__ == "__main__":
    main()
