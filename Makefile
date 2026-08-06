INPUT_DIR = data/input
all: function

install:
	uv sync

%: $(INPUT_DIR)/%_calling_tests.json $(INPUT_DIR)/%s_definition.json
	@uv run python -m src											\
		--input					$(INPUT_DIR)/$@_calling_tests.json	\
		--functions_definition	$(INPUT_DIR)/$@s_definition.json

run: install
	uv run python -m src

debug: install
	uv run python -m pdb -m src

clean:
	find . -name "__pycache__" -exec rm -r {} +
	rm -r data/output

lint: install
	-uv run flake8 src/
	uv run mypy src/ --warn-return-any --warn-unused-ignores --ignore-missing-imports --disallow-untyped-defs --check-untyped-defs

lint-strict: install
	-uv run flake8 src/
	uv run mypy src/ --strict
