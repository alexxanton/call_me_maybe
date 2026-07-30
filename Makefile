all: run

install:
	uv sync

run: install
	uv run python -m src

debug: install
	uv run python -m pdb -m src

clean:
	find . -name "__pycache__" -exec rm -r {} +

lint: install
	-uv run flake8 src/
	uv run mypy src/ --warn-return-any --warn-unused-ignores --ignore-missing-imports --disallow-untyped-defs --check-untyped-defs
