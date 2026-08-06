.PHONY: install run debug clean fclean lint lint-strict test

install:
	uv sync

run:
	uv run python -m src status


debug:
	uv run python -m pdb -m src status


clean:
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type d -name .mypy_cache -exec rm -rf {} +
	find . -type d -name .pytest_cache -exec rm -rf {} +
	find . -name "*.pyc" -delete
	find . -name "*.pyo" -delete

fclean:
	rm -rf .venv
	rm -rf data/processed/*
	rm -rf data/output/*

lint:
	uv run flake8 .
	uv run mypy . \
		--warn-return-any \
		--warn-unused-ignores \
		--ignore-missing-imports \
		--disallow-untyped-defs \
		--check-untyped-defs

lint-strict:
	uv run flake8 .
	uv run mypy . --strict
