"""Application entry point for the RAG command-line interface."""

import fire

from src.cli import RagCLI


def main() -> None:
    """Start the Python Fire command-line interface."""
    fire.Fire(RagCLI)


if __name__ == "__main__":
    main()
