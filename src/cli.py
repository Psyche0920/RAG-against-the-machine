"""Command-line interface for the RAG application."""


class RagCLI:
    """Expose commands for operating the RAG pipeline."""

    def status(self) -> str:
        """Return the current application status.

        Returns:
            A message confirming that the CLI is available.
        """
        return "RAG against the machine is ready."
