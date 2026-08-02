"""Command-line interface for the RAG application."""

from pydantic import ValidationError

from src.dataset import load_dataset, save_dataset


class RagCLI:
    """Expose commands for operating the RAG pipeline."""

    def status(self) -> str:
        """Return the current application status.

        Returns:
            A message confirming that the CLI is available.
        """
        return "RAG against the machine is ready."

    def load(self, file_path: str) -> str:
        """Load and validate a RAG dataset.

        Args:
            file_path: Path to the JSON dataset file.

        Returns:
            A message describing the loaded dataset.
        """
        try:
            dataset = load_dataset(file_path)
        except FileNotFoundError:
            return f"Error: file not found: {file_path}"
        except IsADirectoryError:
            return f"Error: expected a file, got directory: {file_path}"
        except ValidationError as error:
            return f"Error: invalid RAG dataset:\n{error}"

        question_count = len(dataset.rag_questions)
        return (
            f"Dataset loaded successfully: "
            f"{question_count} questions."
        )

    def copy_dataset(
        self,
        input_path: str,
        output_path: str,
    ) -> str:
        """Load, validate, and save a copy of a dataset.

        Args:
            input_path: Path to the input JSON dataset.
            output_path: Path where the copy will be saved.

        Returns:
            A message confirming that the dataset was saved.

        """
        try:
            dataset = load_dataset(input_path)
            save_dataset(dataset, output_path)
        except FileNotFoundError:
            return f"Error: file not found: {input_path}"
        except IsADirectoryError:
            return f"Error: expected a file, got directory: {input_path}"
        except ValidationError as error:
            return f"Error: invalid RAG dataset:\n{error}"

        return f"Dataset saved successfully: {output_path}"
