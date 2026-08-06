"""Load and save RAG datasets as validated JSON files."""

from pathlib import Path

from src.models import RagDataset


def load_dataset(file_path: str | Path) -> RagDataset:
    """Load and validate a RAG dataset from a JSON file.

    Args:
        file_path: Path to the JSON dataset file.

    Returns:
        The validated RAG dataset.

    Raises:
        FileNotFoundError: If the dataset file does not exist.
        IsADirectoryError: If the supplied path points to a directory.
        UnicodeDecodeError: If the file is not valid UTF-8 text.
        ValidationError: If the JSON does not match RagDataset.
    """
    path = Path(file_path)
    json_content = path.read_text(encoding="utf-8")
    return RagDataset.model_validate_json(json_content)


def save_dataset(
    dataset: RagDataset,
    file_path: str | Path,
) -> None:
    """Save a RAG dataset to a formatted JSON file.

    Args:
        dataset: The validated RAG dataset to save.
        file_path: Destination path for the JSON file.
    """
    path = Path(file_path)
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    json_content = dataset.model_dump_json(indent=2)
    path.write_text(
        json_content + "\n",
        encoding="utf-8"
    )
