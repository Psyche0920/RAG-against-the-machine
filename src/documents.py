"""Functions for loading source documents from a corpus."""

from pathlib import Path
from typing import Final, Iterable

from tqdm import tqdm

from src.models.models import Document, MinimalSource


SUPPORTED_SUFFIXES: Final[frozenset[str]] = frozenset(
    {
        ".py",
        ".pyi",
        ".md",
        ".rst",
        ".txt",
    }
)


def is_supported_file(file_path: Path) -> bool:
    """Check whether a file can be loaded into the text corpus.

    Args:
        file_path: Path to the candidate file.

    Returns:
        True when the file has a supported text-file extension.
    """

    return (
        file_path.is_file()
        and file_path.suffix.lower() in SUPPORTED_SUFFIXES
    )


def make_project_relative_path(file_path: Path) -> str:
    """Convert a file path to a project-root-relative POSIX path.

    The Subject requires source paths to match the indexed corpus paths
    exactly, for example:
    data/raw/vllm-0.10.1/docs/features/lora.md

    Args:
        file_path: Path to a file inside the current project.

    Returns:
        A project-relative path using forward slashes.

    Raises:
        ValueError: If the file is outside the current project directory.
    """
    project_root = Path.cwd().resolve()
    # /home/wehan/RAG/Github
    resolved_path = file_path.resolve()
    # /home/wehan/RAG/Github/src/main.py

    relative_path = resolved_path.relative_to(project_root)
    # Path("src/main.py")

    return relative_path.as_posix()
    # "src/main.py"


def load_document(file_path: str | Path) -> Document:
    """Load one supported UTF-8 text document.

    This is the single-file loader used by load_documents().

    Args:
        file_path: Path to the document.

    Returns:
        A Document containing the exact project-relative path and text.

    Raises:
        FileNotFoundError: If the path does not exist.
        IsADirectoryError: If the path points to a directory.
        ValueError: If the file type is unsupported or outside the project.
        UnicodeDecodeError: If the file is not valid UTF-8 text
    """
    path = Path(file_path)

    if not path.exists():
        raise FileNotFoundError(
            f"Document does not exist: {path}"
        )

    if not path.is_file():
        raise IsADirectoryError(
            f"Expected a file, got: {path}"
        )

    if path.suffix.lower() not in SUPPORTED_SUFFIXES:
        raise ValueError(
            f"Unsupported document type: {path.suffix or '<no suffix>'}"
        )

    text = path.read_text(encoding="utf-8")
    relative_path = make_project_relative_path(path)

    return Document(
        file_path=relative_path,
        text=text,
    )


def find_document_paths(
        directory_path: str | Path,
) -> list[Path]:
    """Find supported files recursively in a corpus directory.

    Args:
        directory_path: Root directory of the source corpus.

    Returns:
        A sorted list of supported file paths.

    Raises:
        FileNotFoundError: If the directory does not exist.
        NotADirectoryError: If the path is not a directory.
    """
    directory = Path(directory_path)

    if not directory.exists():
        raise FileNotFoundError(
            f"Corpus directory does not exist: {directory}"
        )

    if not directory.is_dir():
        raise NotADirectoryError(
            f"Expected a corpus directory, got: {directory}"
        )

    # rglob("*") recursively visits entries in this directory and its
    # subdirectories.
    document_paths = [
        path
        for path in directory.rglob("*")
        if is_supported_file(path)
    ]
    return sorted(document_paths)


def load_documents(
        directory_path: str | Path,
        show_progress: bool = True,
) -> list[Document]:
    """Load all supported documents from a corpus directory.

    Args:
        directory_path: Root directory of the source corpus.
        show_progress: Whether to display a tqdm progress bar.

    Returns:
        Documents sorted by their file paths.

    Raises:
        FileNotFoundError: If the directory does not exist.
        NotADirectoryError: If the path is not a directory.
        ValueError: If a discovered file is outside the project.
        UnicodeDecodeError: If a file cannot be decoded as UTF-8.
    """
    document_paths = find_document_paths(directory_path)

    # Iterable[Path] says progress can yield Path objects; it is not a list.
    # tqdm creates the wrapper here, but advances only when it is iterated
    # below.
    progress: Iterable[Path] = tqdm(
        document_paths,
        desc="Loading documents",
        unit="file",
        disable=not show_progress,
    )

    # Load files one at a time; the progress bar advances during this
    # iteration.
    documents = [
        load_document(path)
        for path in progress
    ]

    # Return a list of Document(file_path=..., text=...) objects.
    return documents


def read_source_text(source: MinimalSource) -> str:
    """Read the text a MinimalSource points to from the on-disk corpus.

    Args:
        source: A retrieved source location (file_path + character range).


    Returns:
        The textbeween first_character_index and last_character_index.
        or an empty string if the file cannot be read.
    """
    try:
        text = Path(source.file_path).read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return ""
    start = source.first_character_index
    end = source.last_character_index

    if start < 0 or end < start or end > len(text):
        return ""

    return text[source.first_character_index:source.last_character_index]