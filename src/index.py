"""Build, save, load, and search a BM25 lexical index over Chunks."""

import pickle
import re
from pathlib import Path
from typing import Final, Iterable

from rank_bm25 import BM25Okapi
from tqdm import tqdm

from src.chunking import DEFAULT_MAX_CHUNK_SIZE, chunk_documents
from src.documents import load_documents
from src.models import Chunk


DEFAULT_INDEX_DIRECTORY: Final[str] = "data/processed"
_INDEX_FILE_NAME: Final[str] = "bm25_index.pkl"

# Runs of letters/digits/underscore become one token, e.g.
# "vllm-0.10.1" -> ["vllm", "0", "10", "1"]; punctuation and whitespace
# only separate tokens and never appear in the result.
_TOKEN_PATTERN: Final[re.Pattern[str]] = re.compile(r"[A-Za-z0-9_]+")

# Closed-class English function words that carry almost no retrieval
# signal but appear in nearly every chunk, diluting BM25's term overlap
# with content words like "default"/"value". Only closed-class words are
# removed here; anything that looks like an identifier or proper noun is
# kept.
_STOPWORDS: Final[frozenset[str]] = frozenset({
    "a", "an", "the",
    "is", "are", "was", "were", "be", "been", "being",
    "this", "that", "these", "those",
    "what", "which", "who", "whom",
    "in", "on", "at", "by", "for", "with", "about", "against", "between",
    "into", "through", "during", "before", "after", "above", "below",
    "to", "from", "up", "down", "of",
    "and", "or", "but", "if", "then", "else", "as",
    "it", "its", "do", "does", "did", "doing",
    "have", "has", "had", "having",
    "i", "you", "he", "she", "we", "they", "them", "their", "his", "her",
    "my", "your", "our",
    "not", "no", "nor", "so", "than", "too", "very",
    "can", "will", "just", "should", "now",
})


def tokenize(text: str) -> list[str]:
    """Turn raw text into the lowercase token list BM25 operates on.

    Args:
        text: Raw text to tokenize.

    Returns:
        Lowercase alphanumeric/underscore tokens, in original order,
        with _STOPWORDS removed.
    """
    lowercase_text = text.lower()
    all_tokens = _TOKEN_PATTERN.findall(lowercase_text)
    return [token for token in all_tokens if token not in _STOPWORDS]


class BM25Index:
    """A BM25 lexical index built over a Chunk corpus.

    This is a service class providing indexing/search behaviour, not a
    pydantic data model: `Chunk` and friends are data exchanged between
    stages and need pydantic validation, while `BM25Index` wraps a
    third-party `BM25Okapi` object and search methods, not a JSON format
    exchanged with anything external. The subject explicitly allows
    indexer/retriever/pipeline service objects to be plain classes.
    """

    def __init__(self, chunks: list[Chunk], bm25: BM25Okapi) -> None:
        """Store the Chunk corpus and the BM25 model built over it.

        Args:
            chunks: Chunk list, in the same order used to build `bm25`.
            bm25: BM25Okapi model built from the tokenized chunks.
        """
        # scores[i] from BM25Okapi always corresponds to chunks[i], so
        # this order must not change after the index is built.
        self.chunks = chunks
        self.bm25 = bm25

    def search(self, query: str, k: int) -> list[Chunk]:
        """Return the top-k most relevant Chunks for a query.

        Args:
            query: Natural language or code search question.
            k: Maximum number of Chunks to return.

        Returns:
            Chunks ranked by BM25 score, descending, at most `k` of
            them. Empty if the corpus is empty, the query has no usable
            tokens, or `k` is not positive.
        """
        if k <= 0 or not self.chunks:
            return []

        # Query and corpus must share the same tokenize() rule, since
        # BM25 compares tokens, not raw string similarity.
        query_tokens = tokenize(query)
        if not query_tokens:
            return []

        # A token absent from bm25.idf never appeared in the indexed
        # corpus. If none of the query tokens exist in the vocabulary,
        # every Chunk would receive the same zero score and sorting
        # would return arbitrary corpus entries. Treat that case as no
        # match instead.
        if not any(token in self.bm25.idf for token in query_tokens):
            return []

        scores = self.bm25.get_scores(query_tokens)
        ranked_indices = sorted(
            range(len(self.chunks)),
            key=lambda index: scores[index],
            reverse=True,
        )
        return [self.chunks[index] for index in ranked_indices[:k]]

    @staticmethod
    def build_index(
        raw_directory: str | Path,
        max_chunk_size: int = DEFAULT_MAX_CHUNK_SIZE,
        show_progress: bool = True,
    ) -> "BM25Index":
        """Load, chunk, and index every supported file in a corpus.

        Args:
            raw_directory: Root directory of the source corpus, e.g.
                'data/raw'.
            max_chunk_size: Maximum number of characters per chunk.
            show_progress: Whether to display loading/chunking progress
                bars.

        Returns:
            A fully built BM25Index, ready to search or save.

        Raises:
            FileNotFoundError: 'raw_directory' does not exist.
            NotADirectoryError: 'raw_directory' is not a directory.
            ValueError: 'max_chunk_size' is out of range, or the corpus
                has no indexable content.
        """

        if not 1 <= max_chunk_size <= DEFAULT_MAX_CHUNK_SIZE:
            raise ValueError(
                "max_chunk_size must be between 1 and "
                f"{DEFAULT_MAX_CHUNK_SIZE} characters."
            )

        documents = load_documents(
            raw_directory,
            show_progress=show_progress,
        )
        chunks = chunk_documents(
            documents,
            max_chunk_size,
            show_progress=show_progress,
        )

        # rank_bm25 divides by the average document length internally,
        # so an empty corpus would raise ZeroDivisionError deep inside
        # the third-party library. Fail with a clear message before that
        # boundary instead; this also covers a corpus directory that
        # only has empty or unsupported files.
        if not chunks:
            raise ValueError(
                f"No indexable chunks found in corpus: {raw_directory}"
            )

        progress: Iterable[Chunk] = tqdm(
            chunks,
            desc="Tokenizing",
            unit="chunk",
            disable=not show_progress,
        )
        tokenized_corpus = [
            tokenize(chunk.file_path) + tokenize(chunk.text)
            for chunk in progress
        ]

        bm25 = BM25Okapi(tokenized_corpus, k1=1.0, b=0.4)

        return BM25Index(chunks=chunks, bm25=bm25)

    @staticmethod
    def save_index(
        index: "BM25Index",
        directory_path: str | Path = DEFAULT_INDEX_DIRECTORY,
    ) -> Path:
        """Persist an index to disk so it can be loaded without rebuilding.

        Args:
            index: BM25Index to save.
            directory_path: Directory the index file is written under.

        Returns:
            Path to the written index file.
        """
        directory = Path(directory_path)

        if directory.exists() and not directory.is_dir():
            raise NotADirectoryError(
                f"expected an output directory, got file: {directory}"
            )

        directory.mkdir(parents=True, exist_ok=True)
        index_path = directory / _INDEX_FILE_NAME

        with index_path.open("wb") as index_file:
            pickle.dump(index, index_file)

        return index_path

    @staticmethod
    def load_index(
        directory_path: str | Path = DEFAULT_INDEX_DIRECTORY,
    ) -> "BM25Index":
        """Load a previously saved BM25 index from disk.

        Args:
            directory_path: Directory containing the saved index file.

        Returns:
            The BM25Index restored from pickle data.

        Raises:
            FileNotFoundError: No index file exists at the given
                directory, or the index file could not be restored (e.g.
                it is stale or corrupted).
        """
        index_path = Path(directory_path) / _INDEX_FILE_NAME
        if not index_path.exists():
            raise FileNotFoundError(
                f"No index found at {index_path}; run the index command first."
            )

        try:
            with index_path.open("rb") as index_file:
                index: BM25Index = pickle.load(index_file)
        except (
            pickle.UnpicklingError,
            ModuleNotFoundError,
            AttributeError,
            EOFError,
        ) as error:
            raise FileNotFoundError(
                f"Index at {index_path} could not be loaded (it may be "
                f"stale or corrupted: {error}); run the index command "
                f"again to rebuild it."
            ) from error

        return index
