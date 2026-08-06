"""Chunking strategies that split loaded documents into indexable spans."""

import ast
import re
from pathlib import Path
from typing import Final, Iterable, Iterator

from tqdm import tqdm

from src.models import Chunk, Document

DEFAULT_MAX_CHUNK_SIZE: Final[int] = 2000

_CODE_SUFFIXES: Final[frozenset[str]] = frozenset({".py", ".pyi"})

# .rst (reStructuredText) is not Markdown, but it is still paragraph-
# organized plain text, so it shares the generic text strategy.
_TEXT_SUFFIXES: Final[frozenset[str]] = frozenset({".md", ".rst", ".txt"})

# Two or more consecutive newlines mark a paragraph break (a blank line);
# a single newline is just a line wrap within the same paragraph.
_PARAGRAPH_BREAK: Final[re.Pattern[str]] = re.compile(r"\n{2,}")


def _split_fixed_size(
        text: str,
        file_path: str,
        max_chunk_size: int,
        start_offset: int = 0,
) -> list[Chunk]:
    """Split text into fixed-size, non-overlapping character windows.

    Used both as the fallback for a file that cannot be parsed and to
    break up a single unit (function, paragraph, ...)
    that is still longer than 'max_chunk_size' on its own.

    Args:
        text: Text to split.
        file_path: Project-relative path of the source document.
        max_chunk_size: Maximum number of characters per chunk.
        start_offset: Character offset of 'text' within the full
        document, used to report absolute positions.

    Returns:
        Chunks covering 'text' end to end, each at most
        'max_chunk_size' characters wide.
    """

    chunks = []
    for start in range(0, len(text), max_chunk_size):
        window = text[start:start + max_chunk_size]
        if not window:
            continue
        chunks.append(
            Chunk(
                file_path=file_path,
                text=window,
                first_character_index=start_offset + start,
                # Left-inclusive/right-exclusive slicing, like Python's
                # own slices: last_character_index is the window's right
                # boundary, no -1 adjustment needed.
                last_character_index=start_offset + start + len(window),
            )
        )
    return chunks


def _chunk_span(
        text: str,
        file_path: str,
        start: int,
        end: int,
        max_chunk_size: int,
) -> list[Chunk]:
    """Wrap 'text[start:end]' in one Chunk, splitting it of oversized.

    Args:
        text: Full document text the span was taken from.
        file_path: Project-relative path of the source document.
        start: Character offset where the span begins.
        end: Character offset where the span ends.
        max_chunk_size: Maximum number of characters per chunk.

    Returns:
        A single chunk for the span, or several fixed-size chunks if the
        span is longer than 'max_chunk_size'. An empty list if the span
        is blank.
    """
    # first_character_index/last_character_index reuse start/end as-is:
    # both follow Python's left-inclusive/right-exclusive slice rule, so
    # full_text[first:last] == span with no off-by-one adjustment. This
    # must match the ground-truth dataset's convention (and the
    # moulinette's IoU comparison), which uses the same coordinate system.
    span = text[start:end]
    if not span.strip():
        return []

    if len(span) < max_chunk_size:
        return [
            Chunk(
                file_path=file_path,
                text=span,
                first_character_index=start,
                last_character_index=end,
            )
        ]
    return _split_fixed_size(span, file_path, max_chunk_size, start)


def _line_start_offsets(text: str) -> list[int]:
    """Compute each line's starting character offset within 'text'.

    Bridges AST node positions (line numbers) and character slicing
    (character offsets): `ast` reports where a node starts/ends using
    `node.lineno`/`node.end_lineno`, but `text[start:end]` needs character
    positions, not line numbers. This builds that line-to-offset lookup
    table, e.g. `offsets[node.end_lineno]` gives the character position
    right after that line ends.

    Args:
        text: Full document text, not a single line or chunk.

    Returns:
        `offsets[i]` is the character offset where line `i + 1` starts
        (offsets are 0-indexed, `ast` line numbers are 1-indexed). The
        list ends with one extra entry equal to `len(text)`, the offset
        just past the last character.
    """
    offsets = [0]
    for line in text.splitlines(keepends=True):
        offsets.append(offsets[-1] + len(line))
    return offsets


def chunk_python_code(
        document: Document,
        max_chunk_size: int,
) -> list[Chunk]:
    """Chunk a Python source file along its top-level statement bounds.

    Each top-level function, class, or statement becomes its own chunk
    where possible, so a retrieved snippet usually does not span two
    unrelated definitions. A single node still longer than
    'max_chunk_size' (e.g. a large function) is further split into
    fixed-size windows, and a file that fails to parse falls back to
    fixed-size splitting entirely rather than raising.

    Args:
        document: Python source document to chunk.
        max_chunk_size: Maximum number of characters per chunk.

    Returns:
        Chunks covering the document, ordered by position in the file.
    """
    text = document.text

    try:
        module = ast.parse(text)
    except SyntaxError:
        return _split_fixed_size(text, document.file_path, max_chunk_size)

    # module.body only holds top-level statements (no blank lines or
    # comments), so an empty or comment-only file needs the fallback too.
    if not module.body:
        return _split_fixed_size(text, document.file_path, max_chunk_size)

    line_starts = _line_start_offsets(text)
    chunks: list[Chunk] = []
    cursor = 0  # Character offset of the first not-yet-processed position.

    for node in module.body:
        # end_lineno (not lineno, which is only the start line) is
        # required: using lineno would cut off multi-line functions or
        # classes before their body ends. Missing end_lineno means the
        # AST boundary can't be trusted, so fall back entirely rather
        # than guess.
        if node.end_lineno is None:
            return _split_fixed_size(
                text,
                document.file_path,
                max_chunk_size,
            )
        end = line_starts[node.end_lineno]

        # A single oversized node can still produce multiple chunks;
        # extend (not append) keeps the result a flat list.
        chunks.extend(
            _chunk_span(
                text, document.file_path, cursor, end, max_chunk_size,
            )
        )
        cursor = end

    # ast creates no node for trailing comments/blank lines after the
    # last statement, so that tail still needs its own span.
    chunks.extend(
        _chunk_span(
            text, document.file_path, cursor, len(text), max_chunk_size,
        )
    )
    return chunks


def _iter_paragraphs(text: str) -> Iterator[tuple[int, int]]:
    """Yield each paragraph's (start, end) character boundary in 'text'.

    Args:
        text: Full document text.

    Yields:
        (start, end) pairs usable directly as `text[start:end]`.
    """
    cursor = 0
    for match in _PARAGRAPH_BREAK.finditer(text):
        if match.start() > cursor:
            yield cursor, match.start()
        cursor = match.end()
    if cursor < len(text):
        yield cursor, len(text)


def chunk_markdown_text(
        document: Document,
        max_chunk_size: int,
) -> list[Chunk]:
    """Split a Markdown or plain-text document along paragraph boundaries.

    Paragraphs are packed into a buffer greedily: as long as "buffer +
    next paragraph" stays within 'max_chunk_size', they are merged into
    one chunk, so short adjacent paragraphs can share a chunk. Once
    adding a paragraph would exceed the limit, the buffer is flushed and
    a new one starts with that paragraph.

    An oversized single paragraph is split on its own via
    `_split_fixed_size`; its short trailing remainder is not merged back
    into the next paragraph's buffer, keeping the splitting logic simple
    at the cost of an occasional very short tail chunk.

    Args:
        document: Markdown or plain-text document to chunk.
        max_chunk_size: Maximum number of characters per chunk.

    Returns:
        Chunks in original document order.
    """
    text = document.text
    chunks: list[Chunk] = []

    # buffer_start/buffer_end store only the current buffer's character
    # boundary in `text`, not a text copy.
    buffer_start = 0
    buffer_end = 0
    has_buffer = False

    for para_start, para_end in _iter_paragraphs(text):
        if not has_buffer:
            buffer_start = para_start
            buffer_end = para_end
            has_buffer = True
        elif para_end - buffer_start <= max_chunk_size:
            buffer_end = para_end
        else:
            chunks.extend(
                _chunk_span(
                    text, document.file_path, buffer_start, buffer_end,
                    max_chunk_size,
                )
            )
            buffer_start = para_start
            buffer_end = para_end

    # The loop only flushes the buffer when a *next* paragraph triggers
    # the else-branch; the final buffer needs an explicit flush here.
    if has_buffer:
        chunks.extend(
            _chunk_span(
                text, document.file_path, buffer_start, buffer_end,
                max_chunk_size,
            )
        )

    return chunks


def is_code_path(file_path: str | Path) -> bool:
    """Check whether a path should use the Python code chunking strategy.

    Exported so `chunk_document` and `index.py` (which groups chunks into
    code/text for separate BM25 models) share one suffix whitelist
    instead of maintaining two copies that could drift apart.

    Args:
        file_path: Document path, usually Document.file_path or
            Chunk.file_path.

    Returns:
        True for a known Python suffix; False otherwise (the caller must
        still check for Markdown/text separately).
    """
    return Path(file_path).suffix.lower() in _CODE_SUFFIXES


def chunk_document(
        document: Document,
        max_chunk_size: int = DEFAULT_MAX_CHUNK_SIZE,
) -> list[Chunk]:
    """Dispatch a document to its chunking strategy based on file suffix.

    The subject requires two chunking strategies (Python code and
    Markdown/text) but does not mandate AST-based or blank-line-based
    splitting specifically; those are this project's implementation
    choices.

    Args:
        document: Document to chunk.
        max_chunk_size: Maximum number of characters per chunk.

    Returns:
        Chunks produced by the strategy appropriate for the document.

    Raises:
        ValueError: The file suffix does not match any supported
            chunking strategy.
    """
    if is_code_path(document.file_path):
        return chunk_python_code(document, max_chunk_size)
    if Path(document.file_path).suffix.lower() in _TEXT_SUFFIXES:
        return chunk_markdown_text(document, max_chunk_size)
    suffix = Path(document.file_path).suffix.lower()
    raise ValueError(
        f"Unsupported document type for chunking: {suffix or '<no suffix>'}"
    )


def chunk_documents(
        documents: list[Document],
        max_chunk_size: int = DEFAULT_MAX_CHUNK_SIZE,
        show_progress: bool = True,
) -> list[Chunk]:
    """Chunk every document in a loaded corpus.

    Args:
        documents: Documents to chunk, e.g. from 'load_documents'.
        max_chunk_size: Maximum number of characters per chunk.
        show_progress: Whether to display a tqdm progress bar.

    Returns:
        All chunks from all documents, grouped by source document in
        input order.
    """
    progress: Iterable[Document] = tqdm(
        documents,
        desc="Chunking",
        unit="file",
        disable=not show_progress,
    )
    chunks: list[Chunk] = []
    for document in progress:
        chunks.extend(chunk_document(document, max_chunk_size))
    return chunks
