"""Pydantic data models for the RAG pipeline."""


import uuid
from typing import List

from pydantic import BaseModel, Field


class MinimalSource(BaseModel):
    """Location of a text fragment inside a source file.

    Offsets follow Python slice semantics (left-inclusive, right-exclusive):
    ``full_text[first_character_index:last_character_index]`` reconstructs
    the fragment exactly, the same way it does for ``Chunk``.
    """

    file_path: str
    first_character_index: int
    last_character_index: int


class UnansweredQuestion(BaseModel):
    """A question without an answer."""

    question_id: str = Field(
        default_factory=lambda: str(uuid.uuid4())
    )
    # default=str(uuid.uuid4()) default="ABC"
    # default_factory=lambda: str(uuid.uuid4())
    # default=<function generate at 0x...>
    question: str


class AnsweredQuestion(UnansweredQuestion):
    """A question with its sources and expected answer."""

    sources: List[MinimalSource]
    answer: str


class RagDataset(BaseModel):
    """"Represent a dataset containing RAG questions."""

    rag_questions: List[
        AnsweredQuestion | UnansweredQuestion
    ]


class MinimalSearchResults(BaseModel):
    """Retrieved sources for one question."""

    question_id: str
    question: str
    retrieved_sources: List[MinimalSource]


class MinimalAnswer(MinimalSearchResults):
    """Represent retrieved sources and a generated answer."""

    answer: str


class StudentSearchResults(BaseModel):
    """Represent search results produced for a full dataset."""

    search_results: List[MinimalSearchResults]
    k: int


class StudentSearchResultsAndAnswer(BaseModel):
    """Represent dataset search results with genreated answer."""

    search_results: List[MinimalAnswer]
    k: int


class Document(BaseModel):
    """Represent one text document loaded from the corpus."""

    file_path: str
    text: str


class Chunk(BaseModel):
    """A contiguous span of text extracted from one document.

    Contiguous means that the characters come from one unbroken section.
    `text` contains this chunk only, not the full document.
    Offsets are the chunk's start and end positions in the full document,
    following Python slice semantics (left-inclusive, right-exclusive):
    ``document.text[first_character_index:last_character_index] == text``.
    Chunk offsets must use the same rule as MinimalSource offsets.
    """

    file_path: str
    text: str
    first_character_index: int
    last_character_index: int
