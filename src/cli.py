"""Command-line interface for the RAG application."""

from pathlib import Path
from typing import Iterable

from pydantic import BaseModel, ValidationError
from tqdm import tqdm

from src.chunking import DEFAULT_MAX_CHUNK_SIZE
from src.dataset import load_dataset, save_dataset
from src.index import DEFAULT_INDEX_DIRECTORY, BM25Index
from src.models.models import (
    AnsweredQuestion,
    MinimalSearchResults,
    MinimalSource,
    StudentSearchResults,
    UnansweredQuestion,
)


class RagCLI:
    """Expose commands for operating the RAG pipeline."""

    @staticmethod
    def _format_dataset_error(
        error: FileNotFoundError
        | IsADirectoryError
        | UnicodeDecodeError
        | ValidationError,
        fallback_path: str,
    ) -> str:
        """将数据集相关异常转换为统一的 CLI 错误消息。"""
        # 当前调用中 error_path 通常等于 fallback_path；仍优先保留
        # 文件系统异常自带的 filename，异常无路径时才使用 fallback。
        error_path = getattr(error, "filename", None) or fallback_path

        if isinstance(error, FileNotFoundError):
            return f"Error: file not found: {error_path}"
        if isinstance(error, IsADirectoryError):
            return f"Error: expected a file, got directory: {error_path}"
        if isinstance(error, UnicodeDecodeError):
            return f"Error: dataset is not valid UTF-8: {error_path}"
        return f"Error: invalid RAG dataset:\n{error}"

    @staticmethod
    def _save_json(
        model: BaseModel, directory_path: str, file_name: str
    ) -> Path:
        """把 pydantic 模型写入 ``<directory_path>/<file_name>``。

        Args:
            model: 需要序列化的 pydantic 模型。
            directory_path: 输出目录；不存在时会自动创建。
            file_name: 输出文件名。

        Returns:
            实际写入的文件路径。
        """
        output_path = Path(directory_path) / file_name
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            model.model_dump_json(indent=2) + "\n",
            encoding="utf-8",
        )
        return output_path

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
        # 所有数据集命令使用同一组异常类型和同一种消息格式。
        except (
            FileNotFoundError,
            IsADirectoryError,
            UnicodeDecodeError,
            ValidationError,
        ) as error:
            return self._format_dataset_error(error, file_path)

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
        # save_dataset 出错时，异常中的 filename 能正确指向输出路径。
        except (
            FileNotFoundError,
            IsADirectoryError,
            UnicodeDecodeError,
            ValidationError,
        ) as error:
            return self._format_dataset_error(error, input_path)

        return f"Dataset saved successfully: {output_path}"

    def index(
        self,
        max_chunk_size: int = DEFAULT_MAX_CHUNK_SIZE,
        raw_directory: str = "data/raw",
        index_directory: str = DEFAULT_INDEX_DIRECTORY,
    ) -> str:
        """Ingest the corpus and persist a BM25 index.

        Args:
            max_chunk_size: Maximum number of characters per chunk.
            raw_directory: Root directory of the source corpus.
            index_directory: Directory the index is persisted under.

        Returns:
            A message reporting how many chunks were indexed.
        """
        try:
            bm25_index = BM25Index.build_index(raw_directory, max_chunk_size)
        # find_document_paths/load_document 已经给出可直接展示的错误信息，
        # 不需要像 dataset 命令那样再做一次格式化。
        except (
            FileNotFoundError,
            NotADirectoryError,
            ValueError,
            UnicodeDecodeError,
        ) as error:
            return f"Error: {error}"

        BM25Index.save_index(bm25_index, index_directory)
        return (
            f"Ingestion complete! Indexed {len(bm25_index.chunks)} chunks "
            f"under {index_directory}/"
        )

    def search(
        self,
        query: str,
        k: int = 10,
        index_directory: str = DEFAULT_INDEX_DIRECTORY,
    ) -> str:
        """Return the top-k sources for a single query.

        Args:
            query: Natural language or code search question.
            k: Maximum number of sources to return.
            index_directory: Directory containing the persisted index.

        Returns:
            One "file_path [first_character_index:last_character_index]"
            line per retrieved chunk, or a message when nothing matched.
        """
        try:
            bm25_index = BM25Index.load_index(index_directory)
        except FileNotFoundError as error:
            return f"Error: {error}"

        chunks = bm25_index.search(query, k)
        if not chunks:
            return "No results found."

        return "\n".join(
            f"{chunk.file_path} "
            f"[{chunk.first_character_index}:{chunk.last_character_index}]"
            for chunk in chunks
        )

    def search_dataset(
        self,
        dataset_path: str,
        save_directory: str,
        k: int = 10,
        index_directory: str = DEFAULT_INDEX_DIRECTORY,
    ) -> str:
        """Search every question in a dataset and save the results.

        Args:
            dataset_path: Path to the JSON dataset of questions.
            save_directory: Directory the StudentSearchResults JSON file is
                written under; named after 'dataset_path'.
            k: Maximum number of sources to return per question.
            index_directory: Directory containing the persisted index.

        Returns:
            A message confirming where the results were saved, or a
            message describing why the command could not run.
        """
        try:
            bm25_index = BM25Index.load_index(index_directory)
        except FileNotFoundError as error:
            return f"Error: {error}"

        try:
            dataset = load_dataset(dataset_path)
        except (
            FileNotFoundError,
            IsADirectoryError,
            UnicodeDecodeError,
            ValidationError,
        ) as error:
            return self._format_dataset_error(error, dataset_path)

        progress: Iterable[AnsweredQuestion | UnansweredQuestion] = tqdm(
            dataset.rag_questions,
            # desc is a display-only action label; unit matches one loop item.
            desc="Searching",
            unit="question",
        )

        search_results = [
            MinimalSearchResults(
                question_id=question.question_id,
                question=question.question,
                retrieved_sources=[
                    MinimalSource(
                        file_path=chunk.file_path,
                        first_character_index=chunk.first_character_index,
                        last_character_index=chunk.last_character_index,
                    )
                    for chunk in bm25_index.search(question.question, k)
                ],
            )
            for question in progress
        ]

        output_path = self._save_json(
            StudentSearchResults(search_results=search_results, k=k),
            save_directory,
            Path(dataset_path).name,
        )
        return f"Saved student_search_results to {output_path}"


# =============================================================================
# CLI 命令速记
# =============================================================================
#
# - index：读取原始文件 -> 分块 -> 建立 BM25 -> 保存 pickle 索引。
# - search：加载已有索引 -> 搜索单个 query -> 返回 top-k Chunk 的来源范围。
# - search_dataset：索引只加载一次，逐题搜索，把 Chunk 转成 MinimalSource，
#   最后写出 moulinette 要求的 StudentSearchResults JSON。
# - desc="Searching" / desc="Tokenizing" 只是 tqdm 进度条的动作标签，
#   不参与检索、计分或答案生成。
# - unit 与每次迭代的对象一致：question / file / chunk，也只影响显示。
# - copy_dataset 保留当前名称；它在保存副本前会先加载并验证数据集。
# - 当前调用中 error_path 通常等于 fallback_path，同时兼容异常自带的 filename。
# - _format_dataset_error 统一数据集异常消息；_save_json 统一创建输出目录、
#   Pydantic JSON 序列化和 UTF-8 写入。这两项改善一致性，不影响 Recall。
