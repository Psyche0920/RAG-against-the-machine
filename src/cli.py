"""Command-line interface for the RAG application."""

from pathlib import Path
from typing import Iterable

from pydantic import BaseModel, ValidationError
from tqdm import tqdm

from src.chunking import DEFAULT_MAX_CHUNK_SIZE
from src.dataset import load_dataset, save_dataset
from src.index import DEFAULT_INDEX_DIRECTORY, BM25Index
from src.models import (
    AnsweredQuestion,
    MinimalAnswer,
    MinimalSearchResults,
    MinimalSource,
    StudentSearchResults,
    StudentSearchResultsAndAnswer,
    UnansweredQuestion,
)

from src.documents import read_source_text
from src.generation import DEFAULT_MODEL_NAME, AnswerGenerator


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
        """Format a dataset-related exception for CLI output.

        Args:
            error: Exception raised while reading or validating a dataset.
            fallback_path: Path to display when the exception does not
                contain one.

        Returns:
            A concise error message suitable for display to the user.
        """
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
    def _validate_output_directory(directory_path: str | Path) -> None:
        """Reject an output-directory argument that is an existing file.

        Args:
            directory_path: Directory a CLI command intends to write under.

        Raises:
            NotADirectoryError: If ``directory_path`` already exists as a
                regular file instead of a directory.
        """
        directory = Path(directory_path)
        if directory.exists() and not directory.is_dir():
            raise NotADirectoryError(
                f"expected an output directory, got file: {directory}"
            )

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

        Raises:
            OSError: 如果输出目录或 JSON 文件无法创建或写入。
        """
        RagCLI._validate_output_directory(directory_path)
        output_directory = Path(directory_path)
        output_directory.mkdir(parents=True, exist_ok=True)
        output_path = output_directory / file_name
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
        # 2000 原本只是默认值，调用者仍能传入更大的数字。这里同时把它
        # 作为 subject 规定的硬上限，避免生成 moulinette 会拒绝的来源。
        if not 1 <= max_chunk_size <= DEFAULT_MAX_CHUNK_SIZE:
            return (
                "Error: max_chunk_size must be between 1 and "
                f"{DEFAULT_MAX_CHUNK_SIZE} characters."
            )

        try:
            # 在读取整个 corpus 之前先拒绝错误的输出路径，避免耗时建完
            # 索引后才因为 index_directory 实际是文件而崩溃。
            self._validate_output_directory(index_directory)
            bm25_index = BM25Index.build_index(raw_directory, max_chunk_size)
            BM25Index.save_index(bm25_index, index_directory)
        # find_document_paths/load_document 已经给出可直接展示的错误信息，
        # 不需要像 dataset 命令那样再做一次格式化。
        except (OSError, ValueError) as error:
            return f"Error: {error}"

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
            A StudentSearchResults JSON string containing one question and
            its ranked sources. ``retrieved_sources`` is empty when nothing
            matches.
        """
        if not query.strip():
            return "Error: query must not be empty."
        if k < 0:
            return "Error: k must not be negative."

        try:
            bm25_index = BM25Index.load_index(index_directory)
        except FileNotFoundError as error:
            return f"Error: {error}"

        chunks = bm25_index.search(query, k)
        # 单条 search 与 search_dataset 使用相同的 Pydantic JSON 结构；
        # 不写硬编码文件，只把一条结果序列化到 stdout。
        question = UnansweredQuestion(question=query)
        result = MinimalSearchResults(
            question_id=question.question_id,
            question=question.question,
            retrieved_sources=[
                MinimalSource(
                    file_path=chunk.file_path,
                    first_character_index=chunk.first_character_index,
                    last_character_index=chunk.last_character_index,
                )
                for chunk in chunks
            ],
        )
        return StudentSearchResults(
            search_results=[result],
            k=k,
        ).model_dump_json(indent=2)

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
        if k < 0:
            return "Error: k must not be negative."

        # save_directory 是“目录”参数；如果它已经是普通文件，应在加载
        # 索引和批量搜索之前返回受控错误，而不是让 mkdir 抛 traceback。
        try:
            self._validate_output_directory(save_directory)
        except OSError as error:
            return f"Error: {error}"

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

        try:
            output_path = self._save_json(
                StudentSearchResults(search_results=search_results, k=k),
                save_directory,
                Path(dataset_path).name,
            )
        except OSError as error:
            return (
                f"Error: could not write output under {save_directory}: "
                f"{error}"
            )
        return f"Saved student_search_results to {output_path}"

    def answer(
        self,
        query: str,
        k: int = 10,
        index_directory: str = DEFAULT_INDEX_DIRECTORY,
        model_name: str = DEFAULT_MODEL_NAME,
    ) -> str:
        """Answer a single query using retrieved context.

        Args:
            query: Natural language or code search question.
            k: Maximum number of sources to retrieve as context.
            index_directory: Directory containing the persisted index.
            model_name: Hugging Face model id to generate the answer with.

        Returns:
            A StudentSearchResultsAndAnswer JSON string containing one
            question, its ranked sources, and the generated answer, or an
            error message when generation cannot start.
        """
        # AnswerGenerator.generate() rejects a blank question with
        # ValueError. Check first so an empty/whitespace-only query fails
        # with a clear message instead of loading the model just to crash.
        if not query.strip():
            return "Error: query must not be empty."
        if k < 0:
            return "Error: k must not be negative."

        try:
            bm25_index = BM25Index.load_index(index_directory)
        except OSError as error:
            return f"Error: {error}"

        chunks = bm25_index.search(query, k)
        # Hugging Face 会在模型名称错误、网络不可用且权重未缓存、权重
        # 损坏或内存不足时抛 OSError/ValueError/RuntimeError。它们属于
        # 第三方模型边界的可预期失败，应转换成 CLI 错误而不是 traceback。
        try:
            generator = AnswerGenerator(model_name)
        except (OSError, RuntimeError, ValueError) as error:
            return f"Error: model '{model_name}' could not be loaded: {error}"

        try:
            answer_text = generator.generate(
                query,
                [chunk.text for chunk in chunks],
            )
        except (OSError, RuntimeError, ValueError) as error:
            return f"Error: answer generation failed: {error}"
        # 单条 answer 同样包装成规定的 JSON，并保留生成答案所依据的来源。
        question = UnansweredQuestion(question=query)
        result = MinimalAnswer(
            question_id=question.question_id,
            question=question.question,
            retrieved_sources=[
                MinimalSource(
                    file_path=chunk.file_path,
                    first_character_index=chunk.first_character_index,
                    last_character_index=chunk.last_character_index,
                )
                for chunk in chunks
            ],
            answer=answer_text,
        )
        return StudentSearchResultsAndAnswer(
            search_results=[result],
            k=k,
        ).model_dump_json(indent=2)

    def answer_dataset(
        self,
        student_search_results_path: str,
        save_directory: str,
        model_name: str = DEFAULT_MODEL_NAME,
    ) -> str:
        """Generate answers for every question in a search-results file.

        Args:
            student_search_results_path: Path to a StudentSearchResults
                JSON file, e.g. produced by 'search_dataset'.
            save_directory: Directory the StudentSearchResultsAndAnswer
                JSON file is written under.
            model_name: Hugging Face model id to generate answers with.

        Returns:
            A message confirming where the results were saved, or a
            message describing why the command could not run.
        """
        # 必须在加载 Qwen 之前检查输出参数；否则一个明显错误的目录会让
        # 用户白白等待模型加载，最后才在写文件时失败。
        try:
            self._validate_output_directory(save_directory)
        except OSError as error:
            return f"Error: {error}"

        try:
            json_content = Path(student_search_results_path).read_text(
                encoding="utf-8"
            )
            student_results = StudentSearchResults.model_validate_json(
                json_content
            )
        except (
            FileNotFoundError,
            IsADirectoryError,
            UnicodeDecodeError,
            ValidationError,
        ) as error:
            return self._format_dataset_error(
                error, student_search_results_path
            )
        except OSError as error:
            return f"Error: {error}"

        # 模型只加载一次，再重复回答所有问题；加载失败时不创建一个
        # 看似成功但实际没有答案的输出文件。
        try:
            generator = AnswerGenerator(model_name)
        except (OSError, RuntimeError, ValueError) as error:
            return f"Error: model '{model_name}' could not be loaded: {error}"

        # StudentSearchResults = 多个问题的检索结果。
        # MinimalSearchResults = 一个问题的检索结果。
        # MinimalSource = 该问题检索到的一个来源。
        progress: Iterable[MinimalSearchResults] = tqdm(
            student_results.search_results,
            desc="Answering",
            unit="question",
        )

        # result = 一个 MinimalSearchResults；source = 一个 MinimalSource。
        # 每个 source 恢复成 chunk 文本，供模型生成 answer。
        answers = []
        for result in progress:
            # 空问题是已知的退化输入，保留该题并写入明确说明。
            if not result.question.strip():
                answer_text = "Error: question must not be empty."
            else:
                try:
                    answer_text = generator.generate(
                        result.question,
                        [
                            read_source_text(source)
                            for source in result.retrieved_sources
                        ],
                    )
                except (OSError, RuntimeError, ValueError) as error:
                    return f"Error: answer generation failed: {error}"
            answers.append(
                MinimalAnswer(
                    question_id=result.question_id,
                    question=result.question,
                    retrieved_sources=result.retrieved_sources,
                    answer=answer_text,
                )
            )

        # 输出仍是整个数据集，每项由 MinimalSearchResults
        # 变成带 answer 字段的 MinimalAnswer。
        try:
            output_path = self._save_json(
                StudentSearchResultsAndAnswer(
                    search_results=answers, k=student_results.k
                ),
                save_directory,
                Path(student_search_results_path).name,
            )
        except OSError as error:
            return (
                f"Error: could not write output under {save_directory}: "
                f"{error}"
            )
        return f"Saved student_search_results_and_answer to {output_path}"

    def evaluate(
        self,
        student_search_results_path: str,
        dataset_path: str,
    ) -> str:
        """Report recall@k against a ground-truth dataset, for local use.

        This mirrors the moulinette's recall@k definition (same file_path,
        IoU >= 0.05) but is not the official score used for grading.

        Args:
            student_search_results_path: Path to a StudentSearchResults
                JSON file produced by ``search_dataset``.
            dataset_path: Path to the answered ground-truth dataset.

        Returns:
            Recall@1, recall@3, recall@5, and recall@10 as formatted text,
            or a message describing why the input could not be loaded.
        """
        try:
            student_results = StudentSearchResults.model_validate_json(
                Path(student_search_results_path).read_text(
                    encoding="utf-8"
                )
            )
            dataset = load_dataset(dataset_path)
        except (
            FileNotFoundError,
            IsADirectoryError,
            UnicodeDecodeError,
            ValidationError,
        ) as error:
            return self._format_dataset_error(
                error, student_search_results_path
            )

        # ground_truth 是数据集给出的标准来源，不是程序生成的答案。
        # retrieved_by_id 是 BM25 对每个问题实际找回的来源。
        ground_truth = {
            question.question_id: question.sources
            for question in dataset.rag_questions
            if isinstance(question, AnsweredQuestion)
        }
        retrieved_by_id = {
            result.question_id: result.retrieved_sources
            for result in student_results.search_results
        }

        lines = ["Evaluation Results", "=" * 40]
        # Recall@k：只看排名前 k 的检索结果，但分母仍是全部正确来源。
        for cutoff in (1, 3, 5, 10):
            recalls = []
            for question_id, sources in ground_truth.items():
                if not sources:
                    continue
                retrieved = retrieved_by_id.get(question_id, [])[:cutoff]
                found = sum(
                    1
                    for source in sources
                    if any(
                        _is_match(source, candidate)
                        for candidate in retrieved
                    )
                )
                recalls.append(found / len(sources))
            mean_recall = (
                sum(recalls) / len(recalls) if recalls else 0.0
            )
            lines.append(
                f"Recall@{cutoff}: {mean_recall:.3f} "
                f"({mean_recall * 100:.1f}%)"
            )

        return "\n".join(lines)


def _iou(a_start: int, a_end: int, b_start: int, b_end: int) -> float:
    """Compute the intersection-over-union of two character ranges.

    Args:
        a_start: Inclusive start offset of the first range.
        a_end: Exclusive end offset of the first range.
        b_start: Inclusive start offset of the second range.
        b_end: Exclusive end offset of the second range.

    Returns:
        The intersection length divided by the union length, or zero when
        the ranges do not overlap.
    """
    # IoU = 两个字符区间的重叠长度 / 合并后的总长度。
    intersection = max(0, min(a_end, b_end) - max(a_start, b_start))
    if intersection <= 0:
        return 0.0
    union = max(a_end, b_end) - min(a_start, b_start)
    return intersection / union if union > 0 else 0.0


def _is_match(
    source: MinimalSource,
    candidate: MinimalSource,
    iou_threshold: float = 0.05,
) -> bool:
    """Check whether a retrieved source matches a ground-truth source.

    Args:
        source: Expected source from the ground-truth dataset.
        candidate: Source returned by retrieval.
        iou_threshold: Minimum character-range IoU required for a match.

    Returns:
        True when both sources use the same path and their character ranges
        meet the IoU threshold; otherwise False.
    """
    # 官方命中条件是同一文件且 IoU >= 0.05；它只评估检索位置，
    # 不保证重叠文字包含完整答案，也不评估模型生成的 answer。
    if source.file_path != candidate.file_path:
        return False
    return (
        _iou(
            source.first_character_index,
            source.last_character_index,
            candidate.first_character_index,
            candidate.last_character_index,
        )
        >= iou_threshold
    )


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
