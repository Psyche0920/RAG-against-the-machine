"""Fast smoke tests for core pipeline invariants.

Not part of the graded submission (see the subject's "Additional
Guidelines": test programs are for your own verification, not submitted
or graded). Kept small and dependency-light so `make test` runs in
seconds and never needs the corpus or a downloaded model.
"""

from pathlib import Path
from typing import NoReturn

import pytest
from rank_bm25 import BM25Okapi

from src.chunking import chunk_document
from src.cli import RagCLI, _is_match, _iou
from src.documents import read_source_text
from src.index import BM25Index, tokenize
from src.models import (
    Chunk,
    Document,
    MinimalSource,
    StudentSearchResults,
)


def _save_tiny_index(directory: Path) -> None:
    """Save a one-chunk BM25 index for CLI tests."""
    chunk = Chunk(
        file_path="sample.py",
        text="prefix caching",
        first_character_index=0,
        last_character_index=14,
    )
    BM25Index.save_index(
        BM25Index(
            chunks=[chunk],
            bm25=BM25Okapi([tokenize(chunk.text)]),
        ),
        directory,
    )


# 测试 tokenize()：一句英文问题分词后，虚词（what/is/the/of）应该被去掉，
# 真正有信息量的词（default/value/max_size）应该保留，且都变成小写。
def test_tokenize_lowercases_and_drops_stopwords() -> None:
    tokens = tokenize("What is the default value of MAX_SIZE?")
    assert "what" not in tokens
    assert "is" not in tokens
    assert "the" not in tokens
    assert "of" not in tokens
    assert "default" in tokens
    assert "value" in tokens
    assert "max_size" in tokens


# 测试 tokenize() 的边界情况：空字符串、纯空格，都不应该报错，
# 应该老老实实返回空列表。
def test_tokenize_empty_or_blank_returns_no_tokens() -> None:
    assert tokenize("") == []
    assert tokenize("   ") == []


# 测试完全不在索引词表中的 query：所有 BM25 分数原本都会是 0，
# 不能因此返回语料开头的任意 Chunk，而应该明确表示没有匹配结果。
def test_search_unknown_tokens_returns_no_chunks() -> None:
    chunk = Chunk(
        file_path="sample.py",
        text="prefix caching",
        first_character_index=0,
        last_character_index=14,
    )
    bm25 = BM25Okapi([tokenize(chunk.text)])
    index = BM25Index(chunks=[chunk], bm25=bm25)
    assert index.search("zzzz_nonexistent_token_987654", k=5) == []


# 测试单条 search 也使用 subject 规定的 StudentSearchResults JSON，
# 而不是只返回供人阅读、无法被程序可靠解析的文本行。
def test_single_search_returns_structured_json(tmp_path: Path) -> None:
    _save_tiny_index(tmp_path)

    output = RagCLI().search(
        "prefix caching",
        k=1,
        index_directory=str(tmp_path),
    )
    result = StudentSearchResults.model_validate_json(output)

    assert result.k == 1
    assert len(result.search_results) == 1
    assert result.search_results[0].retrieved_sources == [
        MinimalSource(
            file_path="sample.py",
            first_character_index=0,
            last_character_index=14,
        )
    ]


def test_single_search_rejects_empty_query() -> None:
    """An empty search query returns a controlled CLI error."""
    assert RagCLI().search("") == "Error: query must not be empty."


# 2000 不只是默认值，也是 subject 的硬上限；CLI 必须在读取 corpus 前
# 拒绝 0、负数和大于 2000 的值，避免崩溃或生成无效来源。
def test_index_rejects_chunk_sizes_outside_subject_limit() -> None:
    cli = RagCLI()
    expected = (
        "Error: max_chunk_size must be between 1 and 2000 characters."
    )

    assert cli.index(max_chunk_size=0) == expected
    assert cli.index(max_chunk_size=-1) == expected
    assert cli.index(max_chunk_size=2001) == expected


# CLI 之外的调用者也可能直接使用 build_index()；底层边界必须重复
# 验证，不能让负数穿透到 rank_bm25 后才变成 ZeroDivisionError。
def test_build_index_rejects_negative_chunk_size(tmp_path: Path) -> None:
    with pytest.raises(
        ValueError,
        match="max_chunk_size must be between 1 and 2000 characters",
    ):
        BM25Index.build_index(
            tmp_path,
            max_chunk_size=-1,
            show_progress=False,
        )


# 空目录没有任何 chunk；应由我们返回明确错误，而不是让 BM25Okapi
# 在计算平均文档长度时抛 ZeroDivisionError。
def test_index_empty_corpus_returns_controlled_error(
    tmp_path: Path,
) -> None:
    raw_directory = tmp_path / "raw"
    raw_directory.mkdir()
    index_directory = tmp_path / "processed"

    result = RagCLI().index(
        raw_directory=str(raw_directory),
        index_directory=str(index_directory),
    )

    assert result == (
        f"Error: No indexable chunks found in corpus: {raw_directory}"
    )
    assert not (index_directory / "bm25_index.pkl").exists()


# index/search_dataset/answer_dataset 的输出参数都必须是目录。若该路径
# 已经是文件，应在任何耗时工作之前返回同一个受控错误。
def test_cli_rejects_output_directory_that_is_a_file(
    tmp_path: Path,
) -> None:
    blocked_path = tmp_path / "blocked"
    blocked_path.write_text("not a directory", encoding="utf-8")
    expected = (
        f"Error: expected an output directory, got file: {blocked_path}"
    )
    cli = RagCLI()

    assert cli.index(
        raw_directory=str(tmp_path / "unused-corpus"),
        index_directory=str(blocked_path),
    ) == expected
    assert cli.search_dataset(
        dataset_path=str(tmp_path / "unused-dataset.json"),
        save_directory=str(blocked_path),
    ) == expected
    assert cli.answer_dataset(
        student_search_results_path=str(tmp_path / "unused-results.json"),
        save_directory=str(blocked_path),
    ) == expected


# 用 monkeypatch 模拟 Hugging Face 离线/权重缺失，不访问网络。单条和
# 批量 answer 都应返回 Error 字符串，不能把第三方 OSError traceback
# 泄漏给 CLI 用户。
def test_answer_commands_handle_model_load_error(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    index_directory = tmp_path / "index"
    _save_tiny_index(index_directory)
    student_results_path = tmp_path / "search-results.json"
    student_results_path.write_text(
        StudentSearchResults(search_results=[], k=1).model_dump_json(),
        encoding="utf-8",
    )

    def fail_model_load(_model_name: str) -> NoReturn:
        raise OSError("model unavailable")

    monkeypatch.setattr("src.cli.AnswerGenerator", fail_model_load)
    expected = (
        "Error: model 'missing/model' could not be loaded: "
        "model unavailable"
    )

    assert RagCLI().answer(
        "prefix caching",
        k=1,
        index_directory=str(index_directory),
        model_name="missing/model",
    ) == expected
    assert RagCLI().answer_dataset(
        student_search_results_path=str(student_results_path),
        save_directory=str(tmp_path / "answers"),
        model_name="missing/model",
    ) == expected


# 测试 Python 代码分块：不管切成几个 chunk，每个 chunk 的
# first_character_index/last_character_index 都必须能精确切回原文，
# 即 text[first:last] == chunk.text。这是整个项目最关键的不变量——
# 一旦这里错了，moulinette 对比 file_path + 字符范围时就会全部不匹配。
def test_chunk_python_code_offsets_reconstruct_source() -> None:
    text = "def a():\n    return 1\n\n\ndef b():\n    return 2\n"
    document = Document(file_path="sample.py", text=text)
    chunks = chunk_document(document, max_chunk_size=2000)
    assert chunks
    for chunk in chunks:
        start, end = chunk.first_character_index, chunk.last_character_index
        assert text[start:end] == chunk.text


# 和上面那个一样，但测试 Markdown/文本分块策略（按段落切）。
# 两种分块策略都必须遵守同一条字符偏移规则。
def test_chunk_markdown_text_offsets_reconstruct_source() -> None:
    text = "First paragraph.\n\nSecond paragraph.\n\nThird one.\n"
    document = Document(file_path="sample.md", text=text)
    chunks = chunk_document(document, max_chunk_size=2000)
    assert chunks
    for chunk in chunks:
        start, end = chunk.first_character_index, chunk.last_character_index
        assert text[start:end] == chunk.text


# 测试 read_source_text() 的越界保护：请求的字符范围超出了文件实际长度
# （文件只有 11 个字符，却要读到第 999 个），不应该抛异常崩溃，
# 应该按约定返回空字符串 ""。
def test_read_source_text_out_of_range_returns_empty_string(
    tmp_path: Path,
) -> None:
    file_path = tmp_path / "doc.txt"
    file_path.write_text("hello world", encoding="utf-8")
    source = MinimalSource(
        file_path=str(file_path),
        first_character_index=0,
        last_character_index=999,
    )
    assert read_source_text(source) == ""


# 测试 read_source_text() 遇到根本不存在的文件时：同样不应该崩溃，
# 应该返回空字符串。answer_dataset 依赖这个函数从磁盘找回 chunk 原文，
# 如果它一遇到坏路径就抛异常，整个 answer_dataset 命令都会崩溃。
def test_read_source_text_missing_file_returns_empty_string() -> None:
    source = MinimalSource(
        file_path="does/not/exist.txt",
        first_character_index=0,
        last_character_index=5,
    )
    assert read_source_text(source) == ""


# 测试 _iou()（IoU = 交集长度 / 并集长度）：两个完全相同的区间，
# 重叠比例应该正好是 1.0（100% 重叠）。这是 evaluate 命令算 recall@k
# 时判断"检索结果是否命中"的核心公式，必须先确认它本身算对。
def test_iou_identical_ranges_is_one() -> None:
    assert _iou(0, 10, 0, 10) == 1.0


# 测试 _iou()：两个完全不重叠的区间（0-10 和 20-30），IoU 应该是 0，
# 不应该出现除零错误或负数。
def test_iou_disjoint_ranges_is_zero() -> None:
    assert _iou(0, 10, 20, 30) == 0.0


# 测试 _is_match()：即使字符范围完全重叠，只要 file_path 不一样
# （a.py 对 b.py），也绝对不能算命中。subject 原文强调"file_path 必须
# 精确匹配"，这一条就是在守住这个规则。
def test_is_match_requires_same_file_path() -> None:
    ground_truth = MinimalSource(
        file_path="a.py", first_character_index=0, last_character_index=100
    )
    retrieved = MinimalSource(
        file_path="b.py", first_character_index=0, last_character_index=100
    )
    assert _is_match(ground_truth, retrieved) is False
