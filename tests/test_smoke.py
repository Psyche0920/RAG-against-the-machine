"""Fast smoke tests for core pipeline invariants.

Not part of the graded submission (see the subject's "Additional
Guidelines": test programs are for your own verification, not submitted
or graded). Kept small and dependency-light so `make test` runs in
seconds and never needs the corpus or a downloaded model.
"""

from pathlib import Path

from src.chunking import chunk_document
from src.cli import _is_match, _iou
from src.documents import read_source_text
from src.index import tokenize
from src.models import Document, MinimalSource


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
