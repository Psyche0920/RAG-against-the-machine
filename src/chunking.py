"""Chunking strategies that split loaded documents into indexable spans."""

import ast
import re
from pathlib import Path
from typing import Final, Iterable, Iterator

from tqdm import tqdm

from src.models.models import Chunk, Document

DEFAULT_MAX_CHUNK_SIZE: Final[int] = 2000

# 这一行同时使用了 Final 和 frozenset，但它们限制的是两件不同的事。
#
# 1. frozenset[str]
#    - [str] 表示集合中的每个值都应该是字符串；
#    - frozen 表示集合创建后，不能增加或删除值。
#
#    例如：
#        suffixes = frozenset({".py", ".pyi"})
#        suffixes.add(".js")       # 错误：frozenset 不能增加值
#        suffixes.remove(".py")    # 错误：frozenset 不能删除值
#
#    但是如果没有 Final，变量名仍然可以重新指向一个全新的 frozenset：
#        suffixes = frozenset({".js"})  # 可以；这不是修改旧集合，而是换新集合
#
# 2. Final
#    Final 告诉 mypy 等类型检查器：这个变量名只能赋值一次，以后不要让它
#    指向另一个对象。Final 本身并不会让一个普通 set 变成不可修改的集合。
#
#    例如：
#        suffixes: Final[set[str]] = {".py"}
#        suffixes.add(".js")  # 可以：Final 没有禁止修改普通 set 的内容
#        suffixes = {".txt"}  # 类型检查错误：Final 禁止变量名重新赋值
#
# 3. 两者放在一起
#        suffixes: Final[frozenset[str]] = frozenset({".py"})
#        suffixes.add(".js")                 # frozenset 禁止增加值
#        suffixes = frozenset({".txt"})      # Final 禁止重新赋值
#
# 如果先写 a = ".py"，再写 frozenset({a})，集合保存的是字符串值 ".py"，
# 不是变量 a 本身。以后执行 a = ".js" 不会改变集合中的 ".py"。
_CODE_SUFFIXES: Final[frozenset[str]] = frozenset({".py", ".pyi"})

# 只有这些已知的文档类型才使用 Markdown/text 分段策略。
# .rst 是 reStructuredText，不是 Markdown；但它仍是按段落组织的
# 纯文本，因此这里把它归入通用 text 策略。
_TEXT_SUFFIXES: Final[frozenset[str]] = frozenset({".md", ".rst", ".txt"})

# 一个 ``\n`` 只表示“换到下一行”，不一定开始新段落。
# 例如 Markdown 的同一段可以在源文件中写成多行：
#
#     This is still one
#     paragraph.
#
# 两个 ``\n`` 表示两行之间有一个空行，这才是 Markdown/
# 普通文本中常见的段落边界。``{2,}`` 表示匹配连续两个
# 或更多 ``\n``。如果改成只匹配一个 ``\n``，每一行都会被当成
# 独立段落。
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
                # window 确实包含它的最后一个字符。因为 Python 切片的右边界
                # 不包含在结果中，所以最后一个字符的下标要用“右边界 - 1”。
                last_character_index=start_offset + start + len(window) - 1
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
    # 请分清两套数字：
    #
    # 1. start/end 是代码内部给 Python 切片使用的边界；
    # 2. first_character_index/last_character_index 是最终输出的字符下标。
    #
    # 对 text = "abc" 来说：
    #
    #     start = 0
    #     end = 3                         # 切片的排他右边界
    #     span = text[0:3] == "abc"       # a、b、c 全部包含
    #     first_character_index = 0
    #     last_character_index = end - 1  # 2，c 的真实下标
    #
    # 因此最终 Chunk 是 "abc" 且下标为 (0, 2)，不是 "ab"。
    # Python 统一使用“左边包含，右边不包含”的切片规则：
    #
    #     text = "ABCDE"
    #     text[1:4] == "BCD"
    #
    # 它包含下标 1、2、3，不包含下标 4。因此 end 不是
    # “最后一个字符的下标”，而是“最后一个字符之后的
    # 边界”。这样切片长度可以直接计算为 end - start：
    # 4 - 1 == 3，正好是 "BCD" 的三个字符。
    span = text[start:end]
    if not span.strip():
        return []

    if len(span) < max_chunk_size:
        return [
            Chunk(
                file_path=file_path,
                text=span,
                first_character_index=start,
                # span == text[start:end]，end 是不包含在切片中的右边界；
                # 因此 span 实际包含的最后一个字符位于 end - 1。
                last_character_index=end - 1,

            )
        ]
    return _split_fixed_size(span, file_path, max_chunk_size, start)


def _line_start_offsets(text: str) -> list[int]:
    """计算每一行在完整字符串中的起始偏移量（start offset）。

    这个函数的核心用途：连接 AST 节点与字符切片
    ------------------------------------------
    是的，这个函数就是 AST nodes 和 character splitting 之间的桥梁。

    AST 分析 Python 代码后，会产生函数节点、类节点、赋值节点等。AST
    使用“行号”描述节点的位置：

    - ``node.lineno``：这个节点从原始代码的第几行开始；
    - ``node.end_lineno``：这个节点在原始代码的第几行结束。

    例如 ``node.end_lineno == 2`` 的意思是：

    “这个 AST 节点最后占用了原始代码的第 2 行。”

    但是 AST 不直接告诉我们：

    “第 2 行结束后，是原始字符串中的第几个字符位置？”

    但是字符串切片需要字符位置：

    ``text[开始字符位置:结束字符位置]``

    因此不能直接写 ``text[node.lineno:node.end_lineno]``，因为这里的
    两个数字是行号，不是字符位置。

    本函数建立一个 ``line_starts`` 翻译表，把 AST 的行号翻译成字符
    splitting 可以使用的字符位置。完整转换过程是：

    ``AST 节点 -> 节点的结束行号 -> 查 line_starts -> 字符结束位置``

    调用处执行：

    ``end = line_starts[end_lineno]``

    如果 ``end_lineno == 2``，它会取 ``line_starts[2]``。这个数字表示
    “第 2 行结束之后的位置”，因此切片能够完整包含第 2 行。

    得到字符位置后，``_chunk_span`` 才能执行：

    ``span = text[start:end]``

    这一步才是真正按照字符切割原始文本。如果这个 span 仍然太长，
    ``_split_fixed_size`` 会继续把它切成更小的固定字符窗口。

    先忘记 ``offset`` 这个单词。它在这里其实就是“字符的位置号码”。
    Python 从 0 开始给字符编号。

    例如 ``text = "ab\\ncde\\n"``，把每个字符和下标写出来是：

    ``a(0) b(1) \\n(2) c(3) d(4) e(5) \\n(6)``

    我们只想记住“每一行从哪个号码开始”：

    - 第 1 行从 ``a(0)`` 开始，所以记录 0；
    - 第 2 行从 ``c(3)`` 开始，所以记录 3；
    - 所有文字之后的位置是 7，所以最后还记录 7。

    所以本函数最后返回 ``[0, 3, 7]``。

    返回值是一个普通的整数列表：

    - ``返回值[0] == 0``：第 1 行从字符位置 0 开始；
    - ``返回值[1] == 3``：第 2 行从字符位置 3 开始；
    - ``返回值[2] == 7``：第 2 行结束后的位置是 7。

    有了它，就能取出完整的第 2 行：
    ``text[返回值[1]:返回值[2]]``，也就是 ``text[3:7]``，结果为
    ``"cde\\n"``。

    Args:
        text: 完整的文档文本，而不是某一行或某一个 chunk。

    Returns:
        ``offsets[i]`` 保存第 ``i + 1`` 行的起始偏移量，因为：

        - Python 列表从 0 开始编号；
        - ``ast`` 的 ``lineno`` / ``end_lineno`` 从 1 开始编号。

        例如 ``offsets == [0, 3, 7]``：

        - ``offsets[0] == 0``：第 1 行的开始位置；
        - ``offsets[1] == 3``：第 2 行的开始位置，同时也是第 1 行之后；
        - ``offsets[2] == 7``：第 2 行之后的位置。

        因此，当 AST 给出 ``end_lineno == 2`` 时，故意读取
        ``offsets[2]``，得到第 2 行之后的位置 7。这里不是要寻找“第 2
        行的开始”，而是要寻找“AST 节点最后一行之后的切片边界”。

        列表末尾还会有一个等于 ``len(text)`` 的结束边界。注意它通常不
        是“最后一个字符的下标”；非空文本最后一个字符的下标是
        ``len(text) - 1``，而 ``len(text)`` 是最后一个字符后面的位置。
    """
    # offsets 是本函数最后要返回的列表。
    #
    # 英文 offsets 是 offset 的复数，意思是“多个位置号码”。
    # 第 1 行永远从字符位置 0 开始，所以列表一开始是 [0]。
    offsets = [0]

    # splitlines 把完整文章拆成一行一行。
    # 对 "ab\ncde\n" 来说，keepends=True 会得到：
    #
    #   第一次循环：line = "ab\n"，长度是 3
    #   第二次循环：line = "cde\n"，长度是 4
    #
    # keepends=True 表示不要扔掉每行末尾的 \n。
    # \n 也是真的字符，也占一个位置，所以计算长度时必须保留它。
    for line in text.splitlines(keepends=True):
        # 列表的 [-1] 表示“取最后一个元素”：
        #
        #   [0][-1]       是 0
        #   [0, 3][-1]    是 3
        #   [0, 3, 7][-1] 是 7
        #
        # 因此 offsets[-1] 就是“目前已经数到的位置”。
        current_position = offsets[-1]

        # len(line) 是这一行有多少个字符。
        number_of_characters_in_this_line = len(line)

        # 目前的位置 + 这一行的字符数 = 这一行结束后的位置。
        # 这个位置通常也是下一行开始的位置。
        next_position = (
            current_position + number_of_characters_in_this_line
        )

        # append 表示把新数字放到列表的末尾。
        #
        # 仍以 "ab\ncde\n" 为例：
        #   开始时 offsets 是 [0]
        #   读完 "ab\n"：next_position = 0 + 3 = 3，变成 [0, 3]
        #   读完 "cde\n"：next_position = 3 + 4 = 7，变成 [0, 3, 7]
        offsets.append(next_position)

    # 把算好的列表交给调用这个函数的代码。
    # 例如输入 "ab\ncde\n"，return 的结果就是 [0, 3, 7]。
    return offsets


def chunk_python_code(
        document: Document,
        max_chunk_size: int,
) -> list[Chunk]:
    """Chunk a Python source file along its top-level statement bounds.

    ``ast.parse`` 只解析 Python 语法，不会执行源代码。
    每个顶层函数、类或语句尽量单独形成一个 chunk，让 retrieved
    snippet（搜索返回的 chunk）通常不会跨越两个无关定义。
    如果一个 chunk（例如大型函数）仍超过 ``max_chunk_size``，
    就继续按固定字符长度切分。解析失败时也改用固定长度切分，
    而不是抛出语法错误。

    Args:
        document: Python source document to chunk.
        max_chunk_size: Maximum number of characters per chunk.

    Returns:
        Chunks covering the document, ordered by position in the file.
    """
    text = document.text

    try:
        module = ast.parse(text)
        # print(ast.dump(module, indent=4))
    except SyntaxError:
        return _split_fixed_size(text, document.file_path, max_chunk_size)

    # module.body 只包含顶层语句，不包含空行和普通注释。
    # 因此，空文件或只有注释的文件需要改用固定字符长度切分。
    if not module.body:
        return _split_fixed_size(text, document.file_path, max_chunk_size)

    # 连接 AST nodes 与 character splitting：
    #
    # AST 只给我们节点所在的行号；但 text[start:end] 需要字符位置。
    # line_starts 是“行号 -> 原始 text 中字符位置”的翻译表。
    line_starts = _line_start_offsets(text)
    chunks: list[Chunk] = []
    # cursor 是字符偏移量：它指向第一个尚未处理的字符位置。
    cursor = 0

    for node in module.body:
        # 必须使用 end_lineno，因为 lineno 只是节点的开始行。
        # 对多行函数或类使用 lineno，会在函数体或类体结束前错误切断。
        # 如果缺少真实结束行，改用固定长度切分，不猜测结束位置。
        if node.end_lineno is None:
            return _split_fixed_size(
                text,
                document.file_path,
                max_chunk_size,
            )
        end_lineno = node.end_lineno

        # 第二步：把 AST 行号转换成原始 text 中的字符位置。
        # _line_start_offsets 创建了从行号到字符位置的翻译表。
        # line_starts[2] 是第 2 行之后的字符位置，正好能作为右侧不包含的
        # 切片边界，让 text[cursor:end] 完整包含 AST 节点的最后一行。
        end = line_starts[end_lineno]

        # _chunk_span 返回 list[Chunk]；超长节点可能产生多个 chunk。
        # extend 保持结果是扁平列表，append 则会产生嵌套列表。
        # 注释仍保留在原文 text 中，会进入这次切片覆盖的相邻 chunk。
        chunks.extend(
            _chunk_span(
                text, document.file_path, cursor, end, max_chunk_size,
            )
        )
        cursor = end

    # AST 不为文件末尾的注释和空行创建节点，所以还要处理
    # 最后一个节点之后的剩余文本；如果只剩空白，_chunk_span 会忽略它。
    chunks.extend(
        _chunk_span(
            text, document.file_path, cursor, len(text), max_chunk_size,
        )
    )
    return chunks


def _iter_paragraphs(text: str) -> Iterator[tuple[int, int]]:
    """逐个产生每个段落的 ``(start, end)`` 字符边界。

    这个函数不返回段落文字本身，而是返回段落在原始
    ``text`` 中的开始位置和结束边界。调用者可以用
    ``text[start:end]`` 取回该段落。

    ``_PARAGRAPH_BREAK`` 是正则表达式 ``\n{2,}``，表示连续两个
    或更多换行符，也就是段落之间的空行。

    例如：

    ``text = "AAA\n\nBBBB"``

    字符位置是：

    ``A(0) A(1) A(2) \n(3) \n(4) B(5) B(6) B(7) B(8)``

    两个段落的边界是：

    - 第一段：``(0, 3)``，``text[0:3] == "AAA"``；
    - 第二段：``(5, 9)``，``text[5:9] == "BBBB"``。

    中间的 ``text[3:5] == "\n\n"`` 是段落分隔符，不属于任何
    一个段落。

    Args:
        text: Full document text.

    Yields:
        每次产生一个 ``(start, end)`` 二元组。``start`` 是段落
        第一个字符的下标；``end`` 是段落最后一个字符之后的
        位置，所以可直接用于 ``text[start:end]``。
    """
    # cursor 表示“下一个段落可能开始的字符位置”。
    # 刚开始还没有处理任何字符，所以从位置 0 开始。
    cursor = 0

    # finditer(text) 会按从左到右的顺序，找到每一处“连续两个
    # 或更多换行符”。match 代表当前找到的段落分隔符。
    for match in _PARAGRAPH_BREAK.finditer(text):
        # match.start() 是当前分隔符的第一个字符位置。
        # cursor 到 match.start() 之间的文字就是一个段落。
        #
        # 例如 "AAA\n\nBBBB" 的分隔符从位置 3 开始：
        #   cursor == 0
        #   match.start() == 3
        #   text[0:3] == "AAA"
        #
        # 如果两者相等，它们之间没有文字，因此不产生空段落。
        if match.start() > cursor:
            # yield 和 return 不同：yield 产生一个结果后暂停函数；
            # 下次继续迭代时，函数会从这里后面继续执行。
            yield cursor, match.start()

        # match.end() 是整个分隔符之后的位置。
        # 把 cursor 移到这里，等于跳过中间的空行，下一段
        # 就从分隔符之后开始。在上例中 match.end() == 5。
        cursor = match.end()

    # 循环只能处理“分隔符之前”的段落。最后一个分隔符
    # 之后可能还有最后一段，所以循环结束后要单独检查。
    # cursor < len(text) 说明 cursor 后面确实还有字符。
    if cursor < len(text):
        # 最后一段从 cursor 开始，一直到整个文本的结束边界。
        # 对 "AAA\n\nBBBB" 来说，这里产生 (5, 9)。
        yield cursor, len(text)


def chunk_markdown_text(
        document: Document,
        max_chunk_size: int,
) -> list[Chunk]:
    """按段落边界切分 Markdown 或普通文本文档。

    这个函数先调用 ``_iter_paragraphs``，得到每个段落在原始
    ``text`` 中的 ``(para_start, para_end)`` 字符边界。

    它不会立刻把每个段落都变成独立 chunk，而是先放入一个
    buffer（缓冲区）。只要“缓冲区 + 新段落”不超过
    ``max_chunk_size``，就继续合并。这样短小的相邻段落可以留在
    同一个 chunk 中。

    如果加入新段落会超过上限，就先输出已有缓冲区，
    再用新段落开始下一个缓冲区。

    重要：超长段落会被独立定长切分，它的尾部不会与下一段
    再次合并。例如 ``max_chunk_size == 50`` 且第一段长度是
    52，第一段会产生长度为 50 和 2 的两个 chunk。后面长度为
    2 的 chunk 仍属于第一段，不会被放回段落 buffer 中与第二
    段合并。这保持了“超长段落自己切分”的简单语义，但代价是
    可能产生一个很短的尾部 chunk。

    例如，三个段落的边界为：

    ``(0, 20), (22, 40), (42, 100)``

    当 ``max_chunk_size == 50`` 时：

    - 第一段先进入 buffer：``[0:20]``；
    - 第二段加入后是 ``[0:40]``，长度 40，可以合并；
    - 第三段加入后会成为 ``[0:100]``，超过 50；
    - 因此先输出 ``text[0:40]``，再用 ``[42:100]`` 开始新
      buffer。这个单独段落本身超过 50，``_chunk_span``
      会再把它交给 ``_split_fixed_size`` 进行定长字符切分。

    Args:
        document: 要切分的 Markdown 或普通文本文档。
        max_chunk_size: 每个 chunk 允许的最大字符数。

    Returns:
        按原文顺序排列的 ``list[Chunk]``。
    """
    # Document 同时保存 file_path 和 text。这里取出完整原文，
    # 后面的所有 start/end 都是相对这个 text 的字符位置。
    text = document.text

    # chunks 存放已经确定并输出的 Chunk。开始时还没有任何
    # 结果，所以是空列表。
    chunks: list[Chunk] = []

    # buffer_start 和 buffer_end 不保存文字副本，只保存当前
    # 缓冲区在原始 text 中的字符边界。缓冲区的文字是：
    #
    #     text[buffer_start:buffer_end]
    #
    # 初始数字 0 只是占位值；has_buffer == False 说明它们现在
    # 还不代表真正的段落。
    buffer_start = 0
    buffer_end = 0
    has_buffer = False

    # _iter_paragraphs 每次 yield 一个段落的字符边界。
    # para_start 是该段首字符下标，para_end 是末字符之后的位置。
    for para_start, para_end in _iter_paragraphs(text):
        # 情况 1：缓冲区还是空的。
        # 让当前段落成为缓冲区中的第一个段落。
        if not has_buffer:
            buffer_start = para_start
            buffer_end = para_end
            has_buffer = True

        # 情况 2：缓冲区已有段落，而且加入当前段落后仍不超长。
        # para_end - buffer_start 计算“缓冲区开头到新段落末尾”
        # 的总长度。这个范围也包含两个段落之间的原始空行。
        elif para_end - buffer_start <= max_chunk_size:
            # 起点不变，只把结束边界扩展到新段落末尾。
            buffer_end = para_end

        # 情况 3：加入新段落就会超过 max_chunk_size。
        else:
            # 先把旧缓冲区转换成 Chunk。_chunk_span 返回列表，
            # 所以使用 extend 把其中的 Chunk 逐个加入 chunks。
            # 如果旧 buffer 本身是长度 52 的单个超长段落，而上限
            # 是 50，_chunk_span 会在这里一次性返回 50 + 2 两个
            # Chunk。那个长度 2 的尾部不再参与下一段的 buffer 合并。
            chunks.extend(
                _chunk_span(
                    text, document.file_path, buffer_start, buffer_end,
                    max_chunk_size,
                )
            )

            # 旧缓冲区已经输出。现在用当前新段落开始下一个
            # 缓冲区。has_buffer 仍然是 True。
            buffer_start = para_start
            buffer_end = para_end

    # for 循环结束时，最后一个缓冲区还没有输出，因为它
    # 后面没有“下一段”来触发 else。因此必须在循环外单独输出。
    # 如果文档为空，has_buffer 仍为 False，不会产生空 Chunk。
    if has_buffer:
        chunks.extend(
            _chunk_span(
                text, document.file_path, buffer_start, buffer_end,
                max_chunk_size,
            )
        )

    # 循环中的 chunks.extend 可能已加入若干 Chunk，上面又加入
    # 最后的缓冲区。现在返回完整结果列表。
    return chunks


def chunk_document(
        document: Document,
        max_chunk_size: int = DEFAULT_MAX_CHUNK_SIZE,
) -> list[Chunk]:
    """根据已知文件后缀，把文档交给对应的分块策略。

    题目要求两种分块策略：Python code 和 Markdown/text。题目并没有
    规定具体必须使用 AST 或空行分段；这些是本项目的实现选择。

    通常 ``document`` 来自 ``load_document/load_documents``，而加载器已经
    只允许 ``.py``、``.pyi``、``.md``、``.rst`` 和 ``.txt``。但
    ``Document`` 模型本身不会验证后缀，调用者也可以手工构造
    ``Document(file_path="x.js", ...)``。所以本函数仍使用明确
    白名单，而不是把“任何非 Python 文件”默认当成 Markdown。

    Args:
        document: Document to chunk.
        max_chunk_size: Maximum number of characters per chunk.

    Returns:
        Chunks produced by the strategy appropriate for the document.

    Raises:
        ValueError: 文件后缀不属于任何已支持的分块策略。
    """
    suffix = Path(document.file_path).suffix.lower()
    if suffix in _CODE_SUFFIXES:
        return chunk_python_code(document, max_chunk_size)
    if suffix in _TEXT_SUFFIXES:
        return chunk_markdown_text(document, max_chunk_size)
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
