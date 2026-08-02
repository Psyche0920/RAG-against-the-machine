"""建立、保存、加载和搜索基于 Chunk 的 BM25 词法索引。"""

import pickle
import re
from pathlib import Path
from typing import Final

from rank_bm25 import BM25Okapi

from src.chunking import DEFAULT_MAX_CHUNK_SIZE, chunk_documents
from src.documents import load_documents
from src.models.models import Chunk


# 如果调用者没有指定目录，索引默认保存在这里。
DEFAULT_INDEX_DIRECTORY: Final[str] = "data/processed"

# 索引在上述目录中使用的固定文件名。
_INDEX_FILE_NAME: Final[str] = "bm25_index.pkl"

# 这个正则表达式的意思是：
#
#     [A-Za-z0-9_]+
#
# 方括号 ``[...]`` 表示“允许的单个字符”：
#
# - ``A-Z``：英文大写字母；
# - ``a-z``：英文小写字母；
# - ``0-9``：数字；
# - ``_``：下划线。
#
# 最后的 ``+`` 表示：连续取一个或更多个上述字符，
# 把连续的整段当成一个 token。它不是“每个字符一个 token”。
#
# 例如：
#
#     "2000"                 -> ["2000"]
#     "2,000"                -> ["2", "000"]
#     "max_chunk_size"       -> ["max_chunk_size"]
#     "hello,world."         -> ["hello", "world"]
#     "BM25(text)"           -> ["BM25", "text"]
#     "vllm-0.10.1"          -> ["vllm", "0", "10", "1"]
#
# 因此，连续的 ``2000`` 四个数字会合在一起，成为一个
# token ``"2000"``。只有当中间出现不在允许集合中的字符，
# 例如空格、逗号、句号、括号或连字符 ``-``，才会在那里
# 断开。这些标点本身不会进入返回的 token 列表。
#
# 这不是说“一个长度为 2000 字符的 Chunk 只有一个 token”。
# 普通的 2000 字符文本包含很多空格和标点，所以会分成很多
# token。只有在极端情况下，这 2000 个字符全部是连续的
# 字母/数字/下划线，中间没有任何分隔符，才会成为一个
# 长度为 2000 的巨大 token。
_TOKEN_PATTERN: Final[re.Pattern[str]] = re.compile(r"[A-Za-z0-9_]+")


def tokenize(text: str) -> list[str]:
    """把原始文字转换成 BM25 可以使用的小写 token 列表。

    Args:
        text: 需要分词的原始文字。

    Returns:
        由英文字母、数字或下划线组成的小写 token，
        顺序与它们在原文中的出现顺序相同。
    """
    # 第一步：lower() 让 "Chunk" 和 "chunk" 都变成
    # "chunk"，搜索时不再区分英文大小写。
    lowercase_text = text.lower()

    # 第二步：findall() 找出所有符合 _TOKEN_PATTERN 的
    # 连续文本，并按它们在原文中的出现顺序返回。
    #
    # 例如 lowercase_text == "size: 2,000 chars."：
    #
    # - "size" 匹配；
    # - ": " 不匹配，只起分隔作用；
    # - "2" 匹配；
    # - "," 不匹配，所以 2 和 000 被分开；
    # - "000" 匹配；
    # - "chars" 匹配；
    # - 句号 "." 不匹配，也不会出现在结果中。
    #
    # 最终返回 ["size", "2", "000", "chars"]。
    return _TOKEN_PATTERN.findall(lowercase_text)


class BM25Index:
    """对 Chunk 语料建立的 BM25 词法索引。

    这是一个提供索引和搜索行为的 service class，不是
    Pydantic data model。两者责任不同：

    - ``Chunk`` 等 data model 要在加载、索引、搜索和输出阶段
      之间传递，需要 Pydantic 验证字段类型并转换为 JSON；
    - ``BM25Index`` 是执行搜索的工作对象，内部包含第三方
      ``BM25Okapi`` 对象和搜索方法，不是用来与外部交换的
      JSON 数据格式。

    因此题目要求“跨阶段数据”使用 Pydantic，但明确允许
    indexer、retriever 和 pipeline 这类服务对象使用普通类。
    """

    def __init__(self, chunks: list[Chunk], bm25: BM25Okapi) -> None:
        """保存 Chunk 语料和使用该语料建立的 BM25 模型。

        Args:
            chunks: Chunk 列表，顺序必须与建立 ``bm25`` 时一致。
            bm25: 使用分词后的 Chunk 建立的 BM25Okapi 模型。
        """
        # BM25 返回的第 i 个分数对应 self.chunks[i]，因此
        # 建立索引之后不能打乱 chunks 的顺序。
        self.chunks = chunks
        self.bm25 = bm25

    def search(self, query: str, k: int) -> list[Chunk]:
        """返回与查询最相关的前 k 个 Chunk。

        Args:
            query: 自然语言或代码搜索问题。
            k: 最多返回的 Chunk 数量。

        Returns:
            按 BM25 得分从高到低排列的 Chunk，最多 ``k`` 个。
            如果语料为空、查询没有可用 token，或 ``k`` 不是
            正数，则返回空列表。
        """
        # k <= 0 表示不需要返回结果。空语料也无法搜索。
        if k <= 0 or not self.chunks:
            return []

        # query 必须使用和文档相同的 tokenize() 规则，因为
        # BM25 比较的是 token 是否相同，不是原始字符串是否看起来
        # 相似。例如建立索引时把 "MAX_SIZE" 处理成一个小写
        # token "max_size"，查询也必须得到同样的 "max_size"，
        # 才能命中文档。如果查询使用另一套分词规则，例如
        # 把下划线拆成 "max" 和 "size"，就无法与索引中的
        # "max_size" 直接匹配。
        query_tokens = tokenize(query)
        if not query_tokens:
            return []

        # get_scores() 会给语料中的每个 Chunk 计算一个分数。
        # scores 和 self.chunks 的长度相同，而且位置一一对应：
        #
        #     scores[0] <-> self.chunks[0]
        #     scores[1] <-> self.chunks[1]
        #     scores[2] <-> self.chunks[2]
        scores = self.bm25.get_scores(query_tokens)

        # 举一个具体例子。假设：
        #
        #     self.chunks = [chunk_a, chunk_b, chunk_c]
        #     scores = [0.2, 3.5, 1.1]
        #
        # range(len(self.chunks)) 首先产生下标 [0, 1, 2]。
        # key=lambda index: scores[index] 告诉 sorted：
        #
        #     下标 0 的排序值是 scores[0] == 0.2
        #     下标 1 的排序值是 scores[1] == 3.5
        #     下标 2 的排序值是 scores[2] == 1.1
        #
        # reverse=True 表示从高分到低分排序，所以结果是
        # ranked_indices == [1, 2, 0]。它的意思是 chunk_b 第一名、
        # chunk_c 第二名、chunk_a 第三名。
        ranked_indices = sorted(
            range(len(self.chunks)),
            key=lambda index: scores[index],
            reverse=True,
        )

        # [:k] 只取排名最前面的 k 个下标，再取出对应
        # Chunk。例如 k == 2 时，ranked_indices[:2] == [1, 2]，
        # 最终返回 [self.chunks[1], self.chunks[2]]，也就是
        # [chunk_b, chunk_c]。
        return [self.chunks[index] for index in ranked_indices[:k]]

    # 这些函数保留在你原来的 BM25Index 类中。普通实例
    # 方法会自动收到调用它的对象 self：
    #
    #     index.search("query", 5)
    #
    # Python 实际会像这样传参：
    #
    #     BM25Index.search(index, "query", 5)
    #                        ^^^^^
    #                        这就是 self
    #
    # 但 build_index() 建立新索引时还没有现成 index 对象，它只需要
    # raw_directory 等参数，不读取 self.chunks 或 self.bm25。save_index()
    # 和 load_index() 也不依赖“调用这个方法的 self”。
    #
    # @staticmethod 告诉 Python：这个函数只是放在类的命名空间中，
    # 调用时不要自动插入 self。因此可以写：
    #
    #     BM25Index.build_index("data/raw")
    #
    # 如果没有 @staticmethod，类中方法的第一个参数按惯例应该是
    # self，mypy 也会报告该方法缺少 self 参数。
    @staticmethod
    def build_index(
        raw_directory: str | Path,
        max_chunk_size: int = DEFAULT_MAX_CHUNK_SIZE,
        show_progress: bool = True,
    ) -> "BM25Index":
        """加载、分块并索引语料目录中所有支持的文件。

        Args:
            raw_directory: 源语料的根目录，例如 ``data/raw``。
            max_chunk_size: 每个 Chunk 允许的最大字符数。
            show_progress: 是否显示加载和分块进度条。

        Returns:
            已建立完成、可以搜索或保存的 BM25Index。

        Raises:
            FileNotFoundError: ``raw_directory`` 不存在。
            NotADirectoryError: ``raw_directory`` 不是目录。
        """
        # 第一步：从目录读取支持的文件，得到 Document 列表。
        documents = load_documents(
            raw_directory,
            show_progress=show_progress,
        )

        # 第二步：把 Document 切分成 Chunk 列表。
        chunks = chunk_documents(
            documents,
            max_chunk_size,
            show_progress=show_progress,
        )

        # 第三步：每个 Chunk 变成一个 token 列表。
        # tokenized_corpus[i] 必须始终对应 chunks[i]。
        tokenized_corpus = [tokenize(chunk.text) for chunk in chunks]

        # 第四步：使用分词后的整个语料建立 BM25 模型。
        bm25 = BM25Okapi(tokenized_corpus)

        # 这里不是“一个 Chunk 建立一个 BM25 模型”。
        #
        # chunks 是整个语料的 Chunk 列表：
        #
        #     chunks = [chunk_a, chunk_b, chunk_c, ...]
        #
        # bm25 是使用整个 tokenized_corpus 一次性建立的一个模型。
        # 它需要看到全部 Chunk，才能统计一个词在整个语料中有多稀有。
        #
        # BM25Index(chunks=chunks, bm25=bm25) 创建一个容器对象，同时
        # 保存“全部 Chunk 列表”和“整个语料的一个 BM25 模型”。
        # 搜索时，模型返回每个位置的分数，再通过相同位置取回
        # chunks 中对应的 Chunk。
        return BM25Index(chunks=chunks, bm25=bm25)

    @staticmethod
    def save_index(
        index: "BM25Index",
        directory_path: str | Path = DEFAULT_INDEX_DIRECTORY,
    ) -> Path:
        """把索引保存到磁盘，便于之后直接加载而无需重新建立。

        Args:
            index: 需要保存的 BM25Index。
            directory_path: 用于写入索引文件的目录。

        Returns:
            已写入的索引文件路径。
        """
        directory = Path(directory_path)

        # parents=True 会创建缺少的父目录，exist_ok=True 表示
        # 目录已经存在时不报错。
        directory.mkdir(parents=True, exist_ok=True)
        index_path = directory / _INDEX_FILE_NAME

        # "wb" 是 write binary（二进制写入）：
        #
        # - "w" 表示打开文件用于写入；
        # - "b" 表示写入 bytes 二进制数据，不是 str 文字。
        #
        # pickle.dump(index, index_file) 会把整个 Python 对象图转换成
        # pickle 二进制格式，包括 BM25Index、chunks 列表和 BM25
        # 模型内部数据。这个文件是 bm25_index.pkl，不是 JSON。
        # JSON 只能直接表示字典、列表、字符串、数字等基本数据，
        # 不知道如何直接还原 BM25Okapi 这种 Python 对象。
        #
        # with 块结束时文件会自动关闭。
        with index_path.open("wb") as index_file:
            pickle.dump(index, index_file)

        return index_path

    @staticmethod
    def load_index(
        directory_path: str | Path = DEFAULT_INDEX_DIRECTORY,
    ) -> "BM25Index":
        """从磁盘加载之前保存的 BM25 索引。

        Args:
            directory_path: 包含已保存索引文件的目录。

        Returns:
            从 pickle 数据还原的 BM25Index。

        Raises:
            FileNotFoundError: 指定目录中不存在索引文件。
        """
        index_path = Path(directory_path) / _INDEX_FILE_NAME
        if not index_path.exists():
            raise FileNotFoundError(
                f"No index found at {index_path}; run the index command first."
            )

        # "rb" 是 read binary（二进制读取）。pickle.load() 读取
        # save_index() 写入的 .pkl 二进制数据，并在内存中重新
        # 创建一个 BM25Index Python 对象。
        #
        # save_index() 中的 index 和这里的 index 是不同函数中的
        # 局部变量，同名但互相不可见：
        #
        # - save_index() 的 index：调用者传进来、即将保存的原对象；
        # - load_index() 的 index：从 .pkl 数据新还原的 Python 对象。
        #
        # 它们通常内容等价，但加载出来的是内存中的新对象，
        # 不是 JSON model，也不是原对象的同一个内存实例。
        #
        # 安全注意：pickle.load() 可能执行文件中指定的 Python 代码，
        # 因此只能加载本项目自己生成、可信任的 .pkl 文件。
        with index_path.open("rb") as index_file:
            index: BM25Index = pickle.load(index_file)

        return index


# =============================================================================
# 常见问题总结（FAQ）
# =============================================================================
#
# 1. 为什么 Chunk 使用 Pydantic，BM25Index 不使用？
# -----------------------------------------------------------------------------
# Chunk 是在分块、索引、搜索和输出阶段之间传递的数据，
# 需要验证 file_path、text 和字符下标等字段，因此使用 Pydantic。
#
# BM25Index 是执行搜索的服务对象，它包含搜索方法和第三方
# BM25Okapi 对象，不是要与外部交换的 JSON 数据格式。题目也
# 明确允许 indexer、retriever 和 pipeline 类使用普通 Python 类。
#
#
# 2. 为什么 query 和 Chunk 必须使用相同的 tokenize()？
# -----------------------------------------------------------------------------
# BM25 比较的是 token 字符串。如果文档把 max_size 保留为：
#
#     ["max_size"]
#
# 但 query 使用另一套规则，变成：
#
#     ["max", "size"]
#
# 那么 "max_size" 与 "max"/"size" 都不相等，BM25 无法直接匹配。
# 建立索引和搜索必须共用 tokenize()，才能得到相同 token。
#
#
# 3. scores 、ranked_indices 和 chunks 如何对应？
# -----------------------------------------------------------------------------
# 假设：
#
#     chunks = [chunk_a, chunk_b, chunk_c]
#     scores = [0.2, 3.5, 1.1]
#
# 它们使用相同下标一一对应：
#
#     chunks[0] <-> scores[0] == 0.2
#     chunks[1] <-> scores[1] == 3.5
#     chunks[2] <-> scores[2] == 1.1
#
# sorted 将下标 [0, 1, 2] 按对应分数从高到低排列，得到：
#
#     ranked_indices = [1, 2, 0]
#
# 再用这些下标从 chunks 中取回 [chunk_b, chunk_c, chunk_a]。
#
#
# 4. 什么时候需要 self，什么时候需要 @staticmethod？
# -----------------------------------------------------------------------------
# search() 要读取某个已存在对象的 self.chunks 和 self.bm25，所以
# 它是普通实例方法。调用 index.search(...) 时，Python 会自动把
# index 对象传给 self。
#
# build_index()、save_index() 和 load_index() 不读取调用它们的 self，
# 因此标记为 @staticmethod。@staticmethod 表示调用时不要自动
# 插入 self。如果不想使用 @staticmethod，另一个正确做法是把
# 这些函数移到 BM25Index 类外，变成普通模块函数。
#
#
# 5. staticmethod 中的 index 是什么？它会自动生成吗？
# -----------------------------------------------------------------------------
# 不会。save_index() 中的 index 只是一个普通参数，必须由调用者
# 手动传入：
#
#     my_index = BM25Index.build_index("data/raw")
#     BM25Index.save_index(my_index)
#                            ^^^^^^^^
#                            手动传入的 index 参数
#
# 如果调用 BM25Index.save_index() 而不传参，Python 会报告缺少
# index 参数。只有普通实例方法的 self 才会由 Python 自动传入。
#
#
# 6. chunks 和 bm25 是同一个东西吗？
# -----------------------------------------------------------------------------
# 不是。可以把 chunks 想成图书馆中的书，bm25 想成搜索目录。
#
# chunks 保存需要最终返回的原始内容：文字、文件路径和字符下标。
# bm25 保存用于搜索的统计数据：token 频率、文档长度、IDF 等。
#
# 整个语料只建立一个 BM25 模型，不是每个 Chunk 建立一个。
# BM25 必须看到全部 Chunk，才能判断一个 token 在整个语料中是常见
# 还是稀有。BM25Index 对象因此同时保存：
#
#     BM25Index
#     ├── chunks：全部 Chunk 列表
#     └── bm25：整个语料共用的一个 BM25 模型
#
#
# 7. pickle 是什么？为什么不直接使用 JSON？
# -----------------------------------------------------------------------------
# pickle 是 Python 自带的对象序列化工具。pickle.dump() 把内存中
# 的 Python 对象转换成可写入 .pkl 文件的二进制数据；
# pickle.load() 再把这些二进制数据还原为 Python 对象。
#
# JSON 只能直接表示字符串、数字、列表和字典等基本结构，不知道
# 如何直接还原 BM25Okapi 这种复杂的第三方 Python 对象。
# 并不是绝对不能使用 JSON，但如果使用 JSON，就需要自己设计保存
# 格式，并在加载时重新建立 BM25Okapi。pickle 在这里更直接。
#
# 安全注意：只能 pickle.load() 本项目自己生成的可信 .pkl 文件，
# 不能加载来源不明的 pickle，因为恶意 pickle 可能执行代码。
#
#
# 8. save_index() 和 load_index() 中同名的 index 是什么？
# -----------------------------------------------------------------------------
# 它们是两个不同函数中的局部变量，互相不可见：
#
# - save_index() 的 index：调用者传入、即将保存的 BM25Index 对象；
# - load_index() 的 index：从 .pkl 文件还原出来的新 BM25Index 对象。
#
# 中间文件是 pickle 二进制文件，不是 JSON：
#
#     原 BM25Index 对象
#              ↓ pickle.dump()
#       bm25_index.pkl
#              ↓ pickle.load()
#     新 BM25Index 对象
#
# 两个对象的内容通常等价，但它们不是内存中的同一个实例。
