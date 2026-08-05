"""Answer generation using a local Hugging Face causal language model."""

from collections.abc import Callable
from typing import Final, cast

from transformers import AutoModelForCausalLM, AutoTokenizer

DEFAULT_MODEL_NAME: Final[str] = "Qwen/Qwen3-0.6B"
# 字符与 token 没有固定换算；英文通常数个字符为一个 token，
# 中文、代码和标点会不同，准确数量必须由 Qwen tokenizer 计算。
DEFAULT_MAX_NEW_TOKENS: Final[int] = 512
# 这里限制完整 prompt 的 token 数，不同于 subject 中单个
# chunk 最多 2000 个字符的限制；6000 + 512 低于模型上下文上限。
DEFAULT_MAX_INPUT_TOKENS: Final[int] = 6000

_SYSTEM_PROMPT: Final[str] = (
    "You are a helpful assistant answering questions about the vLLM "
    "codebase. Answer only using the provided sources. If the sources "
    "do not contain the answer, say so instead of guessing."
)


def build_prompt(
    question: str,
    chunk_texts: list[str],
) -> list[dict[str, str]]:
    """Build the chat messages passed to the model.

    Args:
        question: The user's question.
        chunk_texts: Text from retrieved chunks, most relevant first.

    Returns:
        Chat messages ready for 'tokenizer.apply_chat_template'.
    """
    if chunk_texts:
        numbered = "\n\n".join(
            f"[Source {index + 1}]\n{chunk_text}"
            for index, chunk_text in enumerate(chunk_texts)
        )
        user_content = f"Sources:\n{numbered}\n\nQuestion: {question}"
    else:
        user_content = f"No sources were retrieved.\n\nQuestion: {question}"

    return [
        {"role": "system", "content": _SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]


class AnswerGenerator:
    """Loads Qwen/Qwen3-0.6B once and answers questions from context."""

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL_NAME,
        max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS,
        max_input_tokens: int = DEFAULT_MAX_INPUT_TOKENS,
    ) -> None:
        """Load the tokenizer and model.

        Args:
            model_name: Hugging Face model id to load.
            max_new_tokens: Maximum number of tokens to generate.
            max_input_tokens: Maximum prompt length. The tokenizer truncates
                longer prompts to this configured input limit.
        """
        if max_new_tokens <= 0:
            raise ValueError("max_new_tokens must be greater than zero.")

        if max_input_tokens <= 0:
            raise ValueError("max_input_tokens must be greater than zero.")

        # tokenizer 负责在文本和模型能处理的 token ID 之间转换。
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        # model 根据已有 token 反复预测下一个 token，从而生成回答。
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype="auto",
        )
        # eval() 是模型的推理模式，不是 CLI 的 evaluate/官方评分。
        # cast 只向 mypy 说明 eval 是无参数方法，不改变实际运行结果。
        cast(Callable[[], object], self.model.eval)()

        context_window = getattr(
            self.model.config,
            "max_position_embeddings",
            None,
        )
        if (
            isinstance(context_window, int)
            and max_input_tokens + max_new_tokens > context_window
        ):
            raise ValueError(
                "Input and output token limits exceed the model "
                "context window."
            )

        self.max_new_tokens = max_new_tokens
        self.max_input_tokens = max_input_tokens

    def generate(self, question: str, chunk_texts: list[str]) -> str:
        """Generate a grounded answer for one question.

        Args:
            question: The user's question.
            chunk_texts: Text from retrieved chunks, most relevant first.

        Returns:
            The generated answer text.
        """
        if not question.strip():
            raise ValueError("question must not be empty.")

        # messages：包含系统规则、检索到的 chunk 文本和用户问题。
        messages = build_prompt(question, chunk_texts)
        # prompt：按 Qwen 的聊天模板合并成的一段文本。
        prompt = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        # inputs 是 tokenizer 返回的字典式对象，其中 input_ids 由
        # prompt 转换而来，attention_mask 标记哪些位置是有效 token。
        # 例如：input_ids=[[10, 20, 30, 40]]，attention_mask=[[1, 1, 1, 1]]。
        # 两者的 shape 都是 [1, 4]：1 是 batch size，4 是 token 数。
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_input_tokens,
        )
        # **inputs 等价于分别传入 input_ids 和 attention_mask。
        # output_ids 包含原输入 token ID 和模型新生成的 token ID。
        # transformers 的 generate() 用了一个自绑定的泛型 self 类型，
        # AutoModelForCausalLM.from_pretrained() 的推断返回类型无法
        # 精确匹配它，这是上游的类型标注问题，不是真正的类型错误
        # （同样的原因见上面 self.model.eval() 处的 cast）。
        output_ids = self.model.generate(  # type: ignore[misc]
            **inputs,
            max_new_tokens=self.max_new_tokens,
            do_sample=False,
        )
        # 去掉前面的原输入，只保留新生成的回答 token ID。
        generated_ids = output_ids[0][inputs["input_ids"].shape[-1]:]
        answer = cast(
            str,
            self.tokenizer.decode(
                generated_ids, skip_special_tokens=True
            ),
        )
        return answer.strip()
