"""Answer generation using a local Hugging Face causal language model."""

from collections.abc import Callable
from typing import Final, cast

from transformers import AutoModelForCausalLM, AutoTokenizer

DEFAULT_MODEL_NAME: Final[str] = "Qwen/Qwen3-0.6B"
DEFAULT_MAX_NEW_TOKENS: Final[int] = 512
# Budget for the whole prompt in tokens; 6000 + 512 stays under the
# model's context window.
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

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype="auto",
        )
        # .eval() switches the model to inference mode (unrelated to the
        # CLI's `evaluate` command). The cast only tells mypy this is a
        # no-argument method; it doesn't change runtime behaviour.
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

        messages = build_prompt(question, chunk_texts)
        prompt = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_input_tokens,
        )

        output_ids = self.model.generate(  # type: ignore[misc]
            **inputs,
            max_new_tokens=self.max_new_tokens,
            do_sample=False,
        )
        # Drop the original input tokens, keep only the newly generated
        # answer tokens.
        generated_ids = output_ids[0][inputs["input_ids"].shape[-1]:]
        answer = cast(
            str,
            self.tokenizer.decode(
                generated_ids, skip_special_tokens=True
            ),
        )
        return answer.strip()
