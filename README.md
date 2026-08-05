*This project has been created as part of the 42 curriculum by wehan.*

# RAG against the machine

## Description

A **RAG** (Retrieval-Augmented Generation) system that answers questions
about the [vLLM](https://github.com/vllm-project/vllm) 0.10.1 codebase.

| Stage | What it does |
|---|---|
| **Index** | split vLLM source into chunks, build a BM25 index |
| **Retrieve** | find the top-k chunks for a question |
| **Generate** | answer with `Qwen/Qwen3-0.6B`, grounded in retrieved chunks |
| **Evaluate** | score retrieval with recall@k |

## Instructions

### Install

```bash
uv sync
```

### Disk space

| Component | Size |
|---|---|
| `.venv` (incl. CPU-only PyTorch) | ~1.0 GB |
| `data/raw/` (vLLM corpus) | ~40 MB |
| `data/processed/` (BM25 index) | ~40 MB |
| `Qwen/Qwen3-0.6B` weights | ~1.5 GB |
| **Total** | **~2.6 GB** — keep 4–5 GB free |

**2 things that prevent disk errors** (already set in `pyproject.toml`):

- `torch` → pinned to the **CPU-only wheel index**. Default `torch` pulls
  the full CUDA/cuDNN/NCCL stack (extra GBs), never used here.
- **Small `/home`, big scratch disk** (e.g. 42 machines: tiny `/home`,
  huge `/goinfre`) → point the model cache at the big one *before* the
  first `answer` run:
  ```bash
  export HF_HOME=/goinfre/$USER/huggingface
  ```

### Run

```bash
uv run python -m src index --max_chunk_size 2000
uv run python -m src search "<question>" -k 5
uv run python -m src search_dataset --dataset_path <path> --save_directory <dir> -k 10
uv run python -m src answer "<question>" -k 5
uv run python -m src answer_dataset --student_search_results_path <path> --save_directory <dir>
uv run python -m src evaluate --student_search_results_path <path> --dataset_path <path>
```

`make run` / `make debug` / `make clean` / `make lint` / `make lint-strict`
also available.

## Resources

### Learning materials

| Resource | Brief note |
|---|---|
| [RAG course (Chinese)](https://www.bilibili.com/video/BV1hMMuzEEVe/?p=11) | RAG workflow, retrieval, generation, and evaluation. |
| [Hugging Face course (Chinese)](https://www.bilibili.com/video/BV1u3QzY8EyL) | Models, `transformers`, tokenizers, and fine-tuning. |
| [Qwen3-0.6B model card](https://huggingface.co/Qwen/Qwen3-0.6B) | Official model usage and generation settings. |
| [What is a token? (Chinese)](https://www.bilibili.com/video/BV1S5miBvEsu/) | Tokens, tokenizers, BPE, and context windows. |

### Technical references

- [BM25 overview](https://en.wikipedia.org/wiki/Okapi_BM25) and the
  [`rank_bm25` implementation](https://github.com/dorianbrown/rank_bm25)
- [Hugging Face `transformers` documentation](https://huggingface.co/docs/transformers)
- [Python `ast` module](https://docs.python.org/3/library/ast.html)
- [Pydantic documentation](https://docs.pydantic.dev/)
- [Python Fire](https://github.com/google/python-fire)
- [`uv` guide for PyTorch](https://docs.astral.sh/uv/guides/integration/pytorch/)

**AI usage:**

| Task | How AI helped |
|---|---|
| Design review | chunking strategy, BM25 `k1`/`b` tuning, recall@k / IoU logic |
| Debugging | found + fixed a runtime crash (missing `torch` dependency) and a disk-space issue (GPU wheel) |
| Drafting | docstrings/comments and this README — reviewed and verified, not copied blind |

Every AI-suggested change was checked with `flake8`, `mypy --strict`, and a
real run before being kept.

## System architecture

```
data/raw/  →  documents.py   →  chunking.py    →  index.py (BM25Okapi)
(corpus)      load_documents()  chunk_documents() build/save/load index
                                                          │
                                                          ▼
                                              index.py: search(query, k)
                                                          │
                                                          ▼
                                        generation.py: AnswerGenerator
                                            (Qwen/Qwen3-0.6B)
                                                          │
                                                          ▼
                    cli.py: RagCLI (Python Fire) ── index / search /
                    search_dataset / answer / answer_dataset / evaluate
```

| Layer | Type | Why |
|---|---|---|
| `Document`, `Chunk`, `MinimalSource`, `StudentSearchResults`, ... | **pydantic** | cross-stage data, needs validation |
| `BM25Index`, `AnswerGenerator` | **plain class** | service objects, not exchanged as data (allowed by subject) |

## Chunking strategy

| File type | Method | Fallback |
|---|---|---|
| `.py` / `.pyi` | **AST split** — one chunk per top-level statement | parse fails / no `end_lineno` → fixed-size split |
| `.md` / `.rst` / `.txt` | **paragraph split** — pack paragraphs up to `max_chunk_size` | oversized paragraph → fixed-size split |

- Offsets use `[first, last)` — same slice rule as `MinimalSource`.
- Default `--max_chunk_size = 2000` chars (matches moulinette's limit).

## Retrieval method

**BM25** (`rank_bm25.BM25Okapi`) — one of the two lexical methods required.

| Piece | Choice | Why |
|---|---|---|
| Tokenizer | lowercase + `[A-Za-z0-9_]+` + stopword removal | same rule for query and corpus, required for BM25 matching |
| Indexed text | `tokenize(file_path) + tokenize(chunk.text)` | questions often name a file/module |
| `k1` | **1.0** (default 1.5) | slows token-frequency saturation |
| `b` | **0.4** (default 0.75) | less penalty for long code chunks |
| Ranking | `bm25.get_scores()`, sort desc, take top-k | — |

## Performance analysis

| Metric | Requirement | Measured |
|---|---|---|
| Indexing (35,327 chunks) | ≤ 5 min | **~3 s** |
| Retrieval (100 questions) | ≤ 90 s / 200 q | **~4.3 s / 100 q** |
| Recall@5 — docs | ≥ 80% | **86.0%** ✅ |
| Recall@5 — code | ≥ 50% | **59.6%** ✅ |

| Dataset | R@1 | R@3 | R@5 | R@10 |
|---|---|---|---|---|
| Docs | 65.0% | 77.0% | 86.0% | 90.0% |
| Code | 35.4% | 55.6% | 59.6% | 68.7% |

Answer generation (CPU): **~40–50 s/question** — not covered by the
throughput requirement (that's retrieval-only), but budget time for it.

## Design decisions

| Decision | Reason |
|---|---|
| Service classes stay plain Python | not serialized data, subject allows it |
| Index saved as **pickle**, not JSON | `BM25Okapi` has no JSON form |
| `evaluate` re-implements moulinette's recall@k locally | fast local iteration; never calls the moulinette |
| `answer_dataset` re-reads chunk text from disk | its input only has `MinimalSource` locations, not text |

## Challenges faced

| Challenge | Fix |
|---|---|
| Long code chunks scored worse than short docs chunks | tuned BM25 `k1`/`b` against measured recall@5 |
| AST gives line numbers, chunks need character offsets | built a line→char offset table |
| **Silent crash**: `flake8`/`mypy --strict` passed, `answer` crashed at runtime | `torch` was missing from `pyproject.toml`; added it, re-verified with a live run |
| Disk full during install | default `torch` pulls CUDA stack → pinned to CPU-only wheel index |

## Example usage

```
$ uv run python -m src index --max_chunk_size 2000
Ingestion complete! Indexed 35327 chunks under data/processed/

$ uv run python -m src search "How to configure the OpenAI server?" -k 3
data/raw/vllm-0.10.1/docs/serving/openai_compatible_server.md [9867:10100]
data/raw/vllm-0.10.1/vllm/entrypoints/openai/api_server.py [267:400]
data/raw/vllm-0.10.1/examples/online_serving/openai_chat_completion_client_with_tools.py [0:2000]

$ uv run python -m src answer "How do I configure the OpenAI compatible server?" -k 3
The OpenAI compatible server can be configured by starting the vLLM
server with `vllm serve <model>` and passing standard OpenAI-compatible
options such as --port and --api-key...

$ uv run python -m src evaluate \
    --student_search_results_path data/output/search_results/AnsweredQuestions/dataset_docs_public.json \
    --dataset_path data/datasets/AnsweredQuestions/dataset_docs_public.json
Evaluation Results
========================================
Recall@1: 0.650 (65.0%)
Recall@3: 0.770 (77.0%)
Recall@5: 0.860 (86.0%)
Recall@10: 0.900 (90.0%)
```
