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

Single-query `search` and `answer` print structured JSON to stdout. Batch
commands write the same Pydantic-based structures to the requested directory.

`make run` / `make debug` / `make clean` / `make lint` / `make lint-strict` /
`make test` also available.

### Moulinette (official grading — do NOT commit it)

The moulinette is **not part of this repository**. Download it yourself
from the school intranet and run it locally; `git status` should never
show a `moulinette*` file as tracked.

| Step | Command |
|---|---|
| 1. Unzip, make executable | `chmod +x moulinette-ubuntu` (or `-fedora`), then `mv moulinette-ubuntu moulinette` |
| 2. Build the index | `uv run python -m src index --max_chunk_size 2000` |
| 3. Search a dataset | `uv run python -m src search_dataset --dataset_path data/datasets/UnansweredQuestions/dataset_docs_public.json --save_directory data/output/search_results/UnansweredQuestions --k 10` |
| 4. Score with the moulinette | `./moulinette evaluate_student_search_results data/output/search_results/UnansweredQuestions/dataset_docs_public.json data/datasets/AnsweredQuestions/dataset_docs_public.json --k 10 --max_context_length 2000` |

Repeat step 3–4 for the `dataset_code_public.json` dataset. This project's
own `evaluate` command (step 3 above's output, checked with `src evaluate`)
mirrors the moulinette's recall@k math for fast local iteration, but the
**moulinette's own output is always the authoritative score** — `src`
never imports or calls it.

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
| Choosing a retrieval method | chose BM25 over other lexical methods for fast, explainable matching of code identifiers and documentation terms |
| Splitting code and documentation accurately | used AST chunks for Python, paragraph chunks for text, and preserved exact character offsets |
| Ranking code and documentation fairly | tuned BM25 `k1`/`b` against measured recall@5 |
| Grounding answers in retrieved code | built prompts from top-k chunks and instructed Qwen not to guess beyond them |
| Running generation on limited hardware | used Qwen3-0.6B, bounded token lengths, and CPU-only PyTorch |
| `make lint`/`lint-strict` crashed when run as `flake8 .`/`mypy .` (subject's literal command) | they walked into `.venv/` and the vendored vLLM corpus; added `.flake8` and `[tool.mypy] exclude` so `.` only lints/type-checks this project's own code |
| `make test` crashed | `pytest` collected vLLM's own broken test suite from `data/raw/`; scoped it to `testpaths = ["tests"]` and added `pythonpath = ["."]` so `from src...` imports resolve |
| `search`/`answer` crashed on a stale index after refactoring `src/models/` → `src/models.py` | `pickle.load()` raised an uncaught `ModuleNotFoundError`; `load_index()` now catches unpickling failures and reports a clear "rebuild the index" error instead |
| `answer ""` crashed with an uncaught `ValueError` | added an explicit empty-query check before loading the model |
| One blank question in an `answer_dataset` batch would crash the whole run | that row now gets an error string as its answer instead of aborting every other question |

## Example usage

```
$ uv run python -m src index --max_chunk_size 2000
Ingestion complete! Indexed 35327 chunks under data/processed/

$ uv run python -m src search \
    "What HTTP endpoint dynamically loads a LoRA adapter?" -k 1
{
  "search_results": [{
    "question_id": "<generated UUID>",
    "question": "What HTTP endpoint dynamically loads a LoRA adapter?",
    "retrieved_sources": [{
      "file_path": "data/raw/vllm-0.10.1/docs/features/lora.md",
      "first_character_index": 3835,
      "last_character_index": 5714
    }]
  }],
  "k": 1
}

$ uv run python -m src answer \
    "What HTTP endpoint dynamically loads a LoRA adapter?" -k 1
{
  "search_results": [{
    "question_id": "<generated UUID>",
    "question": "What HTTP endpoint dynamically loads a LoRA adapter?",
    "retrieved_sources": [{
      "file_path": "data/raw/vllm-0.10.1/docs/features/lora.md",
      "first_character_index": 3835,
      "last_character_index": 5714
    }],
    "answer": "The HTTP endpoint dynamically loading a LoRA adapter is `/v1/load_lora_adapter`."
  }],
  "k": 1
}

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

## Testing

### Unit tests

```bash
make test    # runs tests/test_smoke.py — tokenizer, chunk offsets,
              # read_source_text bounds, IoU/recall matching
```

Not graded (subject: test programs are for your own verification), but
kept fast and dependency-light so it always runs in a couple of seconds.

### Edge cases — every row below was actually run against this code,
### not assumed. **Every command must exit without a Python traceback.**

| # | Case | Command | Expected | Verified result |
|---|---|---|---|---|
| 1 | Empty query | `search "" -k 5` | no crash | ✅ valid JSON with `retrieved_sources: []` |
| 2 | `k=0` | `search "query" -k 0` | no crash | ✅ valid JSON with `retrieved_sources: []` |
| 3 | Negative `k` | `search "query" -k -3` | no crash | ✅ valid JSON with `retrieved_sources: []` |
| 4 | Query has no searchable match | `search "zzzz_nonexistent_token" -k 5` | no arbitrary results | ✅ valid JSON with `retrieved_sources: []` |
| 5 | Dataset file missing | `search_dataset --dataset_path <missing>` | no crash | ✅ `Error: file not found: ...` |
| 6 | Path is a directory, not a file | `search_dataset --dataset_path data/datasets` | no crash | ✅ `Error: expected a file, got directory: ...` |
| 7 | Malformed JSON | `search_dataset --dataset_path <bad.json>` | no crash | ✅ `Error: invalid RAG dataset: ...` |
| 8 | Valid JSON, missing required fields | `search_dataset --dataset_path <partial.json>` | no crash | ✅ pydantic validation error, formatted |
| 9 | Non-UTF-8 file | `search_dataset --dataset_path <binary file>` | no crash | ✅ `Error: dataset is not valid UTF-8: ...` |
| 10 | Corpus directory missing | `index --raw_directory <missing>` | no crash | ✅ `Error: Corpus directory does not exist: ...` |
| 11 | `search`/`answer` before `index` was ever run | `search "test"` (no `data/processed/`) | no crash | ✅ `Error: No index found at ...; run the index command first.` |
| 12 | Stale/incompatible index file (e.g. after refactoring `src`) | `search "test"` | no crash | ✅ `Error: Index at ... could not be loaded (it may be stale or corrupted...); run the index command again.` — **found and fixed during this review**, see Challenges below |
| 13 | Empty query to `answer` | `answer "" -k 3` | no crash | ✅ `Error: query must not be empty.` — **found and fixed during this review** |
| 14 | Zero chunks retrieved, non-empty query | `answer "the is of a" -k 3` | no crash | ✅ model answers from `"No sources were retrieved."` |
| 15 | `answer_dataset`, input file missing | `answer_dataset --student_search_results_path <missing>` | no crash | ✅ `Error: file not found: ...` |
| 16 | `answer_dataset`, one blank question in the batch | (dataset with an empty `question`) | no crash, batch still completes | ✅ that row gets `"Error: question must not be empty."` as its answer, rest of the batch still runs — **found and fixed during this review** |
| 17 | `evaluate` with an empty search-results file | `evaluate --student_search_results_path <empty>` | no crash | ✅ reports `Recall@k: 0.000 (0.0%)` for all k |

Three real crashes (#12, #13, #16) were found by actually running these
cases, not by inspection, and are now fixed and re-verified — see
**Challenges faced** below.
