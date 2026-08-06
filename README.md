*This project has been created as part of the 42 curriculum by wehan.*

# RAG against the machine

## Description

A **RAG** (Retrieval-Augmented Generation) system that answers questions
about the [vLLM] codebase.

| Stage | What it does |
|---|---|
| **Index** | split vLLM source into chunks, build a BM25 index |
| **Retrieve** | find the top-k chunks for a question |
| **Generate** | answer with `Qwen/Qwen3-0.6B`, grounded in retrieved chunks |
| **Evaluate** | score retrieval with recall@k |

## Instructions

### Install

```bash
make install
```

### If there is not enough disk space

```bash
mkdir -p /goinfre/$USER/huggingface
export HF_HOME=/goinfre/$USER/huggingface
```

### Run

```bash
uv run python -m src index --max_chunk_size 2000
uv run python -m src search \
  "What HTTP endpoint is used to dynamically load a LoRA adapter in vLLM?" \
  --k 5
uv run python -m src search_dataset \
  --dataset_path data/datasets/UnansweredQuestions/dataset_docs_public.json \
  --save_directory data/output/search_results/UnansweredQuestions \
  --k 10
uv run python -m src answer \
  "What HTTP endpoint is used to dynamically load a LoRA adapter in vLLM?" \
  --k 5
uv run python -m src answer_dataset \
  --student_search_results_path data/output/search_results/UnansweredQuestions/dataset_docs_public.json \
  --save_directory data/output/search_results_and_answer/UnansweredQuestions
uv run python -m src evaluate \
  --student_search_results_path data/output/search_results/UnansweredQuestions/dataset_docs_public.json \
  --dataset_path data/datasets/AnsweredQuestions/dataset_docs_public.json
```

Single-query `search` and `answer` print structured JSON to stdout. Batch
commands write the same Pydantic-based structures to the requested directory.

Generating answers for a full dataset can be slow on CPU. For a quick
three-question smoke test, create a smaller search-results file with `jq` and
pass it to `answer_dataset`:

```bash
jq '.search_results = .search_results[:3]' \
  data/output/search_results/UnansweredQuestions/dataset_docs_public.json \
  > /tmp/search_results_3.json

uv run python -m src answer_dataset \
  --student_search_results_path /tmp/search_results_3.json \
  --save_directory data/output/search_results_and_answer/Test
```

This only limits the local smoke test; the original 100-question search-results
file remains unchanged.

`make install` / `make run` / `make debug` / `make clean` / `make lint` /
`make lint-strict` also available.

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

**Moulinette's own output is always the authoritative score** — `src`
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

AI was used for code generation assistance, debugging, and documentation throughout the project.

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
It was chosen over TF-IDF because its term-frequency saturation and document-length
normalization rank unevenly sized code and documentation chunks more fairly, while
remaining fast and explainable on CPU.

| Piece | Choice | Why |
|---|---|---|
| Tokenizer | lowercase + `[A-Za-z0-9_]+` + stopword removal | same rule for query and corpus, required for BM25 matching |
| Indexed text | `tokenize(file_path) + tokenize(chunk.text)` | questions often name a file/module |
| `k1` | **1.0** (default 1.5) | makes term frequency saturate sooner, so repeated tokens add less score |
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
| Keeping project checks focused | configured lint, type checking, and tests to exclude the virtual environment, generated data, and the vendored vLLM corpus |
| Handling invalid input and I/O failures | validated CLI arguments and converted missing files, malformed JSON, invalid paths, and empty queries into concise errors instead of tracebacks |
| Handling external and persisted-state failures | caught stale index, model-loading, and generation errors and returned actionable recovery messages |

## Example usage

```
$ uv run python -m src index --max_chunk_size 2000
Ingestion complete! Indexed 35327 chunks under data/processed/

$ uv run python -m src search \
    "What HTTP endpoint dynamically loads a LoRA adapter?" -k 1
{
  "search_results": [{
    "question_id": "generated UUID",
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
    "question_id": "generated UUID",
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
    --student_search_results_path data/output/search_results/UnansweredQuestions/dataset_docs_public.json \
    --dataset_path data/datasets/AnsweredQuestions/dataset_docs_public.json
Evaluation Results
========================================
Recall@1: 0.650 (65.0%)
Recall@3: 0.770 (77.0%)
Recall@5: 0.860 (86.0%)
Recall@10: 0.900 (90.0%)
```

## Edge-case checks

Create the deliberately invalid input files used below:

```bash
printf '{"rag_questions": [' > /tmp/bad_dataset.json
printf '{"rag_questions": [{}]}\n' > /tmp/partial_dataset.json
printf '\xff\xfe' > /tmp/non_utf8_dataset.json
printf '{"search_results": [], "k": 10}\n' > /tmp/empty_search_results.json
mkdir -p /tmp/rag-stale-index
printf 'not a pickle index' > /tmp/rag-stale-index/bm25_index.pkl
printf '{"search_results":[{"question_id":"blank-question","question":"","retrieved_sources":[]}],"k":3}\n' > /tmp/blank_question_results.json
```

Each command should finish without an unhandled Python traceback:

| Case | Command | Expected result |
|---|---|---|
| Empty query | `uv run python -m src search "" --k 5` | `Error: query must not be empty.` |
| `k=0` | `uv run python -m src search "query" --k 0` | valid JSON with no retrieved sources |
| Negative `k` | `uv run python -m src search "query" --k -3` | `Error: k must not be negative.` |
| Unknown query | `uv run python -m src search "zzzz_nonexistent_token" --k 5` | valid JSON with no arbitrary results |
| Missing dataset | `uv run python -m src search_dataset --dataset_path /tmp/does-not-exist.json --save_directory /tmp/rag-missing-check --k 10` | concise file-not-found error |
| Dataset path is a directory | `uv run python -m src search_dataset --dataset_path data/datasets --save_directory /tmp/rag-directory-check --k 10` | concise expected-file error |
| Malformed JSON | `uv run python -m src search_dataset --dataset_path /tmp/bad_dataset.json --save_directory /tmp/rag-bad-json-check --k 10` | dataset validation error |
| Missing JSON fields | `uv run python -m src search_dataset --dataset_path /tmp/partial_dataset.json --save_directory /tmp/rag-partial-check --k 10` | Pydantic validation error |
| Non-UTF-8 dataset | `uv run python -m src search_dataset --dataset_path /tmp/non_utf8_dataset.json --save_directory /tmp/rag-encoding-check --k 10` | UTF-8 error |
| Missing corpus | `uv run python -m src index --raw_directory /tmp/does-not-exist-corpus --index_directory /tmp/rag-index-check` | concise corpus-not-found error |
| Missing index | `uv run python -m src search "test" --index_directory /tmp/does-not-exist-index` | error instructing the user to build the index |
| Corrupt index | `uv run python -m src search "test" --index_directory /tmp/rag-stale-index` | error instructing the user to rebuild the index |
| Empty answer query | `uv run python -m src answer "" --k 3` | `Error: query must not be empty.` |
| Missing answer input | `uv run python -m src answer_dataset --student_search_results_path /tmp/does-not-exist.json --save_directory /tmp/rag-answer-check` | concise file-not-found error |
| Blank question in a batch | `uv run python -m src answer_dataset --student_search_results_path /tmp/blank_question_results.json --save_directory /tmp/rag-blank-answer-check` | that row receives an error answer without aborting the batch |
| Empty evaluation input | `uv run python -m src evaluate --student_search_results_path /tmp/empty_search_results.json --dataset_path data/datasets/AnsweredQuestions/dataset_docs_public.json` | zero recall at every reported cutoff |
