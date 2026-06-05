# rag

A **local-first RAG experiment harness** over Markdown and PDF documents.

Every stage of the pipeline — parsing, chunking, embedding, indexing, retrieval, reranking,
generation — is a swappable strategy selected from `config.yaml`. Change one component,
re-run eval, measure the delta. Built to compare techniques, not just to answer questions.

See [`ARCHITECTURE.md`](./ARCHITECTURE.md) for the full design.

## Stack

Local-first: local embeddings (sentence-transformers), local vector store (FAISS), local
cross-encoder reranker. Only the final generation call hits an external API.

## Pipelines

Two pipelines run at different times (ARCHITECTURE.md §0); the persisted index in
`vectorstore/` is the boundary between them.

```
ingest   load → parse → chunk → embed → index → persist to vectorstore/
                 (the scripts/*.py chain below)
query    load index → transform query → retrieve → rerank → generate  (wired via rag/registry.py)
eval     run the golden set through the query path, score each stage separately
                 (evaluate.py)
```

## Layout

```
data/                  source docs + generated corpus (gitignored)
vectorstore/           persisted index: index.faiss + chunks.jsonl + meta.json (gitignored)
eval/golden.jsonl      query → expected_answer / expected_chunk_ids
rag/                   pipeline package — one subpackage per stage:
  loaders/ parsers/ chunkers/ embedders/ indexers/ retrievers/ rerankers/ generator/ eval/
  registry.py          name → implementation (the strategy switch)
  config.py  types.py
scripts/               ingest pipeline + corpus/golden derivation (run with `uv run`)
evaluate.py            offline evaluation harness
config.yaml            selects which strategy per stage
```

## Getting started

```bash
# 1. build the Markdown corpus from the rag-mini-wikipedia benchmark
uv run python scripts/prepare_data.py

# 2. chunk → 3. embed → 4. index (each reads config.yaml)
uv run python scripts/chunk_corpus.py --write
uv run python scripts/embed_chunks.py --write
uv run python scripts/index_corpus.py --write

# 5. derive the eval golden set (depends on the chunk ids from step 2)
uv run python scripts/derive_golden.py

# 6. measure — score retrieval / rerank / generation separately
uv run python evaluate.py              # deterministic + free (generation needs ANTHROPIC_API_KEY)
uv run python evaluate.py --no-generate  # retrieval + rerank only, no API
uv run python evaluate.py --judge        # add LLM-as-judge generation scoring
```

Generation and the `--judge` scorer call Claude — put `ANTHROPIC_API_KEY` in a `.env`
file (read automatically) or the environment.

## Chunking

`config.yaml` selects the active chunker:

```yaml
chunker:
  name: recursive
  size: 256
  overlap: 32
```

Available strategies:

- `fixed` — fixed whitespace-token windows; useful as a baseline.
- `recursive` — paragraph/sentence-aware windows with token fallback; default for Markdown.
- `sentence_window` — overlapping sentence groups for smaller, precise retrieval units.

## Embedding

`config.yaml` also selects the encoder shared by ingest and query:

```yaml
embedder:
  name: local
  model: BAAI/bge-small-en-v1.5
  batch_size: 32
  normalize_embeddings: true
```

Available strategies:

- `local` — sentence-transformers bi-encoder for production retrieval.
- `hashing` — deterministic lightweight vectors for tests and smoke runs.

## Indexing

`config.yaml` selects how embedded chunks are indexed and persisted to `vectorstore/`:

```yaml
indexer:
  name: faiss_flat
```

Available strategies:

- `faiss_flat` — exact (brute-force) inner-product search; baseline.

The index records the embedder it was built with; loading it under a different embedder
raises `IndexMismatchError`, enforcing the shared-vector-space invariant (ingest and query
must use the same encoder).

## Retrieval

`config.yaml` selects how a query is turned into candidate chunks. Over-retrieve here (k≈50);
the reranker compresses the candidates afterwards.

```yaml
retriever:
  name: dense
  k: 50
```

Available strategies:

- `dense` — semantic search over the vector index, embedding the query with the same
  encoder used at ingest; baseline.

## Query transforms

`config.yaml` optionally rewrites the *retrieval* query before it is embedded — the
generator still answers the user's **original** question, so a poor transform costs retrieval
quality but never correctness. Omit the block to default to the identity baseline.

```yaml
query_transform:
  name: rewrite
  model: claude-haiku-4-5
```

Available strategies:

- `passthrough` — identity; the default when no block is present.
- `rewrite` — Claude expands abbreviations, entities, and synonyms into one search query.
- `step_back` — Claude generalizes a specific question into a broader one to retrieve
  background.

The LLM transforms are best-effort: an empty response *or any provider/network error* falls
back to the original query, so a failed transform never aborts a sweep. The static system
prompt is prompt-cached. On this corpus's terse factoid queries both transforms *hurt*
ranking — see [Results](#results).

## Reranking

`config.yaml` selects how candidates are re-scored and compressed before generation. A
reranker re-scores `(query, chunk)` pairs it already holds and never touches the index.

```yaml
reranker:
  name: cross_encoder
  model: BAAI/bge-reranker-base
  top_n: 8
```

Available strategies:

- `noop` — pass candidates through unchanged (optional `top_n` cap); for the vertical slice.
- `cross_encoder` — local cross-encoder that scores each pair jointly and keeps the top
  `top_n`; usually the biggest quality jump per line of code.

## Generation

`config.yaml` selects the final stage: a grounded, cited answer built from the reranked
chunks. The generator answers *only* from the chunks, cites the source passage of each
fact, and says "I don't know" when the answer isn't present — treating chunk text as data,
never as instructions.

```yaml
generator:
  name: anthropic
  model: claude-haiku-4-5
  max_tokens: 1024
```

Available strategies:

- `anthropic` — Claude via the Messages API. Needs `ANTHROPIC_API_KEY` in the environment
  (a `.env` is read by the scripts). The static grounding prompt is prompt-cached; only the
  per-query chunks vary.

## Evaluation

`evaluate.py` runs the golden set (`eval/golden.jsonl`) through the query path and **scores
each stage separately** — the whole point of the swappable design: change a stage in
`config.yaml`, re-run, read the delta.

```bash
uv run python evaluate.py                      # deterministic, free, offline-ish
uv run python evaluate.py --no-generate        # retrieval + rerank only; no API at all
uv run python evaluate.py --judge              # add LLM-as-judge generation scoring
uv run python evaluate.py --json runs/base.json # dump full results for diffing
```

Metrics:

- **Retrieval** — recall@k, MRR (are the expected chunk ids retrieved, and how high?).
- **Rerank** — precision@n (how clean is the top-n after reranking?).
- **Generation** — `answer_correct` (deterministic substring check, always on) plus, behind
  `--judge`, LLM-as-judge **faithfulness** and **answer_relevance** (the only generation
  signal that survives paraphrase).

The golden set's `expected_chunk_ids` are **derived after ingest** and coupled to the active
chunker, so regenerate it with `scripts/derive_golden.py` whenever the chunker changes.

## Results

A full sweep of every technique available today lives in
[`docs/EXPERIMENT_REPORT.md`](./docs/EXPERIMENT_REPORT.md) (charts in `docs/figures/`,
machine-readable numbers in `docs/experiment_results.json`). Regenerate with
`uv run python scripts/run_experiments.py` then `scripts/plot_experiments.py`.

**Caveats first:** the corpus is `rag-mini-wikipedia` (3,200 short single-fact passages) and
the golden set is just **33 hand-verified queries**, each variant a **single run**. Treat
every number as directional. Retrieval/rerank metrics are deterministic and reproduce
exactly; generation/judge metrics (and the LLM transforms) vary run-to-run, with a measured
noise floor of **~0.09** on `answer_correct` — deltas at or below that are noise, not effects.

What the sweep establishes on *this* corpus:

- **Dense retrieval already does the heavy lifting** — recall@5 ≈ 0.74, MRR ≈ 0.58 from
  embeddings alone. The gold chunk is usually found and ranked high.
- **The cross-encoder reranker buys no measurable quality here** and costs ~0.9 s/query: it
  re-orders an identical candidate set, leaving recall/MRR untouched and nudging precision@n
  *down* (0.106 → 0.087). On short single-fact passages it has little to fix.
- **LLM query transforms (`rewrite`, `step_back`) hurt retrieval ranking** on terse factoid
  queries — MRR 0.58 → 0.48 (rewrite) / 0.41 (step_back) — and add ~1.3 s/query. They're
  built for verbose/conversational/multi-hop queries this golden set deliberately lacks.
- **`recursive` ≈ `fixed`** on this corpus (clean short passages give structure-aware
  splitting nothing to exploit); **`sentence_window`** trades retrieval recall for rerank
  precision (recall@5 0.74 → 0.54, p@n 0.087 → 0.136) — useful only paired with
  parent-document expansion, which this slice doesn't yet implement.

The point isn't that these techniques are bad — every negative is plausibly a
corpus/query-distribution artifact. It's that the harness *surfaces* which stage actually
moved the numbers here, which is exactly what per-stage scoring exists to do.

## Status

The full vertical slice and the eval harness are in (build order §6.1–6.2): Markdown →
recursive chunk → local embed → FAISS flat → dense retrieve → cross-encoder rerank →
grounded generate, plus `evaluate.py` scoring each stage. Every stage is swappable from
`config.yaml`.

Query transforms (`rewrite`, `step_back`) and the alternative chunkers are now in and
measured (see [Results](#results)). Next, each gated behind "did eval improve?"
(ARCHITECTURE.md §6): PDF parser, `hybrid_rrf` retrieval (dense + BM25), and decoupled
parent-document retrieval (to pair with `sentence_window`).
