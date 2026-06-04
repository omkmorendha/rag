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

```
ingest.py    load → parse → chunk → embed → index → persist to vectorstore/
query.py     load index → transform query → retrieve → rerank → build prompt → generate
evaluate.py  run golden set through the query path, score each component separately
```

## Layout

```
data/                  source docs (gitignored)
vectorstore/           persisted indexes (gitignored)
eval/golden.jsonl      query → expected_answer / expected_chunk_ids
rag/                   pipeline package (stage interfaces + strategy implementations)
ingest.py              offline ingestion entry point
query.py               online query entry point
evaluate.py            offline evaluation harness
config.yaml            selects which strategy per stage
```

## Getting started

> Scaffolding stage — entry points and stage implementations are still being built out.
> See the build order in [`ARCHITECTURE.md`](./ARCHITECTURE.md#6-suggested-build-order).

```bash
# 1. build the Markdown corpus
uv run python scripts/prepare_data.py

# 2. chunk the corpus using config.yaml
uv run python scripts/chunk_corpus.py --write

# 3. encode chunks using config.yaml
uv run python scripts/embed_chunks.py --write

# 4. build + persist the vector index using config.yaml
uv run python scripts/index_corpus.py --write

# 5. query
python query.py "your question here"

# 6. measure
python evaluate.py
```

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

## Status

Early scaffolding. Building the first vertical slice (Markdown → recursive chunk → local
embed → FAISS flat → dense retrieve → grounded generate), then layering in alternative
strategies behind eval.
