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
# 1. drop documents into data/
# 2. ingest the corpus (build + persist the index)
python ingest.py

# 3. query
python query.py "your question here"

# 4. measure
python evaluate.py
```

## Status

Early scaffolding. Building the first vertical slice (Markdown → recursive chunk → local
embed → FAISS flat → dense retrieve → grounded generate), then layering in alternative
strategies behind eval.
