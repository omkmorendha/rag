# RAG Harness — Architecture

A **local-first experiment harness** for RAG over Markdown + PDF documents. The goal is
not "a RAG that works" but "a frame where every stage is a swappable strategy, so you can
change one component, re-run eval, and measure the difference." Optimize for swappability
and measurement, not raw performance.

Stack: local embeddings (sentence-transformers), local vector store (FAISS), local
cross-encoder reranker. Only the final generation call hits an external API.

---

## 0. Two pipelines, not one

Ingestion and querying run at **different times and frequencies**:

- **Ingestion** is offline, runs once per corpus change. Output: a persisted index on disk.
- **Query** is online, runs per user question. Input: the loaded index.

If they share a function you re-parse and re-embed the entire corpus on every query.
So there are **two entry points**, and the boundary between them is *the persisted index*.

```
ingest.py :  load → parse → chunk → embed → index → PERSIST to vectorstore/
query.py  :  load index → transform query → retrieve → rerank → build prompt → generate
evaluate.py: run golden set through query path, score each component separately
```

---

## 1. The type that threads everything

The single most important design decision for swappability: **every stage speaks `Chunk`.**
A chunk must carry its `text` and `metadata` all the way to the prompt — not just a vector
ID. If stages pass bare strings or bare vectors, you can't swap them independently.

```python
@dataclass
class Document:
    id: str
    text: str          # clean, parser-normalized text
    metadata: dict     # source path, doc type, title, created_at

@dataclass
class Chunk:
    id: str
    text: str                       # survives all the way into the prompt
    metadata: dict                  # source, page, section, timestamp, parent_id
    embedding: list[float] | None = None
    score: float | None = None      # set by retriever, OVERWRITTEN by reranker
```

`metadata` is append-only and generous: "you cannot filter on what you don't have." Store
source, page/section, timestamp, and `parent_id` (for decoupled retrieval, see §4).

---

## 2. Strategy pattern + registry

Each stage is an **interface** (one method). Each technique is an **implementation**.
`config.yaml` names which implementation to use per stage; `registry.py` maps name →
implementation; the entry points just wire them. This is what makes "choose technique via
params" real.

```yaml
# config.yaml
embedder:  { name: local, model: BAAI/bge-small-en-v1.5 }   # one model, shared at ingest+query
chunker:   { name: recursive, size: 512, overlap: 64 }
indexer:   { name: faiss_flat }
retriever: { name: hybrid_rrf, k: 50 }
reranker:  { name: cross_encoder, model: BAAI/bge-reranker-base, top_n: 8 }
query_transform: { name: passthrough }
generator: { name: anthropic, model: claude-sonnet-4-6 }
```

**Hard invariant:** the `embedder` block is read by *both* ingest and query. If the corpus
is embedded with model A and the query with model B, retrieval silently returns garbage —
same vector space is the entire point of a bi-encoder. The config makes the embedder a
single source of truth.

---

## 3. Stage interfaces

### load_files() → list[raw file handles]
Walk `data/`, dispatch by extension. Returns paths/bytes, not parsed content.

### parser: parse(raw) → Document
One parser per format (`md`, `pdf`). **This is where quality is won or lost** — garbage in,
garbage out. People skip straight to chunking and then blame the chunker for the parser's
mess.
- **md:** mostly clean already; preserve heading structure (the recursive chunker needs it).
- **pdf:** the hard case. `pymupdf` for text-first PDFs; reach for `unstructured` or a
  vision model when tables/columns/scans appear. Capture `page` into metadata.

### chunker: chunk(doc) → list[Chunk]
The dual mandate: small enough for a precise embedding signal, large enough to carry usable
context.
- `fixed` — N tokens, fixed overlap. Truncates sentences; baseline only.
- `recursive` — split on document structure (heading → paragraph → sentence). Default for
  Markdown.
- `semantic` — split where embedding similarity drops. Best quality, most compute.
- See §4 for sentence-window / parent-document (decoupling retrieval unit from output unit).

### embedder: embed(texts) → list[vector]
Bi-encoder, encodes query and documents **independently** (that's what makes search scale,
and what makes it lossy). Used in BOTH pipelines:
- ingest: embed every chunk.
- query: **embed the query with the same embedder** — make this explicit, either the
  retriever embeds internally or you pass `query_embedding`.

Config checklist: dimension (expressive vs storage/speed), max sequence length (**must
exceed max chunk size or you truncate silently**), domain fit, multilingual.

### indexer: build(chunks) → Index ; Index.search(query_vec, k) → list[Chunk]
Persisted to `vectorstore/`. `faiss_flat` (brute force, exact, baseline), `faiss_hnsw`
(graph ANN). Quantization is index-level, add later. **Note:** hybrid retrieval needs *two*
indexes — the dense vector index AND a sparse BM25 index.

### query_transform: transform(query) → query | list[query]
Runs *before* retrieval to fight query-document asymmetry.
- `passthrough` — default.
- `step_back` — generalize to a broader question.
- `hyde` — generate a hypothetical answer, retrieve against *that*.
- `decomposition` — split a multi-part query, retrieve per sub-query, merge.

### retriever: retrieve(query, k) → list[Chunk]
Over-retrieve here (k≈50); the reranker compresses later.
- `dense` — semantic, via the vector index.
- `bm25` — sparse/lexical, via the BM25 index.
- `hybrid_rrf` — run dense AND bm25, fuse with Reciprocal Rank Fusion (the modern default).
  This means **two retrieval paths**, not one. Build dense first, then bm25, then fuse.

### reranker: rerank(query, chunks) → list[Chunk]
A **cross-encoder** over `(query, chunk.text)` pairs you already hold — it re-scores text in
hand and never touches the index. Signature: `rerank(query, retrieved_chunks)`.
What it buys: compresses 50 mediocre → 8 high-quality, drops duplicates (diversity), and
lets you **reorder for lost-in-the-middle** — best chunks at the start and end of the prompt
where the model actually attends.
- `noop` — passthrough (for the first vertical slice).
- `cross_encoder` — local reranker model.

### generator: build_prompt(query, chunks) → prompt ; generate(prompt) → answer
Split into two:
1. **build_prompt** — assemble instructions + chunks with stable **citation identifiers**
   and key metadata, ordered for lost-in-the-middle (don't bury the best chunk in the
   middle).
2. **grounding** — instruct: answer *only* from retrieved text, require citations, and
   explicitly say "I don't know" when the answer isn't in the chunks. This is the dial
   between stale-training-data answers and unhelpful over-refusal.

---

## 4. Decoupling retrieval unit from output unit

Changes the data model, so it gets its own note. You retrieve on a *small* unit (a sentence
or small window, precise embedding) but feed the LLM a *larger* unit (the parent
paragraph/section, full context). Implement via `parent_id` on the small chunk's metadata
plus a parent store; after retrieval+rerank, swap each small chunk for its parent before
`build_prompt`. Keep this out of the first slice; add once the baseline works.

---

## 5. Evaluation

The key principle: **score components separately.** Evaluation needs:

1. **A golden dataset** (`eval/golden.jsonl`): each row is
   `{query, expected_answer, expected_chunk_ids}`. Built by hand; it's the foundation, and
   it also tells you what embedder/chunker to pick.
2. **Per-component scoring:**
   - **Retrieval:** recall@k / MRR — are the expected chunk ids in the retrieved set?
   - **Rerank:** precision@n — did the right chunks rise to the top?
   - **Generation:** faithfulness (is the answer grounded in the chunks?) and answer
     relevance — typically LLM-as-judge against `expected_answer`.

Evaluation is its own offline harness (`evaluate.py`), not a step inside the query path.
The payoff of the whole strategy/registry design: change `chunker` in config, re-run
`evaluate.py`, read the recall delta. That loop is what this repo is *for*.

---

## 6. Suggested build order

1. **Vertical slice (noop where possible):** md loader → recursive chunk → local embed →
   faiss_flat → dense retrieve → noop rerank → grounded generate. Persist the index. Prove
   end-to-end.
2. **Eval harness + a tiny golden set (~10 queries).** Now you can measure everything below.
3. **PDF parser.** The quality cliff; do it once you can measure the gain.
4. **cross_encoder reranker.** Usually the biggest quality jump per line of code.
5. **hybrid_rrf retriever** (add bm25 index + fusion).
6. **Alternative chunkers** (semantic), **query transforms** (hyde/step_back),
   **decoupled parent-document**, **quantization** — each gated behind "did eval improve?"

---

## Layout

```
rag/
├── data/                  # source docs (gitignored)
├── vectorstore/           # persisted indexes (gitignored)
├── eval/golden.jsonl      # query → expected_answer / expected_chunk_ids
├── rag/
│   ├── types.py           # Document, Chunk
│   ├── config.py          # load config.yaml
│   ├── registry.py        # name → implementation (the strategy switch)
│   ├── loaders/  parsers/  chunkers/  embedders/  indexers/
│   ├── retrievers/  rerankers/  query_transform/
│   └── generator.py
├── ingest.py    # offline
├── query.py     # online
├── evaluate.py  # offline
├── config.yaml
└── pyproject.toml
```
