# RAG technique comparison — what each stage actually buys

This report uses the harness's whole reason for existing (ARCHITECTURE.md §5): change one
stage in `config.yaml`, re-run, read the per-stage delta. It compares the techniques
available today across four experiment families — an **ablation ladder**, a **chunker**
sweep, a **reranker** sweep, and a **query-transform** sweep — scoring every stage
separately.

**The headline is uncomfortable and that is the point:** on this corpus, the two techniques
the architecture doc expected to help most — the cross-encoder reranker and query
transforms — *do not help, and mostly hurt*. The harness is doing its job: it tells you what
actually moved the numbers here, not what should in general.

## How to read this (and what NOT to over-read)

- **Corpus + golden set:** `rag-mini-wikipedia`, 3,200 short single-fact passages. Golden
  set = **33 hand-verified queries** (29 single-passage, 4 multi-passage). This is small.
  Treat every number as directional, not decisive.
- **The noise floor is large.** Four variants in this sweep run a *byte-identical* query
  pipeline (recursive chunk → cross-encoder rerank → passthrough → Haiku). Their
  `answer_correct` scores span **0.394–0.485 — a 0.091 spread that is pure generation +
  judge non-determinism**. So:
  > **Any `answer_correct` delta below ~0.09 is indistinguishable from noise.** Do not read
  > it as a technique effect. The tables below are annotated where this applies.
- **`answer_correct` is a brittle substring check** (`expected_answer.lower() in
  answer.lower()`), so it under-counts correct paraphrases. `faithfulness` and
  `answer_relevance` (LLM-judge) are the trustworthy generation signals.
- **`recall@k` (k=50) is saturated** at ~0.85 — over-retrieving 50 of 3,200 chunks almost
  always catches the gold one. **Read `recall@5` and `MRR` for retrieval deltas.**
- **Latency** is per-query wall-clock **p50/p95 in ms**, steady-state (cold model loads are
  warmed off first). p50/p95 are reported because the mean is cold-load-sensitive.
- **Reproducibility:** retrieval/rerank metrics are deterministic; generation, judge, and
  the LLM query transforms vary run-to-run. Re-running will shift generation numbers by up
  to the ~0.09 noise floor.

Full machine-readable numbers: `vectorstore/experiments/results.json`. Auto-generated
tables: `vectorstore/experiments/report.md`. Regenerate everything with
`uv run python scripts/run_experiments.py` then `uv run python scripts/plot_experiments.py`.

---

## 1. Ablation ladder — what each query-time stage adds

Hold ingest fixed (recursive chunk, bge-small embed, FAISS flat); add one query-time stage
at a time.

![ablation](figures/ablation.png)

| rung | recall@5 | MRR | p@n | ans_correct | faith | relevance | lat p50/p95 (ms) |
|---|---|---|---|---|---|---|---|
| retrieve only | 0.737 | 0.583 | **0.106** | — | — | — | **8 / 28** |
| + rerank | 0.737 | 0.583 | 0.087 | — | — | — | 932 / 1200 |
| + generate (full) | 0.737 | 0.583 | 0.087 | 0.485 | 0.991 | 0.748 | 2906 / 5081 |

**Reading.**
- **Retrieval already does the heavy lifting.** recall@5 = 0.74, MRR = 0.58 with nothing but
  dense search. The gold chunk is usually found and usually ranked high.
- **Reranking moved precision@n the *wrong* way** (0.106 → 0.087) — see §3; it is not a
  win on this corpus.
- **Generation is the expensive, valuable rung:** faithfulness 0.99, relevance 0.75 — the
  grounding prompt works. But it costs **~2 seconds** (8 ms → 2.9 s p50), ~360× the
  retrieve-only latency.

> The ablation's real lesson here: **almost all the cost is in rerank + generate, and on
> this corpus rerank buys nothing measurable.** A retrieve-only + generate pipeline would be
> ~1 s faster per query at no measured quality loss.

---

## 2. Chunker sweep — recursive vs fixed vs sentence-window

One-factor-at-a-time over the three chunkers. Each variant **re-chunks, re-embeds,
re-indexes, and re-derives its own golden set** (chunk IDs embed the chunker name, so the
ground truth must be regenerated per chunker — otherwise retrieval scores garbage).

![chunker](figures/chunker.png)

| chunker | recall@5 | recall@10 | MRR | p@n | ans_correct | faith | relevance |
|---|---|---|---|---|---|---|---|
| fixed | 0.737 | 0.747 | 0.582 | 0.087 | 0.455 | 0.965 | 0.742 |
| recursive | 0.737 | 0.747 | 0.583 | 0.087 | 0.424 | 0.992 | 0.742 |
| sentence_window | **0.536** | **0.579** | 0.579 | **0.136** | 0.485 | 0.988 | 0.747 |

**Reading.**
- **fixed ≈ recursive on retrieval.** Identical recall@5/@10, MRR within 0.001. On a corpus
  of short, already-clean passages, structure-aware splitting has nothing to exploit — the
  recursive chunker's advantage shows up on long, headed documents, which this corpus lacks.
  The `ans_correct` gap (0.455 vs 0.424) is **below the 0.09 noise floor → not a real
  difference.**
- **sentence_window makes a genuine trade.** recall@5 drops hard (0.74 → 0.54) — smaller
  units mean the gold sentence-window is more often outside the top-5 — but precision@n
  *rises* (0.087 → 0.136), because the windows that do surface are tighter. This is the
  textbook small-retrieval-unit trade-off, and it is the one real, above-noise retrieval
  effect in the chunker family.

> **No chunker is a free win here.** recursive is a safe default; sentence_window is only
> worth it if you add parent-document expansion (retrieve the precise window, feed the
> larger parent) — which this slice does not yet implement.

---

## 3. Reranker sweep — the counterintuitive result

noop vs cross-encoder, holding retrieval and generation identical (reuses one shared index).
The architecture doc calls the cross-encoder "usually the biggest quality jump per line of
code." **Here it is a regression.**

![reranker](figures/reranker.png)

| reranker | recall@5 | MRR | p@n | ans_correct | faith | relevance | lat p50/p95 (ms) |
|---|---|---|---|---|---|---|---|
| noop | 0.737 | 0.583 | **0.106** | **0.515** | 0.991 | **0.789** | **1779 / 3238** |
| cross_encoder | 0.737 | 0.583 | 0.087 | 0.394 | 0.995 | 0.744 | 2720 / 3759 |

**Reading — and the honest caveat.**
- Retrieval is identical by construction (same candidates), so recall/MRR don't move.
- The cross-encoder **lowers precision@n (0.106 → 0.087) and relevance (0.789 → 0.744)** and
  **adds ~940 ms** of latency. The `ans_correct` drop (0.515 → 0.394, −0.121) is the one
  generation delta that clears the noise floor — but only just (floor ≈ 0.09), so read it as
  "a real but modest regression," not a cliff.
- **Why would a reranker hurt?** On short single-fact passages the bi-encoder already ranks
  the gold chunk near the top; the cross-encoder re-scores the top-50 on `(query, passage)`
  surface relevance and sometimes promotes a topically-similar-but-wrong passage above the
  gold one, pushing the answer out of the top-8 the generator sees. The reranker is built
  for noisy, long-tail candidate sets; it has little to fix and some room to break here.

> **On this corpus, the cross-encoder reranker costs latency and quality.** That is a
> corpus-specific result, not a universal one — but it is exactly the kind of finding the
> harness exists to surface, and it would have been invisible without per-stage scoring.

---

## 4. Query-transform sweep — LLM rewriting before retrieval

passthrough (baseline) vs an LLM `rewrite` (expand entities/synonyms) vs `step_back`
(generalize the question). The transform rewrites the *retrieval* query only; the generator
still answers the original question.

![query_transform](figures/query_transform.png)

| transform | recall@5 | recall@10 | MRR | ans_correct | faith | relevance | lat p50/p95 (ms) |
|---|---|---|---|---|---|---|---|
| passthrough | 0.737 | 0.747 | 0.583 | 0.394 | 0.995 | 0.738 | 2840 / 6035 |
| rewrite | 0.722 | 0.747 | 0.481 | 0.455 | 0.989 | 0.739 | 4259 / 5603 |
| step_back | **0.495** | 0.616 | **0.405** | 0.394 | 0.964 | 0.723 | **4525 / 5921** |

**Reading.**
- **Both transforms hurt retrieval ranking.** rewrite drops MRR 0.583 → 0.481; step_back
  drops it harder (→ 0.405) and craters recall@5 (0.74 → 0.50). Rewriting a short factual
  question ("Which county was Lincoln born in?") into a broader or synonym-expanded query
  moves the *query* embedding away from the terse passage that answers it — the opposite of
  the asymmetry these techniques fix on verbose/conversational queries.
- **Generation barely moves** and stays inside the noise floor (rewrite's ans +0.061,
  step_back's ans ±0.000 — both ≤ floor). relevance/faithfulness are flat-to-slightly-down.
- **They add ~1.4–1.7 s of latency** (an extra LLM call per query) for no upside here.

> **Query transforms are a net loss on terse factoid queries.** They earn their cost on
> long, conversational, or multi-hop questions — which this golden set deliberately does not
> contain. The result is "no benefit *for this query distribution*," not "no benefit ever."

---

## 5. What this sweep does and does not establish

**Establishes (above the noise floor, on this corpus):**
- Dense retrieval alone is strong (recall@5 ≈ 0.74); over-retrieval saturates recall@50.
- The cross-encoder reranker is a latency + quality regression on short factual passages.
- sentence_window trades retrieval recall for rerank precision (real, measurable).
- Query transforms degrade ranking on terse factoid queries.
- Latency is dominated by rerank (~0.9 s) and generation (~2 s); transforms add ~1.5 s.

**Does NOT establish (and the report must not be read as claiming):**
- That these techniques are bad *in general*. Every negative result above is plausibly a
  corpus/query-distribution artifact (short passages, terse single-hop questions, N=33).
- Any `answer_correct` difference under ~0.09 (most of the chunker and transform generation
  deltas) — those are inside the measured generation+judge noise.
- Multi-hop / parent-document behavior — the golden set is 88% single-passage by design.

**To make the conclusions stronger** (future work, each gated behind "did eval improve?"):
grow the golden set well past 33 and add multi-hop questions; run each judged variant N
times and report variance, not a single number; add a verbose/conversational query subset
where transforms are expected to help; implement parent-document expansion to pair with
sentence_window.

---

_Generated from `scripts/run_experiments.py` (sweep) + `scripts/plot_experiments.py`
(figures). One variant (`qt_rewrite`) initially failed on a judge JSON-parsing bug that was
fixed and re-run; all 11 variants are present. Numbers from LLM-backed stages are not
bit-reproducible; expect generation metrics to vary by up to the ~0.09 noise floor on
re-runs._
