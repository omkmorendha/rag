"""Offline evaluation harness (ARCHITECTURE.md §5).

Runs the golden set through the query path and scores each stage separately:

    retrieval:  recall@k, MRR
    rerank:     precision@n
    generation: answer_correct (substring); faithfulness + answer_relevance with --judge

The whole point of the strategy/registry design: change a stage in config.yaml, re-run
this, and read the delta.

    uv run python evaluate.py                 # deterministic, free, offline
    uv run python evaluate.py --judge         # add LLM-as-judge generation scoring
    uv run python evaluate.py --json runs/baseline.json

The judge needs ANTHROPIC_API_KEY. Generation always runs (it is part of the query path),
but ANTHROPIC_API_KEY is required for that too — use --no-generate to score retrieval and
rerank only, fully offline and free.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_ROOT))

from rag.eval import (  # noqa: E402
    AnthropicJudge,
    QueryResult,
    aggregate,
    evaluate_row,
    load_golden,
    result_to_dict,
)
from rag.registry import (  # noqa: E402
    build_embedder,
    build_generator,
    build_indexer,
    build_query_transform,
    build_reranker,
    build_retriever,
    retriever_k,
)

DEFAULT_GOLDEN = REPO_ROOT / "eval" / "golden.jsonl"
DEFAULT_VECTORSTORE = REPO_ROOT / "vectorstore"


def _load_dotenv() -> None:
    """Load .env, overriding empty shell vars (an empty ANTHROPIC_API_KEY shadows .env)."""
    env_path = REPO_ROOT / ".env"
    if not env_path.exists():
        return
    for line in env_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key, value = key.strip(), value.strip()
        if value and not os.environ.get(key):  # fill if unset OR empty
            os.environ[key] = value


class _NullGenerator:
    """Stand-in when --no-generate is set: never calls the API."""

    def generate(self, query: str, chunks: list) -> str:  # noqa: ARG002
        return ""


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--golden", type=Path, default=DEFAULT_GOLDEN)
    parser.add_argument("--config", type=Path, default=REPO_ROOT / "config.yaml")
    parser.add_argument("--vectorstore", type=Path, default=DEFAULT_VECTORSTORE)
    parser.add_argument("--n", type=int, default=8, help="top-n for precision@n")
    parser.add_argument(
        "--judge",
        action="store_true",
        help="LLM-as-judge generation scoring (needs API)",
    )
    parser.add_argument(
        "--no-generate",
        action="store_true",
        help="skip generation; score retrieval + rerank only (offline, free)",
    )
    parser.add_argument(
        "--json", type=Path, default=None, help="dump full results as JSON"
    )
    parser.add_argument(
        "--warmup",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="pay lazy model-load cost once before timing (use --no-warmup to skip)",
    )
    args = parser.parse_args()

    _load_dotenv()

    golden = load_golden(args.golden)
    k = retriever_k(path=args.config)

    index = build_indexer(path=args.config).load(args.vectorstore)
    embedder = build_embedder(path=args.config)
    retriever = build_retriever(index, embedder, path=args.config)
    query_transform = build_query_transform(path=args.config)
    reranker = build_reranker(path=args.config)
    generator = (
        _NullGenerator() if args.no_generate else build_generator(path=args.config)
    )
    judge = AnthropicJudge() if args.judge else None

    if args.warmup:
        # Pay the lazy model-load cost (sentence-transformers embedder + cross-encoder
        # reranker load on first call) so per-row latencies are steady-state. We never
        # warm generation/judge — those hit the API and cost money. Guarded so a warmup
        # failure never aborts the actual run.
        try:
            candidates = retriever.retrieve("warmup", k)
            reranker.rerank("warmup", candidates)
        except Exception as exc:  # noqa: BLE001 - warmup is best-effort
            print(f"warmup skipped: {exc}", file=sys.stderr)

    results: list[QueryResult] = []
    for row in golden:
        result = evaluate_row(
            row,
            retriever=retriever,
            reranker=reranker,
            generator=generator,
            k=k,
            n=args.n,
            judge=judge,
            query_transform=query_transform,
        )
        results.append(result)

    _print_report(results, k=k, n=args.n, judged=judge is not None)

    if args.json is not None:
        payload = {
            "config": str(args.config),
            "k": k,
            "n": args.n,
            "summary": aggregate(results),
            "results": [result_to_dict(r) for r in results],
        }
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        print(f"\nwrote {args.json}")

    return 0


def _print_report(results: list[QueryResult], *, k: int, n: int, judged: bool) -> None:
    print(f"Evaluated {len(results)} queries  (k={k}, n={n})\n")
    for r in results:
        line = (
            f"  recall@{k}={r.recall_at_k:.2f}  mrr={r.mrr:.2f}  "
            f"p@{n}={r.precision_at_n:.2f}  ans={'Y' if r.answer_correct else 'N'}"
        )
        if judged and r.faithfulness is not None:
            line += f"  faith={r.faithfulness:.2f}  rel={r.answer_relevance:.2f}"
        if r.query_latency_ms is not None:
            line += f"  lat={r.query_latency_ms:.0f}ms"
        print(f"- {r.query}")
        print(line)

    summary = aggregate(results)
    print("\n=== aggregate ===")
    for key in ("recall_at_k", "mrr", "precision_at_n", "answer_correct"):
        if key in summary:
            print(f"  {key}: {summary[key]:.3f}")
    if "recall_at_5" in summary:
        print(f"  recall@5: {summary['recall_at_5']:.3f}")
    if "recall_at_10" in summary:
        print(f"  recall@10: {summary['recall_at_10']:.3f}")
    if "faithfulness" in summary:
        print(f"  faithfulness: {summary['faithfulness']:.3f}")
    if "answer_relevance" in summary:
        print(f"  answer_relevance: {summary['answer_relevance']:.3f}")
    if "query_latency_ms" in summary:
        print(
            f"  latency_ms: mean={summary['query_latency_ms']:.0f}  "
            f"p50={summary.get('query_latency_ms_p50', 0.0):.0f}  "
            f"p95={summary.get('query_latency_ms_p95', 0.0):.0f}"
        )


if __name__ == "__main__":
    raise SystemExit(main())
