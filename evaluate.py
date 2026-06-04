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
    args = parser.parse_args()

    _load_dotenv()

    golden = load_golden(args.golden)
    k = retriever_k(path=args.config)

    index = build_indexer(path=args.config).load(args.vectorstore)
    embedder = build_embedder(path=args.config)
    retriever = build_retriever(index, embedder, path=args.config)
    reranker = build_reranker(path=args.config)
    generator = (
        _NullGenerator() if args.no_generate else build_generator(path=args.config)
    )
    judge = AnthropicJudge() if args.judge else None

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
        print(f"- {r.query}")
        print(line)

    summary = aggregate(results)
    print("\n=== aggregate ===")
    for key, value in summary.items():
        print(f"  {key}: {value:.3f}")


if __name__ == "__main__":
    raise SystemExit(main())
