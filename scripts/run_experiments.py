"""Run the technique-comparison sweep and emit a per-stage report.

This is the payoff of the swappable-stage design (ARCHITECTURE.md §5): each *variant* is
the base ``config.yaml`` with one stage overridden; the runner re-runs the relevant slice
of the pipeline per variant into an isolated directory and scores every stage, so a config
change can be traced to the stage it helped or hurt.

Why a separate process per variant (subprocess, not in-process import): the embedder and
cross-encoder cache loaded models on their instances and the registry reads config at call
time, so a fresh ``uv run python`` per stage guarantees no cross-variant config bleed or
stale-model leakage. The existing scripts already expose the exact CLI needed
(``chunk_corpus``/``embed_chunks``/``index_corpus``/``derive_golden``/``evaluate``), so the
runner is pure orchestration.

Two invariants are respected by construction:

- **embedder-match** (ARCHITECTURE.md §2): each variant *builds and loads* its index under
  its own materialized config, so ``FaissFlatIndexer.load`` never sees a mismatched embedder.
- **chunker-coupling** (ARCHITECTURE.md §5): ``expected_chunk_ids`` embed the chunker name
  and only exist after that chunker runs, so the golden set is re-derived per variant from
  that variant's own ``chunks.jsonl``. A chunker sweep that skipped this would score
  retrieval against ids that do not exist.

Layout per variant under ``--experiments-root`` (default ``vectorstore/experiments``)::

    <variant>/config.yaml            materialized base + override
    <variant>/chunks.jsonl           re-chunked corpus
    <variant>/embedded_chunks.jsonl  re-embedded chunks
    <variant>/index.faiss + ...      the saved index
    <variant>/golden.jsonl           re-derived golden set
    <variant>/results.json           evaluate.py --json output

Reuse-ingest (default on): variants that change only query-time stages (reranker,
query_transform) share the ingest of a variant with the same chunker+embedder+indexer
config slice, so the 3200-passage corpus is not re-chunked/re-embedded four times. The
chunker sweep is the only family that legitimately re-ingests per variant.

    uv run python scripts/run_experiments.py                 # full sweep, judged
    uv run python scripts/run_experiments.py --no-judge      # deterministic, free-ish
    uv run python scripts/run_experiments.py --only ablation_retrieve_only
    uv run python scripts/run_experiments.py --no-reuse-ingest
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

DEFAULT_CONFIG = REPO_ROOT / "config.yaml"
DEFAULT_CORPUS = REPO_ROOT / "data" / "corpus"
DEFAULT_EXPERIMENTS_ROOT = REPO_ROOT / "vectorstore" / "experiments"


# --- variant matrix ----------------------------------------------------------
#
# Each variant is {name, family, overrides, eval_flags}. `overrides` is deep-merged onto
# the base config. `eval_flags` are extra evaluate.py flags. The two retrieve-only ablation
# rungs use --no-generate (there is no answer to judge); everything else is judged when
# --judge is on.

_CROSS_ENCODER = {
    "name": "cross_encoder",
    "model": "BAAI/bge-reranker-base",
    "top_n": 8,
}
_NOOP_RERANK = {"name": "noop", "top_n": 8}


def variant_matrix() -> list[dict[str, Any]]:
    """The full set of variants, grouped by experiment family."""
    return [
        # --- ablation ladder: what each query-time stage adds -----------------
        {
            "name": "ablation_retrieve_only",
            "family": "ablation",
            "overrides": {"reranker": _NOOP_RERANK},
            "eval_flags": ["--no-generate"],
        },
        {
            "name": "ablation_retrieve_rerank",
            "family": "ablation",
            "overrides": {"reranker": _CROSS_ENCODER},
            "eval_flags": ["--no-generate"],
        },
        {
            "name": "ablation_full",
            "family": "ablation",
            "overrides": {"reranker": _CROSS_ENCODER},
            "eval_flags": [],  # generation + judge (when --judge)
        },
        # --- chunker sweep (full re-ingest per variant) -----------------------
        {
            "name": "chunker_fixed",
            "family": "chunker",
            "overrides": {"chunker": {"name": "fixed", "size": 256, "overlap": 32}},
            "eval_flags": [],
        },
        {
            "name": "chunker_recursive",
            "family": "chunker",
            "overrides": {"chunker": {"name": "recursive", "size": 256, "overlap": 32}},
            "eval_flags": [],
        },
        {
            "name": "chunker_sentence_window",
            "family": "chunker",
            "overrides": {
                "chunker": {"name": "sentence_window", "sentence_count": 4, "overlap": 1}
            },
            "eval_flags": [],
        },
        # --- reranker sweep (reuse ingest) ------------------------------------
        {
            "name": "reranker_noop",
            "family": "reranker",
            "overrides": {"reranker": _NOOP_RERANK},
            "eval_flags": [],
        },
        {
            "name": "reranker_cross_encoder",
            "family": "reranker",
            "overrides": {"reranker": _CROSS_ENCODER},
            "eval_flags": [],
        },
        # --- query_transform sweep (reuse ingest) -----------------------------
        {
            "name": "qt_passthrough",
            "family": "query_transform",
            "overrides": {"query_transform": {"name": "passthrough"}},
            "eval_flags": [],
        },
        {
            "name": "qt_rewrite",
            "family": "query_transform",
            "overrides": {"query_transform": {"name": "rewrite"}},
            "eval_flags": [],
        },
        {
            "name": "qt_step_back",
            "family": "query_transform",
            "overrides": {"query_transform": {"name": "step_back"}},
            "eval_flags": [],
        },
    ]


# --- config materialization --------------------------------------------------


def deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Recursively merge ``override`` onto a copy of ``base``.

    Exception: when the override *changes a stage's ``name``* (e.g. reranker
    cross_encoder → noop), the old technique's params (``model``, ``top_n``, …) are
    meaningless and would be passed to the new class as unexpected kwargs. So an override
    dict that sets a different ``name`` than the base **replaces** the block wholesale
    rather than merging into it.
    """
    merged = copy.deepcopy(base)
    for key, value in override.items():
        base_value = merged.get(key)
        if isinstance(base_value, dict) and isinstance(value, dict):
            changes_name = (
                "name" in value
                and "name" in base_value
                and value["name"] != base_value["name"]
            )
            if changes_name:
                merged[key] = copy.deepcopy(value)
            else:
                merged[key] = deep_merge(base_value, value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def ingest_key(config: dict[str, Any]) -> str:
    """Stable hash of the stages that determine the index + golden set.

    Two variants share ingest iff their chunker, embedder, and indexer blocks match — the
    reranker / query_transform / generator are query-time and do not affect the corpus.
    """
    slice_ = {stage: config.get(stage, {}) for stage in ("chunker", "embedder", "indexer")}
    blob = json.dumps(slice_, sort_keys=True).encode("utf-8")
    return hashlib.blake2b(blob, digest_size=8).hexdigest()


# --- subprocess driving ------------------------------------------------------


def _run(cmd: list[str], *, label: str) -> None:
    """Run a subprocess, raising RuntimeError with context on failure."""
    print(f"    $ {' '.join(cmd[3:])}", flush=True)  # drop the `uv run python` prefix
    result = subprocess.run(cmd, cwd=REPO_ROOT, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(
            f"{label} failed (exit {result.returncode}):\n"
            f"{result.stdout[-2000:]}\n{result.stderr[-2000:]}"
        )


def _py(*args: str) -> list[str]:
    return ["uv", "run", "python", *args]


def ingest(config_path: Path, out_dir: Path) -> None:
    """Chunk → embed → index → derive-golden into ``out_dir``."""
    chunks = out_dir / "chunks.jsonl"
    embedded = out_dir / "embedded_chunks.jsonl"
    golden = out_dir / "golden.jsonl"

    _run(
        _py("scripts/chunk_corpus.py", "--config", str(config_path),
            "--corpus-dir", str(DEFAULT_CORPUS), "--output", str(chunks), "--write"),
        label="chunk",
    )
    _run(
        _py("scripts/embed_chunks.py", "--config", str(config_path),
            "--input", str(chunks), "--output", str(embedded), "--write"),
        label="embed",
    )
    _run(
        _py("scripts/index_corpus.py", "--config", str(config_path),
            "--input", str(embedded), "--output-dir", str(out_dir), "--write"),
        label="index",
    )
    _run(
        _py("scripts/derive_golden.py", "--chunks", str(chunks), "--output", str(golden)),
        label="derive_golden",
    )


def evaluate(
    config_path: Path, vectorstore: Path, golden: Path, results: Path,
    *, n: int, judge: bool, extra_flags: list[str],
) -> None:
    """Run evaluate.py for one variant, writing results.json."""
    cmd = _py(
        "evaluate.py",
        "--config", str(config_path),
        "--vectorstore", str(vectorstore),
        "--golden", str(golden),
        "--n", str(n),
        "--json", str(results),
    )
    if judge and "--no-generate" not in extra_flags:
        cmd.append("--judge")
    cmd.extend(extra_flags)
    _run(cmd, label="evaluate")


# --- report emission ---------------------------------------------------------

_METRIC_COLUMNS = [
    ("recall_at_5", "recall@5"),
    ("recall_at_10", "recall@10"),
    ("recall_at_k", "recall@k"),
    ("mrr", "MRR"),
    ("precision_at_n", "p@n"),
    ("answer_correct", "ans_correct"),
    ("faithfulness", "faith"),
    ("answer_relevance", "relevance"),
]


def _fmt(value: Any) -> str:
    if value is None:
        return "—"
    return f"{value:.3f}"


def _latency_cell(summary: dict[str, Any]) -> str:
    p50 = summary.get("query_latency_ms_p50")
    p95 = summary.get("query_latency_ms_p95")
    if p50 is None:
        return "—"
    return f"{p50:.0f}/{p95:.0f}"


def render_report(collected: dict[str, dict[str, Any]], *, k: int, n: int) -> str:
    """Render the markdown report, one table per experiment family."""
    matrix = {v["name"]: v for v in variant_matrix()}
    families: dict[str, list[str]] = {}
    for name in collected:
        fam = matrix.get(name, {}).get("family", "other")
        families.setdefault(fam, []).append(name)

    # Per family, the first listed variant is the baseline for delta columns.
    family_order = ["ablation", "chunker", "reranker", "query_transform"]
    baselines = {
        "ablation": "ablation_retrieve_only",
        "chunker": "chunker_recursive",
        "reranker": "reranker_noop",
        "query_transform": "qt_passthrough",
    }

    lines: list[str] = []
    lines.append("# Technique comparison report\n")
    lines.append(
        f"Each variant is the base config with one stage overridden, scored over the "
        f"golden set (k={k} over-retrieve, n={n} for precision@n). Latency is per-query "
        f"wall-clock **p50/p95 in ms** (mean is cold-load sensitive, so p50/p95 are the "
        f"headline). recall@k saturates on this corpus — read recall@5 and MRR for "
        f"retrieval deltas.\n"
    )
    lines.append(
        "> Generation metrics (faith/relevance) are only present for judged variants; "
        "retrieve-only ablation rungs have no answer to score (shown as —).\n"
    )

    header = (
        "| variant | "
        + " | ".join(label for _, label in _METRIC_COLUMNS)
        + " | lat p50/p95 |"
    )
    sep = "|" + "---|" * (len(_METRIC_COLUMNS) + 2)

    for fam in family_order:
        if fam not in families:
            continue
        lines.append(f"\n## {fam}\n")
        base_name = baselines.get(fam, "")
        base_summary = collected.get(base_name, {}).get("summary", {})
        lines.append(header)
        lines.append(sep)
        # Keep matrix declaration order within the family.
        ordered = [v["name"] for v in variant_matrix() if v["name"] in families[fam]]
        for name in ordered:
            entry = collected[name]
            if "error" in entry:
                lines.append(f"| {name} | **ERROR** {entry['error'][:80]} |" + " |" * 9)
                continue
            summary = entry["summary"]
            cells = [_fmt(summary.get(key)) for key, _ in _METRIC_COLUMNS]
            row = f"| {name} | " + " | ".join(cells) + f" | {_latency_cell(summary)} |"
            lines.append(row)
        # Delta-vs-baseline note for the headline retrieval/generation metrics.
        if base_summary:
            lines.append(
                f"\n_Baseline: `{base_name}`. Deltas below are vs. this row._\n"
            )
            for name in ordered:
                if name == base_name or "error" in collected[name]:
                    continue
                summary = collected[name]["summary"]
                deltas = []
                for key, label in (
                    ("recall_at_5", "recall@5"),
                    ("mrr", "MRR"),
                    ("answer_correct", "ans"),
                    ("faithfulness", "faith"),
                ):
                    if key in summary and key in base_summary:
                        d = summary[key] - base_summary[key]
                        deltas.append(f"{label} {d:+.3f}")
                if deltas:
                    lines.append(f"- **{name}**: " + ", ".join(deltas))

    # Per-stage latency attribution — the harness's headline feature (ARCHITECTURE.md §5):
    # mean wall-clock ms each stage spent, so cost can be traced to the stage that incurred
    # it. These are means (not p50/p95) to keep the columns additive to query_latency_ms.
    lines.append("\n## per-stage latency (mean ms)\n")
    lines.append("| variant | transform | retrieve | rerank | generate | total |")
    lines.append("|---|---|---|---|---|---|")
    for variant in variant_matrix():
        name = variant["name"]
        entry = collected.get(name)
        if not entry or "summary" not in entry:
            continue
        s = entry["summary"]

        def _ms(key: str, summary: dict[str, Any] = s) -> str:
            value = summary.get(key)
            return f"{value:.0f}" if value is not None else "—"

        lines.append(
            f"| {name} | {_ms('transform_ms')} | {_ms('retrieve_ms')} | "
            f"{_ms('rerank_ms')} | {_ms('generate_ms')} | {_ms('query_latency_ms')} |"
        )

    lines.append("\n---\n")
    lines.append(
        "_Generated by `scripts/run_experiments.py`. **Each variant is a single run.** "
        "Generation/judge metrics (`ans_correct`, `faith`, `relevance`) and the LLM "
        "transforms vary run-to-run; the measured generation noise floor is ~0.09 "
        "(`ans_correct` range across four identical-pipeline variants), so treat any "
        "generation delta of that order or smaller as noise, not a technique effect. `—` "
        "means a metric was not measured (e.g. no generation under `--no-generate`), not a "
        "score of zero. See `EXPERIMENT_REPORT.md` for the full interpretation._\n"
    )
    return "\n".join(lines)


# --- orchestration -----------------------------------------------------------


def run(
    *,
    base_config: dict[str, Any],
    experiments_root: Path,
    only: list[str] | None,
    n: int,
    judge: bool,
    reuse_ingest: bool,
    warmup: bool,
) -> dict[str, dict[str, Any]]:
    """Run the (optionally filtered) variant matrix; return collected results."""
    variants = variant_matrix()
    if only:
        wanted = set(only)
        variants = [v for v in variants if v["name"] in wanted]
        missing = wanted - {v["name"] for v in variants}
        if missing:
            raise SystemExit(f"unknown variant(s): {', '.join(sorted(missing))}")

    experiments_root.mkdir(parents=True, exist_ok=True)
    collected: dict[str, dict[str, Any]] = {}
    ingest_cache: dict[str, Path] = {}  # ingest_key -> directory holding index+golden

    for variant in variants:
        name = variant["name"]
        print(f"\n=== {name} ({variant['family']}) ===", flush=True)
        out_dir = experiments_root / name
        out_dir.mkdir(parents=True, exist_ok=True)

        config = deep_merge(base_config, variant["overrides"])
        config_path = out_dir / "config.yaml"
        config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")

        results_path = out_dir / "results.json"
        try:
            key = ingest_key(config)
            source = ingest_cache.get(key) if reuse_ingest else None
            if source is not None:
                print(f"    reusing ingest from {source.name}", flush=True)
                index_dir = source
                golden = source / "golden.jsonl"
            else:
                ingest(config_path, out_dir)
                ingest_cache[key] = out_dir
                index_dir = out_dir
                golden = out_dir / "golden.jsonl"

            flags = list(variant["eval_flags"])
            if not warmup:
                flags.append("--no-warmup")
            evaluate(
                config_path, index_dir, golden, results_path,
                n=n, judge=judge, extra_flags=flags,
            )
            payload = json.loads(results_path.read_text(encoding="utf-8"))
            collected[name] = {
                "overrides": variant["overrides"],
                "eval_flags": variant["eval_flags"],
                "summary": payload["summary"],
            }
            print(f"    done: {payload['summary']}", flush=True)
        except Exception as exc:  # noqa: BLE001 - one variant failing must not stop the sweep
            print(f"    !! {name} failed: {exc}", file=sys.stderr, flush=True)
            collected[name] = {"overrides": variant["overrides"], "error": str(exc)}

    return collected


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--experiments-root", type=Path, default=DEFAULT_EXPERIMENTS_ROOT)
    parser.add_argument("--n", type=int, default=8, help="top-n for precision@n")
    parser.add_argument(
        "--only", type=str, default=None,
        help="comma-separated variant names to run (default: all)",
    )
    parser.add_argument(
        "--judge", action=argparse.BooleanOptionalAction, default=True,
        help="LLM-as-judge generation scoring on judged variants (needs API)",
    )
    parser.add_argument(
        "--reuse-ingest", action=argparse.BooleanOptionalAction, default=True,
        help="share ingest across variants with the same chunker+embedder+indexer",
    )
    parser.add_argument(
        "--warmup", action=argparse.BooleanOptionalAction, default=True,
        help="pay lazy model-load cost once before timing each variant",
    )
    args = parser.parse_args(argv)

    base_config = yaml.safe_load(args.config.read_text(encoding="utf-8")) or {}
    only = [s.strip() for s in args.only.split(",")] if args.only else None

    # Read k from the base config for the report header.
    k = base_config.get("retriever", {}).get("k", 50)

    started = time.perf_counter()
    collected = run(
        base_config=base_config,
        experiments_root=args.experiments_root,
        only=only,
        n=args.n,
        judge=args.judge,
        reuse_ingest=args.reuse_ingest,
        warmup=args.warmup,
    )
    elapsed = time.perf_counter() - started

    results_json = args.experiments_root / "results.json"
    results_json.write_text(
        json.dumps({"k": k, "n": args.n, "variants": collected}, indent=2) + "\n",
        encoding="utf-8",
    )
    report = render_report(collected, k=k, n=args.n)
    report_md = args.experiments_root / "report.md"
    report_md.write_text(report, encoding="utf-8")

    n_ok = sum(1 for v in collected.values() if "error" not in v)
    print(
        f"\nran {len(collected)} variants ({n_ok} ok, {len(collected) - n_ok} failed) "
        f"in {elapsed:.0f}s",
        flush=True,
    )
    print(f"wrote {report_md}")
    print(f"wrote {results_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
