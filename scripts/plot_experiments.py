"""Render the experiment sweep's results.json into PNG charts for the report/PR.

Reads ``vectorstore/experiments/results.json`` (written by ``run_experiments.py``) and
writes one chart per experiment family plus a latency chart into ``docs/figures/``:

    ablation.png         recall@5 / MRR / precision@n across the ablation ladder
    chunker.png          retrieval + generation metrics across chunkers
    reranker.png         noop vs cross_encoder
    query_transform.png  passthrough vs rewrite vs step_back
    latency.png          per-query p50/p95 latency across all variants (log scale)

Charts are deterministic given the input JSON (no timestamps), so re-running on the same
results reproduces identical files.

    uv run python scripts/plot_experiments.py
    uv run python scripts/plot_experiments.py --results <path> --out-dir docs/figures
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")  # headless: write files, never open a window
import matplotlib.pyplot as plt  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RESULTS = REPO_ROOT / "vectorstore" / "experiments" / "results.json"
DEFAULT_OUT = REPO_ROOT / "docs" / "figures"

# Variant -> family + display order, mirroring run_experiments.variant_matrix().
FAMILIES: dict[str, list[str]] = {
    "ablation": [
        "ablation_retrieve_only",
        "ablation_retrieve_rerank",
        "ablation_full",
    ],
    "chunker": ["chunker_fixed", "chunker_recursive", "chunker_sentence_window"],
    "reranker": ["reranker_noop", "reranker_cross_encoder"],
    "query_transform": ["qt_passthrough", "qt_rewrite", "qt_step_back"],
}

# Metrics drawn in the per-family grouped bar charts (key -> label).
RETRIEVAL_METRICS = [
    ("recall_at_5", "recall@5"),
    ("recall_at_10", "recall@10"),
    ("mrr", "MRR"),
    ("precision_at_n", "p@n"),
]
GENERATION_METRICS = [
    ("answer_correct", "ans_correct"),
    ("faithfulness", "faithfulness"),
    ("answer_relevance", "relevance"),
]

_COLORS = ["#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B3", "#937860"]


def _short(name: str) -> str:
    """Strip the family prefix for readable x-axis labels."""
    for prefix in ("ablation_", "chunker_", "reranker_", "qt_"):
        if name.startswith(prefix):
            return name[len(prefix) :]
    return name


def _summaries(
    variants: dict[str, Any], names: list[str]
) -> list[tuple[str, dict[str, Any]]]:
    """Return (name, summary) for present, non-errored variants in order."""
    out: list[tuple[str, dict[str, Any]]] = []
    for name in names:
        entry = variants.get(name)
        if entry and "summary" in entry:
            out.append((name, entry["summary"]))
    return out


def _grouped_bars(
    ax: Any,
    rows: list[tuple[str, dict[str, Any]]],
    metrics: list[tuple[str, str]],
    *,
    title: str,
) -> None:
    """Draw a grouped bar chart: one group per variant, one bar per metric."""
    labels = [_short(name) for name, _ in rows]
    x = range(len(rows))
    n_metrics = len(metrics)
    width = 0.8 / max(n_metrics, 1)

    for i, (key, mlabel) in enumerate(metrics):
        values = [summary.get(key) for _, summary in rows]
        positions = [j + (i - (n_metrics - 1) / 2) * width for j in x]
        plotted = [(p, v) for p, v in zip(positions, values, strict=True) if v is not None]
        if not plotted:
            continue
        ax.bar(
            [p for p, _ in plotted],
            [v for _, v in plotted],
            width=width,
            label=mlabel,
            color=_COLORS[i % len(_COLORS)],
        )
    ax.set_xticks(list(x))
    ax.set_xticklabels(labels, rotation=0)
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("score")
    ax.set_title(title)
    ax.legend(fontsize=8, ncol=n_metrics)
    ax.grid(axis="y", linestyle=":", alpha=0.5)


def plot_family(
    family: str, rows: list[tuple[str, dict[str, Any]]], out_dir: Path
) -> Path | None:
    """Plot retrieval (+ generation if any judged) metrics for one family."""
    if not rows:
        return None
    has_generation = any(
        any(s.get(k) is not None for k, _ in GENERATION_METRICS) for _, s in rows
    )
    n_panels = 2 if has_generation else 1
    fig, axes = plt.subplots(1, n_panels, figsize=(6.5 * n_panels, 4.2), squeeze=False)
    _grouped_bars(
        axes[0][0], rows, RETRIEVAL_METRICS, title=f"{family} — retrieval / rerank"
    )
    if has_generation:
        _grouped_bars(
            axes[0][1], rows, GENERATION_METRICS, title=f"{family} — generation"
        )
    fig.tight_layout()
    path = out_dir / f"{family}.png"
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


def plot_latency(variants: dict[str, Any], out_dir: Path) -> Path | None:
    """Per-query p50/p95 latency across all variants (log y — values span ~10ms..3s)."""
    rows: list[tuple[str, float, float]] = []
    for names in FAMILIES.values():
        for name, summary in _summaries(variants, names):
            p50 = summary.get("query_latency_ms_p50")
            p95 = summary.get("query_latency_ms_p95")
            if p50 is not None:
                rows.append((name, p50, p95 if p95 is not None else p50))
    if not rows:
        return None

    fig, ax = plt.subplots(figsize=(max(7, len(rows) * 0.9), 4.2))
    x = range(len(rows))
    width = 0.4
    ax.bar([i - width / 2 for i in x], [r[1] for r in rows], width, label="p50",
           color=_COLORS[0])
    ax.bar([i + width / 2 for i in x], [r[2] for r in rows], width, label="p95",
           color=_COLORS[1])
    ax.set_yscale("log")
    ax.set_ylabel("query latency (ms, log)")
    ax.set_xticks(list(x))
    ax.set_xticklabels([_short(r[0]) for r in rows], rotation=40, ha="right", fontsize=8)
    ax.set_title("per-query latency: p50 / p95 across variants")
    ax.legend()
    ax.grid(axis="y", linestyle=":", alpha=0.5, which="both")
    fig.tight_layout()
    path = out_dir / "latency.png"
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args(argv)

    data = json.loads(args.results.read_text(encoding="utf-8"))
    variants = data["variants"]
    args.out_dir.mkdir(parents=True, exist_ok=True)

    written: list[Path] = []
    for family, names in FAMILIES.items():
        rows = _summaries(variants, names)
        path = plot_family(family, rows, args.out_dir)
        if path is not None:
            written.append(path)
    latency = plot_latency(variants, args.out_dir)
    if latency is not None:
        written.append(latency)

    for path in written:
        print(f"wrote {path.relative_to(REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
