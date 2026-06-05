"""Render the agentic-RAG model sweep into charts for the report/PR.

Reads ``docs/agent_experiment_results.json`` (written by ``run_agent_experiments.py``) and
writes two charts into ``docs/figures/``:

    agent_quality.png    answer_correct / faithfulness / answer_relevance per model tier
    agent_latency.png    mean + p50/p95 end-to-end latency per model tier (log scale)

The tiers (haiku → sonnet → opus) are drawn in capability order. Charts are deterministic
given the input JSON.

    uv run python scripts/plot_agent_experiments.py
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RESULTS = REPO_ROOT / "docs" / "agent_experiment_results.json"
DEFAULT_OUT = REPO_ROOT / "docs" / "figures"

# Draw tiers in capability order regardless of JSON key order.
TIER_ORDER = ["agent_haiku", "agent_sonnet", "agent_opus"]

QUALITY_METRICS = [
    ("answer_correct", "ans_correct (substring)"),
    ("faithfulness", "faithfulness"),
    ("answer_relevance", "relevance"),
    ("precision_at_n", "citation p@n"),
]

_COLORS = ["#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B3"]


def _short(name: str) -> str:
    return name[len("agent_") :] if name.startswith("agent_") else name


def _ordered_rows(variants: dict[str, Any]) -> list[tuple[str, dict[str, Any]]]:
    """(name, summary) in capability order, then any extras not in TIER_ORDER."""
    rows: list[tuple[str, dict[str, Any]]] = []
    for name in TIER_ORDER:
        entry = variants.get(name)
        if entry and "summary" in entry:
            rows.append((name, entry["summary"]))
    for name, entry in variants.items():
        if name not in TIER_ORDER and "summary" in entry:
            rows.append((name, entry["summary"]))
    return rows


def plot_quality(rows: list[tuple[str, dict[str, Any]]], out_dir: Path) -> Path | None:
    if not rows:
        return None
    labels = [_short(name) for name, _ in rows]
    x = range(len(rows))
    n_metrics = len(QUALITY_METRICS)
    width = 0.8 / n_metrics

    fig, ax = plt.subplots(figsize=(max(7, len(rows) * 2.2), 4.5))
    for i, (key, mlabel) in enumerate(QUALITY_METRICS):
        values = [s.get(key) for _, s in rows]
        positions = [j + (i - (n_metrics - 1) / 2) * width for j in x]
        plotted = [(p, v) for p, v in zip(positions, values, strict=True) if v is not None]
        if not plotted:
            continue
        bars = ax.bar(
            [p for p, _ in plotted],
            [v for _, v in plotted],
            width=width,
            label=mlabel,
            color=_COLORS[i % len(_COLORS)],
        )
        ax.bar_label(bars, fmt="%.2f", fontsize=7, padding=1)
    ax.set_xticks(list(x))
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("score")
    ax.set_title("Agentic RAG — answer quality by model tier")
    ax.legend(fontsize=8, ncol=n_metrics)
    ax.grid(axis="y", linestyle=":", alpha=0.5)
    fig.tight_layout()
    path = out_dir / "agent_quality.png"
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


def plot_latency(rows: list[tuple[str, dict[str, Any]]], out_dir: Path) -> Path | None:
    if not rows:
        return None
    labels = [_short(name) for name, _ in rows]
    x = list(range(len(rows)))
    width = 0.27

    fig, ax = plt.subplots(figsize=(max(7, len(rows) * 2.0), 4.5))
    series = [
        ("query_latency_ms", "mean", _COLORS[0], -1),
        ("query_latency_ms_p50", "p50", _COLORS[2], 0),
        ("query_latency_ms_p95", "p95", _COLORS[1], 1),
    ]
    for key, label, color, offset in series:
        # A log axis cannot place a 0 (log(0) = -inf), so leave missing metrics out
        # entirely rather than coercing them to 0.0 — only plot real values.
        positions: list[float] = []
        heights: list[float] = []
        for i, (_, s) in enumerate(rows):
            value = s.get(key)
            if value is None:
                continue
            positions.append(i + offset * width)
            heights.append(float(value))
        if not heights:
            continue
        bars = ax.bar(positions, heights, width, label=label, color=color)
        ax.bar_label(bars, fmt="%.0f", fontsize=7, padding=1)
    ax.set_yscale("log")
    ax.set_ylabel("end-to-end latency per query (ms, log)")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_title("Agentic RAG — latency by model tier")
    ax.legend(fontsize=8)
    ax.grid(axis="y", linestyle=":", alpha=0.5, which="both")
    fig.tight_layout()
    path = out_dir / "agent_latency.png"
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args(argv)

    data = json.loads(args.results.read_text(encoding="utf-8"))
    rows = _ordered_rows(data["variants"])
    args.out_dir.mkdir(parents=True, exist_ok=True)

    written = [p for p in (plot_quality(rows, args.out_dir), plot_latency(rows, args.out_dir)) if p]
    for path in written:
        display = path.relative_to(REPO_ROOT) if path.is_relative_to(REPO_ROOT) else path
        print(f"wrote {display}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
