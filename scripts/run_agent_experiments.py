"""Agentic-RAG experiment — point a Claude Agent at the corpus and let it investigate.

This is a *different shape* of experiment from ``run_experiments.py``. Instead of the
embed → FAISS → rerank → generate pipeline, we hand a Claude Agent the ``data/`` folder
and let it explore the corpus on its own (its own choice of Read/Grep/Glob/…) to answer
each golden question. We run the same 33-question golden set once per model tier
(Haiku, Sonnet, Opus) and emit metrics in the *same JSON shape* as the pipeline runs so
the two can be compared side by side.

What carries over from the pipeline:

- the grounding/answer criteria (reused from ``rag.generator.anthropic.SYSTEM_PROMPT``),
  reframed to "explore the folder" instead of "answer from these chunks";
- ``answer_correct`` — case-insensitive substring of the expected answer (``rag.eval.metrics``);
- ``faithfulness`` + ``answer_relevance`` — the same ``AnthropicJudge`` (claude-haiku-4-5).

What is *agentic-specific*:

- the "retrieved chunks" handed to the judge are reconstructed from the **source
  citations** the agent emits (``[source: passage NNN]``) — we pull those passages out of
  the corpus and pass them as the grounding context, mirroring how the pipeline feeds the
  reranked chunks to the judge;
- ``citation_recall@k`` / ``precision_at_n`` are computed over the agent's *cited* passage
  IDs (mapped to the ``passage:N:recursive:0`` chunk-id convention the golden set uses),
  so they line up with ``expected_chunk_ids``. Pure embedding metrics (``recall@k``, ``mrr``)
  have no analogue here and are left unset.

Run:

    uv run python scripts/run_agent_experiments.py                 # all three tiers
    uv run python scripts/run_agent_experiments.py --models haiku  # one tier
    uv run python scripts/run_agent_experiments.py --limit 3       # smoke test

Auth (two separate credentials, on purpose):

- the **agent** runs on your **Claude subscription** — the Agent SDK drives the ``claude``
  CLI, which uses the OAuth login from ``claude /login`` (stored in the OS keychain). We
  strip ``ANTHROPIC_API_KEY`` from the agent subprocess env (``_agent_env``) so it does NOT
  fall back to the metered API;
- the **judge** is a direct Messages API call and uses ``ANTHROPIC_API_KEY`` (from .env,
  same as evaluate.py).

So: be logged in via ``claude /login`` (subscription) AND have ANTHROPIC_API_KEY in .env
(for the judge). Needs the ``claude`` CLI on PATH.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from claude_agent_sdk import (  # noqa: E402
    AssistantMessage,
    ClaudeAgentOptions,
    ResultMessage,
    TextBlock,
    query,
)

from rag.eval import (  # noqa: E402
    AnthropicJudge,
    QueryResult,
    aggregate,
    load_golden,
    result_to_dict,
)
from rag.eval.metrics import answer_contains, precision_at_n, recall_at_k  # noqa: E402
from rag.generator.anthropic import SYSTEM_PROMPT as PIPELINE_SYSTEM  # noqa: E402
from rag.types import Chunk  # noqa: E402

DATA_DIR = REPO_ROOT / "data"
CORPUS_DIR = DATA_DIR / "corpus"
DEFAULT_GOLDEN = REPO_ROOT / "eval" / "golden.jsonl"
OUT_DIR = REPO_ROOT / "vectorstore" / "agent_experiments"
DOCS_JSON = REPO_ROOT / "docs" / "agent_experiment_results.json"

# Model tiers. The label is what the run is stored under; the id is the model the SDK uses.
MODELS = {
    "haiku": "claude-haiku-4-5",
    "sonnet": "claude-sonnet-4-6",
    "opus": "claude-opus-4-8",
}

# Reuse the pipeline's grounding criteria verbatim, then bolt on the "go explore the
# folder yourself" framing — the answer/citation contract is identical so the judge scores
# the agent on the same terms as the pipeline generator.
AGENT_SYSTEM = (
    PIPELINE_SYSTEM
    + """

<investigation>
You are NOT given the chunks. Instead you have full access to a corpus of Markdown files in
the current working directory (the `corpus/` folder). Each file holds many passages, each
under a `## passage NNN` heading. Investigate the corpus however you see fit — search, grep,
read files — to find the passage(s) that answer the user's query, then answer using the SAME
answer criteria above (ground every claim in what you found; if it is not in the corpus, say
"I don't know"; cite the passage you used as [source: passage NNN]).

Keep the final answer short and direct, exactly as the answer criteria require. End your
turn with the answer only.
</investigation>"""
)

# "[source: passage 334]", "[source: passage 12, 13]", "(source: passage 7)", etc.
_CITE_RE = re.compile(r"passage\s+(\d+)", re.IGNORECASE)


def _load_dotenv() -> None:
    """Load .env (same approach as evaluate.py) so ANTHROPIC_API_KEY is present."""
    env_path = REPO_ROOT / ".env"
    if not env_path.exists():
        return
    for line in env_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = (part.strip() for part in line.split("=", 1))
        if value and not os.environ.get(key):
            os.environ[key] = value


def _load_passages() -> dict[int, str]:
    """Map every ``passage NNN`` id in the corpus to its text (for judge grounding)."""
    passages: dict[int, str] = {}
    header = re.compile(r"^##\s+passage\s+(\d+)\s*$", re.IGNORECASE)
    for path in sorted(CORPUS_DIR.glob("*.md")):
        current: int | None = None
        buf: list[str] = []
        for line in path.read_text(encoding="utf-8").splitlines():
            m = header.match(line)
            if m:
                if current is not None:
                    passages[current] = "\n".join(buf).strip()
                current = int(m.group(1))
                buf = []
            elif current is not None:
                buf.append(line)
        if current is not None:
            passages[current] = "\n".join(buf).strip()
    return passages


def _cited_passage_ids(answer: str) -> list[int]:
    """Ordered, de-duplicated passage ids the agent cited in its answer."""
    seen: set[int] = set()
    out: list[int] = []
    for m in _CITE_RE.finditer(answer):
        pid = int(m.group(1))
        if pid not in seen:
            seen.add(pid)
            out.append(pid)
    return out


def _cited_chunks(passage_ids: list[int], passages: dict[int, str]) -> list[Chunk]:
    """Reconstruct the agent's grounding context from its citations, as Chunks.

    Mirrors the chunk-id convention the golden set uses (``passage:N:recursive:0``) so the
    judge sees the same kind of context the pipeline feeds it.
    """
    chunks: list[Chunk] = []
    for pid in passage_ids:
        text = passages.get(pid)
        if text is None:
            continue
        chunks.append(
            Chunk(
                id=f"passage:{pid}:recursive:0",
                text=text,
                metadata={"passage_id": pid},
            )
        )
    return chunks


def _agent_env() -> dict[str, str]:
    """Environment for the agent subprocess — deliberately WITHOUT the API key.

    The Agent SDK drives the ``claude`` CLI. If ``ANTHROPIC_API_KEY`` / ``ANTHROPIC_AUTH_TOKEN``
    is present, the CLI authenticates against the metered API and bills per-token. We strip
    both so the CLI falls back to the subscription OAuth credentials from ``claude /login``
    (stored in the OS keychain) — i.e. the agent runs on the user's Claude subscription, not
    the API. The judge still uses the API key (it's a separate ``anthropic.Anthropic()`` call
    in this process, untouched by this env).
    """
    env = dict(os.environ)
    env.pop("ANTHROPIC_API_KEY", None)
    env.pop("ANTHROPIC_AUTH_TOKEN", None)
    return env


async def _ask_agent(query_text: str, model_id: str) -> tuple[str, float, dict]:
    """Run one golden query through the agent; return (answer, latency_ms, usage)."""
    options = ClaudeAgentOptions(
        model=model_id,
        cwd=str(DATA_DIR),  # agent investigates the data/ folder (full reins)
        system_prompt=AGENT_SYSTEM,
        permission_mode="bypassPermissions",  # read-only investigation, no prompts
        max_turns=30,
        env=_agent_env(),  # subscription auth (no API key) — see _agent_env()
    )

    answer_parts: list[str] = []
    usage: dict = {}
    t0 = time.perf_counter()
    async for message in query(prompt=query_text, options=options):
        if isinstance(message, AssistantMessage):
            for block in message.content:
                if isinstance(block, TextBlock):
                    answer_parts.append(block.text)
        elif isinstance(message, ResultMessage):
            usage = {
                "cost_usd": getattr(message, "total_cost_usd", None),
                "num_turns": getattr(message, "num_turns", None),
                "duration_ms": getattr(message, "duration_ms", None),
            }
    latency_ms = (time.perf_counter() - t0) * 1000.0
    # The final assistant text block(s) carry the answer; keep the last non-empty turn.
    answer = answer_parts[-1].strip() if answer_parts else ""
    return answer, latency_ms, usage


async def _run_model(
    label: str,
    model_id: str,
    golden: list,
    passages: dict[int, str],
    judge: AnthropicJudge,
    n: int,
) -> dict:
    """Run the whole golden set through one model tier and score every row."""
    results: list[QueryResult] = []
    raw_rows: list[dict] = []

    for i, row in enumerate(golden, start=1):
        answer, latency_ms, usage = await _ask_agent(row.query, model_id)
        cited = _cited_passage_ids(answer)
        cited_ids = [f"passage:{pid}:recursive:0" for pid in cited]
        cited_chunks = _cited_chunks(cited, passages)

        # Substring answer check (same as pipeline). Empty answer => not measured.
        answer_correct = answer_contains(answer, row.expected_answer) if answer else None

        # Citation-based retrieval signals (agentic analogue of recall@k / precision@n):
        # how well did the passages the agent *chose to cite* match the expected ones?
        if cited_ids:
            cite_recall = recall_at_k(cited_ids, row.expected_chunk_ids, len(cited_ids))
            cite_precision = precision_at_n(cited_ids, row.expected_chunk_ids, n)
        else:
            cite_recall = 0.0
            cite_precision = 0.0

        result = QueryResult(
            query=row.query,
            recall_at_k=cite_recall,       # citation recall (best-effort agentic analogue)
            mrr=0.0,                        # no ranked list in agentic retrieval; left 0
            precision_at_n=cite_precision,  # precision of cited passages
            answer_correct=answer_correct,
            query_latency_ms=latency_ms,
            generate_ms=latency_ms,         # the whole turn is the agent's "generation"
        )

        # Judge against the agent's cited passages (its self-reported grounding context).
        score = judge.score(row.query, cited_chunks, row.expected_answer, answer)
        result.faithfulness = score.faithfulness
        result.answer_relevance = score.answer_relevance

        results.append(result)
        raw = result_to_dict(result)
        raw["answer"] = answer
        raw["cited_passages"] = cited
        raw["usage"] = usage
        raw_rows.append(raw)

        ac = "—" if answer_correct is None else ("Y" if answer_correct else "N")
        print(
            f"  [{label} {i:>2}/{len(golden)}] ans={ac} "
            f"faith={score.faithfulness:.2f} rel={score.answer_relevance:.2f} "
            f"cites={cited or '∅'} lat={latency_ms:.0f}ms",
            flush=True,
        )

    summary = aggregate(results)
    return {"summary": summary, "results": raw_rows}


async def _main_async(args: argparse.Namespace) -> int:
    _load_dotenv()
    # The AGENT runs on the subscription (keychain OAuth, see _agent_env) — no API key
    # needed for it. The JUDGE is a direct Messages API call and DOES need the key.
    if not os.environ.get("ANTHROPIC_API_KEY"):
        print(
            "ANTHROPIC_API_KEY not set — needed for the LLM judge "
            "(faithfulness/answer_relevance). The agent itself uses your Claude "
            "subscription via the claude CLI login, not this key.",
            file=sys.stderr,
        )
        return 1

    golden = load_golden(args.golden)
    if args.limit:
        golden = golden[: args.limit]
    passages = _load_passages()
    judge = AnthropicJudge()

    selected = args.models or list(MODELS)
    unknown = [m for m in selected if m not in MODELS]
    if unknown:
        print(f"unknown model(s): {unknown}. choose from {list(MODELS)}", file=sys.stderr)
        return 1

    variants: dict[str, dict] = {}
    for label in selected:
        model_id = MODELS[label]
        print(f"\n=== agent run: {label} ({model_id}) on {len(golden)} queries ===")
        run = await _run_model(label, model_id, golden, passages, judge, args.n)

        run_dir = OUT_DIR / f"agent_{label}"
        run_dir.mkdir(parents=True, exist_ok=True)
        payload = {
            "experiment": "agentic_rag",
            "model": model_id,
            "n": args.n,
            "summary": run["summary"],
            "results": run["results"],
        }
        (run_dir / "results.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        print(f"  wrote {run_dir / 'results.json'}")

        variants[f"agent_{label}"] = {
            "model": model_id,
            "eval_flags": ["--judge"],
            "summary": run["summary"],
        }
        _print_summary(label, run["summary"])

    DOCS_JSON.parent.mkdir(parents=True, exist_ok=True)
    DOCS_JSON.write_text(
        json.dumps({"variants": variants, "n": args.n}, indent=2) + "\n", encoding="utf-8"
    )
    print(f"\nwrote aggregated {DOCS_JSON}")
    return 0


def _print_summary(label: str, summary: dict) -> None:
    print(f"  --- {label} aggregate ---")
    for key in (
        "answer_correct",
        "faithfulness",
        "answer_relevance",
        "recall_at_k",
        "precision_at_n",
        "query_latency_ms",
    ):
        if key in summary:
            print(f"    {key}: {summary[key]:.3f}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--golden", type=Path, default=DEFAULT_GOLDEN)
    parser.add_argument(
        "--models",
        nargs="*",
        default=None,
        help=f"subset of {list(MODELS)} (default: all three)",
    )
    parser.add_argument("--n", type=int, default=8, help="top-n for citation precision@n")
    parser.add_argument("--limit", type=int, default=None, help="cap queries (smoke test)")
    args = parser.parse_args()
    return asyncio.run(_main_async(args))


if __name__ == "__main__":
    raise SystemExit(main())
