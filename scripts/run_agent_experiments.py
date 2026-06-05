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
- ``faithfulness`` + ``answer_relevance`` — the same judge rubric (claude-haiku-4-5),
  but run through the Agent SDK on the subscription (``SubscriptionJudge``) so the whole
  experiment stays off the metered API.

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

Auth — the WHOLE experiment runs on your Claude subscription, nothing on the metered API:

- the **agent** runs via the Agent SDK, which drives the ``claude`` CLI and uses the OAuth
  login from ``claude /login`` (stored in the OS keychain). We strip ``ANTHROPIC_API_KEY``
  from the agent subprocess env (``_agent_env``) so it never falls back to the API;
- the **judge** (``SubscriptionJudge``) scores via a one-shot SDK ``query`` too, same
  stripped env — so it is on the subscription as well. (The old API-key judge,
  ``rag.eval.judge.AnthropicJudge``, is unused here.)

So: just be logged in via ``claude /login``. No ANTHROPIC_API_KEY needed. Needs the
``claude`` CLI on PATH.
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
from typing import Any

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
    QueryResult,
    aggregate,
    load_golden,
    result_to_dict,
)
from rag.eval.judge import JUDGE_SYSTEM, JudgeScore, _parse_score  # noqa: E402
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

# Retry policy for the agent turn — the subscription rate-limits (Opus on Pro especially),
# surfaced as an SDK exception. Retry with exponential backoff before giving up on a row.
AGENT_MAX_RETRIES = 4
AGENT_RETRY_BACKOFF_S = 8.0

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


class SubscriptionJudge:
    """LLM-as-judge that runs on the Claude subscription, not the metered API.

    Same rubric as ``rag.eval.judge.AnthropicJudge`` (it reuses ``JUDGE_SYSTEM`` and
    ``_parse_score``), but it scores via a one-shot Agent SDK ``query`` instead of a direct
    Messages API call — so the judge, like the agent, runs on the ``claude`` CLI login
    (subscription) with no API key. This keeps the whole experiment on the sub and off the
    metered API. Model defaults to haiku (cheap, the scoring task is simple). No tools: the
    judge only reasons over the text it is handed.
    """

    def __init__(self, model: str = "claude-haiku-4-5") -> None:
        self.model = model

    async def score(
        self,
        query_text: str,
        chunks: list[Chunk],
        expected_answer: str,
        answer: str,
    ) -> JudgeScore:
        context = "\n".join(f"- {c.text}" for c in chunks)
        user = (
            f"{JUDGE_SYSTEM}\n\n"
            f"Question:\n{query_text}\n\n"
            f"Retrieved chunks:\n{context}\n\n"
            f"Expected answer:\n{expected_answer}\n\n"
            f"Generated answer:\n{answer}"
        )
        options = ClaudeAgentOptions(
            model=self.model,
            system_prompt=JUDGE_SYSTEM,
            allowed_tools=[],            # judge reasons over text only — no investigation
            permission_mode="bypassPermissions",
            max_turns=1,
            env=_agent_env(),            # subscription auth (no API key)
        )
        parts: list[str] = []
        async for message in query(prompt=user, options=options):
            if isinstance(message, AssistantMessage):
                for block in message.content:
                    if isinstance(block, TextBlock):
                        parts.append(block.text)
        return _parse_score("".join(parts))


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

    # The subscription is rate-limited (notably Opus on Pro). The SDK surfaces this as an
    # exception on a subsequent call, which would otherwise abort the whole tier. Retry the
    # turn with exponential backoff; on persistent failure return an empty answer flagged
    # via usage["error"] so the row is recorded as unanswered, not lost.
    t0 = time.perf_counter()
    last_error: str | None = None
    for attempt in range(AGENT_MAX_RETRIES):
        answer_parts: list[str] = []
        usage: dict = {}
        try:
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
            answer = answer_parts[-1].strip() if answer_parts else ""
            return answer, latency_ms, usage
        except Exception as exc:  # noqa: BLE001 - retry transient rate-limit / SDK errors
            last_error = f"{type(exc).__name__}: {exc}"
            if attempt < AGENT_MAX_RETRIES - 1:
                await asyncio.sleep(AGENT_RETRY_BACKOFF_S * (2**attempt))

    latency_ms = (time.perf_counter() - t0) * 1000.0
    return "", latency_ms, {"error": last_error}


async def _score_row(
    row: Any,
    *,
    model_id: str,
    passages: dict[int, str],
    judge: SubscriptionJudge,
    n: int,
) -> tuple[QueryResult, dict]:
    """Run one golden row through the agent + judge; return (result, raw_dict).

    Both the agent and the (subscription) judge are async SDK calls, so concurrent rows
    overlap naturally on the event loop. A judge failure on one row is caught and recorded
    as unscored (faithfulness/relevance left None) rather than aborting the whole tier.
    """
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
    # A judge failure (parse error, transient SDK error) must not abort the whole tier —
    # record the row as unscored and move on.
    judge_error: str | None = None
    try:
        score = await judge.score(row.query, cited_chunks, row.expected_answer, answer)
        result.faithfulness = score.faithfulness
        result.answer_relevance = score.answer_relevance
    except Exception as exc:  # noqa: BLE001 - one bad judge call shouldn't kill the run
        judge_error = f"{type(exc).__name__}: {exc}"

    raw = result_to_dict(result)
    raw["answer"] = answer
    raw["cited_passages"] = cited
    raw["usage"] = usage
    if judge_error:
        raw["judge_error"] = judge_error
    return result, raw


async def _run_model(
    label: str,
    model_id: str,
    golden: list,
    passages: dict[int, str],
    judge: SubscriptionJudge,
    n: int,
    concurrency: int,
    query_delay: float = 0.0,
) -> dict:
    """Run the whole golden set through one model tier, ``concurrency`` rows at a time.

    A semaphore caps how many agent turns run concurrently. ``gather`` preserves input
    order, so the persisted ``results`` list stays in golden order regardless of which
    rows finish first. Progress lines print as each row *completes* (so they may be
    out of order on screen), each tagged with its golden index.

    ``query_delay`` staggers row starts by ``delay * index`` seconds — used to pace a
    rate-limited tier (e.g. Opus on Pro) well under the limit by spreading calls out in
    time even at low concurrency.
    """
    total = len(golden)
    sem = asyncio.Semaphore(concurrency)

    async def _one(i: int, row: Any) -> tuple[QueryResult, dict]:
        if query_delay:
            await asyncio.sleep(query_delay * (i - 1))  # stagger starts (i is 1-based)
        async with sem:
            result, raw = await _score_row(
                row,
                model_id=model_id,
                passages=passages,
                judge=judge,
                n=n,
            )
        ac = "—" if result.answer_correct is None else ("Y" if result.answer_correct else "N")
        faith = "ERR" if result.faithfulness is None else f"{result.faithfulness:.2f}"
        rel = "ERR" if result.answer_relevance is None else f"{result.answer_relevance:.2f}"
        print(
            f"  [{label} {i:>2}/{total}] ans={ac} "
            f"faith={faith} rel={rel} "
            f"cites={raw['cited_passages'] or '∅'} lat={result.query_latency_ms:.0f}ms",
            flush=True,
        )
        return result, raw

    pairs = await asyncio.gather(
        *(_one(i, row) for i, row in enumerate(golden, start=1))
    )
    results = [r for r, _ in pairs]   # in golden order (gather preserves order)
    raw_rows = [raw for _, raw in pairs]

    summary = aggregate(results)
    return {"summary": summary, "results": raw_rows}


async def _main_async(args: argparse.Namespace) -> int:
    # Both the agent and the judge run on the subscription (see _agent_env / module
    # docstring); no API key is required. We still load .env for any other settings, but it
    # is not an error for ANTHROPIC_API_KEY to be absent.
    _load_dotenv()

    golden = load_golden(args.golden)
    if args.limit:
        golden = golden[: args.limit]
    passages = _load_passages()
    judge = SubscriptionJudge()

    selected = args.models or list(MODELS)
    unknown = [m for m in selected if m not in MODELS]
    if unknown:
        print(f"unknown model(s): {unknown}. choose from {list(MODELS)}", file=sys.stderr)
        return 1

    # Merge into any existing aggregated file so running a SUBSET of models (e.g.
    # --models sonnet opus) preserves tiers saved by an earlier run (e.g. haiku).
    variants: dict[str, dict] = {}
    if DOCS_JSON.exists():
        try:
            variants = json.loads(DOCS_JSON.read_text(encoding="utf-8")).get("variants", {})
        except (json.JSONDecodeError, OSError):
            variants = {}

    for label in selected:
        model_id = MODELS[label]
        print(
            f"\n=== agent run: {label} ({model_id}) on {len(golden)} queries "
            f"(concurrency={args.concurrency}) ==="
        )
        run = await _run_model(
            label,
            model_id,
            golden,
            passages,
            judge,
            args.n,
            args.concurrency,
            args.query_delay,
        )

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
    parser.add_argument(
        "--concurrency",
        type=int,
        default=3,
        help="how many agent queries to run at once per tier (default: 3)",
    )
    parser.add_argument(
        "--query-delay",
        type=float,
        default=0.0,
        help="seconds to stagger each query's start by (paces a rate-limited tier; "
        "e.g. --query-delay 30 --concurrency 1 for Opus on Pro)",
    )
    args = parser.parse_args()
    return asyncio.run(_main_async(args))


if __name__ == "__main__":
    raise SystemExit(main())
