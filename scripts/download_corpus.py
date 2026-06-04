"""Download the raw rag-mini-wikipedia benchmark parquet files.

Self-contained so a third party can fetch the corpus from a clean checkout:

    uv run python scripts/download_corpus.py

Downloads two parquet files from the Hugging Face dataset
`rag-datasets/rag-mini-wikipedia` (CC-BY-3.0) into `data/raw/`:

    data/raw/passages.parquet   3,200 Wikipedia passages  {passage, id}
    data/raw/test.parquet         918 question/answer rows {question, answer, id}

It does NOT convert anything to Markdown — that is `prepare_data.py`'s job. Run this
first, then `prepare_data.py`, or just run `prepare_data.py` which calls this for you.
"""

from __future__ import annotations

import sys
import urllib.request
from pathlib import Path

# Resolve to data/raw/ relative to the repo root (this file lives in scripts/).
REPO_ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = REPO_ROOT / "data" / "raw"

_HF_BASE = (
    "https://huggingface.co/datasets/rag-datasets/rag-mini-wikipedia/"
    "resolve/main/data"
)

# remote path -> local filename
FILES = {
    "passages.parquet/part.0.parquet": "passages.parquet",
    "test.parquet/part.0.parquet": "test.parquet",
}

_PARQUET_MAGIC = b"PAR1"


def _download(remote_rel: str, dest: Path) -> None:
    url = f"{_HF_BASE}/{remote_rel}"
    print(f"  downloading {url}")
    print(f"          -> {dest}")
    with urllib.request.urlopen(url) as resp:  # noqa: S310 (trusted HF host)
        data = resp.read()
    if data[:4] != _PARQUET_MAGIC:
        raise RuntimeError(
            f"{url} did not return a parquet file "
            f"(first bytes: {data[:8]!r}). The dataset layout may have changed."
        )
    dest.write_bytes(data)
    print(f"          ok ({len(data):,} bytes)")


def download_corpus(force: bool = False) -> dict[str, Path]:
    """Download the parquet files into data/raw/. Returns {name: path}.

    Skips files that already exist unless `force` is True.
    """
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    out: dict[str, Path] = {}
    for remote_rel, local_name in FILES.items():
        dest = RAW_DIR / local_name
        if dest.exists() and not force:
            print(f"  {dest} already present, skipping (use --force to re-download)")
        else:
            _download(remote_rel, dest)
        out[local_name] = dest
    return out


def main() -> int:
    force = "--force" in sys.argv
    print("Downloading rag-mini-wikipedia corpus (CC-BY-3.0)...")
    download_corpus(force=force)
    print("Done. Next: uv run python scripts/prepare_data.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
