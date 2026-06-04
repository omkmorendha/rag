"""Load config.yaml — the single source of truth for which strategy each stage uses.

Both ingest and query call `load_config()` so the stage selection (especially the
embedder, ARCHITECTURE.md §2) cannot drift between the two pipelines.
"""

from __future__ import annotations

from pathlib import Path
 
import yaml
from dotenv import load_dotenv

REPO_ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = REPO_ROOT / "config.yaml"

# Load .env once, at import time, so os.environ (e.g. ANTHROPIC_API_KEY) is populated
# before any stage reads it. `uv run` does not auto-load .env. .env is gitignored.
# override=True so the file wins over a stale/empty shell var of the same name.
load_dotenv(REPO_ROOT / ".env", override=True)


def load_config(path: Path | str = CONFIG_PATH) -> dict:
    """Parse config.yaml into a plain dict of stage -> {name, **params}."""
    with open(path, encoding="utf-8") as f:
        config = yaml.safe_load(f)
    if not isinstance(config, dict):
        raise ValueError(f"{path}: expected a mapping at the top level, got {type(config)}")
    return config
