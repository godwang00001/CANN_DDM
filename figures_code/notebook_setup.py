from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path


def find_repo_root(start: Path | None = None) -> Path:
    start_path = Path.cwd().resolve() if start is None else Path(start).resolve()
    for candidate in [start_path, *start_path.parents]:
        if (candidate / "rate_model_core").is_dir() and (candidate / "figures_code").is_dir():
            return candidate
    raise RuntimeError("Could not locate the repository root from the current working directory.")


REPO_ROOT = find_repo_root()

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "matplotlib-cann-ddm"))

try:
    get_ipython().run_line_magic("config", "InlineBackend.figure_format = 'retina'")
except Exception:
    pass
