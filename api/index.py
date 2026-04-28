from __future__ import annotations

import sys
from pathlib import Path

# Ensure the Vercel function can import the local backend package.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Vercel expects a top-level ASGI app named `app`.
from backend.main import app  # noqa: E402
