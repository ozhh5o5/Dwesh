"""Vercel FastAPI entrypoint.

Vercel's FastAPI framework detection looks for a top-level module exporting an
ASGI app named `app` (e.g. app.py, index.py, server.py).

The actual application lives in backend/main.py.
"""

from backend.main import app
