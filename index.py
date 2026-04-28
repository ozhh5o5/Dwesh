"""Alternative Vercel FastAPI entrypoint.

Vercel supports FastAPI detection from app.py/index.py/server.py. Keeping this as
an alias makes deployments more resilient across configuration changes.
"""

from backend.main import app
