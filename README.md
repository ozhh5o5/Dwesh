# DWESH

DWESH (**D**etect, **W**eigh, **E**valuate, **S**olve **H**idden bias) is an end-to-end platform for auditing and stress-testing AI systems used in high‑stakes decisions (hiring, lending, healthcare).

This repository is prepared for **Google Solution Challenge 2026** submission.

## What DWESH does

- **Audit fairness**: computes core fairness metrics and highlights policy violations.
- **Explain results**: generates clear, stakeholder-friendly narratives (Gemini-assisted when configured).
- **Mitigate bias**: suggests interventions and visualizes before/after impact.
- **Simulate outcomes**: explores longer-term effects via DecisionTwin-style simulators (LPS, parallel universes, cost analysis).
- **Governance**: maintains an integrity-style audit trail of actions and outputs.

## Key features in the UI

- Fairness Audit (domain presets + CSV upload)
- Bias Heatmap + dashboard summaries
- Report Card (A–F grade + export/print)
- Shadow AI Detector (text scan for undisclosed LLM usage)
- Benchmark page (compare model families)
- Drift Monitoring and Audit Trail views

## Architecture

- **Frontend**: single-page dashboard in [frontend/index.html](frontend/index.html)
- **Backend**: FastAPI service in [backend/main.py](backend/main.py)
- **Sample data**: small curated CSVs in [frontend/sample_data](frontend/sample_data)

The backend serves the frontend at `/`, so you can run everything from one server.

## Run locally (Windows/macOS/Linux)

### 1) Backend

From the repo root:

- Create and activate a virtual environment
- Install dependencies
- Start the server

Commands:

- `python -m venv .venv`
- Windows PowerShell: `.\.venv\Scripts\Activate.ps1`
- `pip install -r requirements.txt`
- `python backend/main.py`

Open:

- `http://127.0.0.1:8000`

### 2) Optional: MongoDB

DWESH can use MongoDB for persistence.

- Set `MONGO_URL` (default: `mongodb://localhost:27017`)
- Set `DB_NAME` (default in code)

You can place these in `backend/.env` (ignored by git).

## Deploy on Vercel (frontend + backend)

DWESH can be deployed to **a single Vercel project**:

- Static SPA served at `/` from `frontend/index.html`
- FastAPI served from a single Vercel Function (entrypoint: `app.py`)
- Sample CSVs served at `/sample_data/*` from `frontend/sample_data/*`

### 1) Vercel project settings

When importing this repo into Vercel, set the **Root Directory** to:

- `.` (leave blank)

Vercel will use `vercel.json` for routing.

### 2) Environment variables (recommended)

Serverless deployments do not reliably preserve in-memory state between requests.
For features that refer back to a `run_id` (e.g., `/api/report/{run_id}`), set a MongoDB instance (MongoDB Atlas works well):

- `MONGO_URL` = your Mongo connection string
- `DB_NAME` = (optional) database name (default: `fairforge_arena`)

### 3) Deploy

Option A — GitHub integration (recommended):

- Push to GitHub and import the project in the Vercel dashboard.

Option B — Vercel CLI:

- `npm i -g vercel`
- `vercel login`
- `cd GDP--Hackathon-main/GDP--Hackathon-main`
- `vercel`

### 4) Verify

After deploy:

- Open `/` to load the UI
- Check `/api/health` for API status

### Notes / limitations on Vercel

- Vercel Functions are **serverless**: long-running/background jobs and in-memory training state are not guaranteed to persist.
- This repo's `requirements.txt` is intentionally kept Vercel-friendly; large GPU/torch RL dependencies are not included.

## Repository hygiene (Solution Challenge)

This repo is set up to avoid committing generated artifacts:

- Ignores `node_modules/`, virtual environments, Playwright reports
- Ignores generated decks/docs (`*.pptx`, `*.pdf`)
- Ignores datasets by default (`*.csv`, `*.xlsx`) while keeping curated demo samples in [frontend/sample_data](frontend/sample_data)
## More details

- Technical deep-dive: [DOCUMENTATION.md](DOCUMENTATION.md)

