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



## More details

- Technical deep-dive: [DOCUMENTATION.md](DOCUMENTATION.md)

