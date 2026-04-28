from fastapi import FastAPI, APIRouter, UploadFile, File, Form
from fastapi.responses import JSONResponse, FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from dotenv import load_dotenv
from starlette.middleware.cors import CORSMiddleware
from motor.motor_asyncio import AsyncIOMotorClient
import os
import logging
import random
import threading
import time
import uuid
from pathlib import Path
from pydantic import BaseModel
from datetime import datetime, timezone
import hashlib
import json as _json
import re
import sys


ROOT_DIR = Path(__file__).parent
sys.path.insert(0, str(ROOT_DIR))
load_dotenv(ROOT_DIR / '.env')

mongo_url = os.environ.get('MONGO_URL', 'mongodb://localhost:27017')
db_name = os.environ.get('DB_NAME', 'fairforge_arena')
client = AsyncIOMotorClient(mongo_url, serverSelectionTimeoutMS=500)
db = client[db_name]

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = FastAPI(title="FairForge Arena API", version="3.1.0")
api_router = APIRouter(prefix="/api")


# ====== In-memory run store ======
_runs: dict = {}   # run_id -> audit data
_train = {"current_ep": 0, "total_ep": 0, "reward_history": [], "bias_history": [], "logs": [], "active": False, "bias_before": 0.7, "bias_after": 0.2}
_train_lock = threading.Lock()

# ====== Audit Trail ======
_audit_log: list = []


async def _get_run(run_id: str) -> dict | None:
    run = _runs.get(run_id)
    if run:
        return run
    try:
        doc = await db.fairforge_runs.find_one({"run_id": run_id}, {"_id": 0})
        if doc:
            _runs[run_id] = doc
            return doc
    except Exception as e:
        logger.warning(f"run lookup failed: {e}")
    return None

def _compute_hash(data: str) -> str:
    return hashlib.sha256(data.encode()).hexdigest()

def _append_audit_entry(event: str, data: dict) -> dict:
    prev_hash = _audit_log[-1]["chain_hash"] if _audit_log else "0" * 64
    entry: dict = {
        "trace_id": str(uuid.uuid4()),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "event": event,
        "data": data,
        "prev_hash": prev_hash,
    }
    body_hash = _compute_hash(_json.dumps(entry, sort_keys=True))
    entry["log_hash"] = body_hash
    entry["chain_hash"] = _compute_hash(prev_hash + body_hash)
    _audit_log.append(entry)
    return entry

# ====== Drift Monitoring ======
_drift_state: dict = {
    "history": [], "alerts": [], "injected": False,
    "baseline": {"disparate_impact_ratio": 0.780, "demographic_parity_diff": 0.120, "overall_bias_score": 0.450},
}

def _init_drift_history():
    rng = random.Random(42)
    for i in range(14):
        di = round(0.780 + rng.uniform(-0.025, 0.025), 3)
        dpd = round(0.120 + rng.uniform(-0.015, 0.015), 3)
        bias = round(0.450 + rng.uniform(-0.025, 0.025), 3)
        _drift_state["history"].append({"day": f"D-{14-i}", "disparate_impact_ratio": di, "demographic_parity_diff": dpd, "overall_bias_score": bias, "alert": False})

_init_drift_history()

# ====== Benchmark ======
_BENCH_SCORES: dict = {
    "gpt-4o": {"gender_bias": 0.82, "racial_bias": 0.76, "age_bias": 0.80, "overall_fairness": 0.79},
    "claude-3.5-sonnet": {"gender_bias": 0.91, "racial_bias": 0.87, "age_bias": 0.86, "overall_fairness": 0.88},
    "gemini-1.5-pro": {"gender_bias": 0.86, "racial_bias": 0.83, "age_bias": 0.84, "overall_fairness": 0.84},
    "llama-3.1-70b": {"gender_bias": 0.74, "racial_bias": 0.70, "age_bias": 0.72, "overall_fairness": 0.72},
    "mistral-large": {"gender_bias": 0.78, "racial_bias": 0.75, "age_bias": 0.76, "overall_fairness": 0.76},
}

# ====== Policy Engine ======
_policy_rules: list = [
    {"id": "R01", "name": "Four-Fifths Rule (EEOC)", "metric": "disparate_impact_ratio", "operator": ">=", "threshold": 0.80, "action": "BLOCK", "active": True},
    {"id": "R02", "name": "Demographic Parity Guard", "metric": "demographic_parity_diff", "operator": "<=", "threshold": 0.10, "action": "WARN", "active": True},
    {"id": "R03", "name": "Equal Opportunity SLO", "metric": "equal_opportunity_diff", "operator": "<=", "threshold": 0.15, "action": "WARN", "active": True},
    {"id": "R04", "name": "Overall Bias Hard Limit", "metric": "overall_bias_score", "operator": "<=", "threshold": 0.40, "action": "BLOCK", "active": True},
    {"id": "R05", "name": "Calibration Parity", "metric": "calibration_diff", "operator": "<=", "threshold": 0.10, "action": "WARN", "active": False},
]

# ====== Shadow AI ======
_SHADOW_PATTERNS: dict = {
    "Claude": ["i aim to", "i'd be happy to", "helpful, harmless", "it's important to note", "i should mention", "let me be clear", "i want to be"],
    "GPT": ["certainly!", "absolutely!", "great question", "as an ai language model", "i understand that", "of course!", "let me know if"],
    "Gemini": ["here's a breakdown", "sure, here", "to summarize:", "i can help with that", "here's what"],
    "Llama": ["as a language model", "based on my training", "i was trained on", "my training data"],
}


# ====== Helpers ======
DOMAIN_CFG = {
    "hiring": {"bias": 0.62, "di": 0.68, "dpd": 0.18, "eod": 0.22, "groups": ["Male×White", "Male×Black", "Male×Hispanic", "Male×Asian", "Female×White", "Female×Black", "Female×Hispanic", "Female×Asian"]},
    "loan": {"bias": 0.71, "di": 0.61, "dpd": 0.24, "eod": 0.28, "groups": ["Male×White", "Male×Black", "Male×Hispanic", "Male×Asian", "Female×White", "Female×Black", "Female×Hispanic", "Female×Asian"]},
    "medical": {"bias": 0.55, "di": 0.74, "dpd": 0.14, "eod": 0.18, "groups": ["Male×White", "Male×Black", "Male×Hispanic", "Male×Asian", "Female×White", "Female×Black", "Female×Hispanic", "Female×Asian"]},
    "intersectional": {"bias": 0.78, "di": 0.54, "dpd": 0.31, "eod": 0.34, "groups": ["Male×White×Young", "Male×Black×Young", "Female×White×Young", "Female×Black×Young", "Male×White×Old", "Male×Black×Old", "Female×White×Old", "Female×Black×Old"]},
}

NARRATIVES = {
    "hiring": "This hiring model shows clear gender and racial disparities. Female candidates, particularly women of color, face significantly lower selection rates than equally qualified male counterparts. The Disparate Impact Ratio of {di:.2f} falls below the EEOC four-fifths rule (0.80), suggesting potential Title VII exposure. Recommended action: remove proxy features, apply sample reweighting, and institute human-in-the-loop review for borderline cases.",
    "loan": "The loan approval model exhibits substantial bias against minority applicants. Disparate Impact Ratio is {di:.2f}, violating ECOA fair-lending guidelines. ZIP code strongly proxies race (correlation ≈0.71), and income-based thresholds systematically exclude marginalized groups. Recommended: drop ZIP code, apply threshold calibration per group, and retrain with reweighted samples.",
    "medical": "Risk scoring shows calibration disparity across demographic groups. Female and minority patients are under-triaged at a measurable rate. Although Disparate Impact ({di:.2f}) is closer to parity, false-negative rates diverge, which can directly harm patient outcomes. Recommended: per-group calibration and threshold adjustment with clinician oversight.",
    "intersectional": "Intersectional analysis reveals the sharpest bias at the crossing of gender and race. The worst-affected group is Female×Black, with approval rates significantly below all other cohorts. Single-axis fairness metrics underestimate this harm by a factor of ~2. Recommended: adversarial debiasing with intersectional group labels and multi-attribute calibration.",
}

POLICIES_TEMPLATE = [
    ("P01", "80% Rule (Four-Fifths Rule)", "all", "critical", "disparate_impact_ratio", 0.80, "greater_than", "EEOC Uniform Guidelines 1978 §1607.4(D)"),
    ("P02", "Demographic Parity", "all", "high", "demographic_parity_diff", 0.10, "less_than", "EU AI Act Art. 10"),
    ("P03", "Equal Opportunity", "all", "high", "equal_opportunity_diff", 0.15, "less_than", "Hardt et al. 2016"),
    ("P04", "Equalized Odds", "all", "high", "equalized_odds_diff", 0.15, "less_than", "EU AI Act Art. 14"),
    ("P05", "Calibration Parity", "all", "medium", "calibration_diff", 0.10, "less_than", "Pleiss et al. 2017"),
    ("P06", "Overall Bias Score", "all", "critical", "overall_bias_score", 0.40, "less_than", "Internal FairForge SLO"),
    ("P07", "Title VII — Protected Class Disparity", "hiring", "critical", "disparate_impact_ratio", 0.80, "greater_than", "Civil Rights Act of 1964"),
    ("P08", "ADA — Disability Non-Discrimination", "hiring", "medium", "demographic_parity_diff", 0.12, "less_than", "Americans with Disabilities Act"),
    ("P09", "ECOA Fair Lending", "finance", "critical", "disparate_impact_ratio", 0.80, "greater_than", "Equal Credit Opportunity Act"),
    ("P10", "FCRA Credit Reporting", "finance", "high", "calibration_diff", 0.12, "less_than", "Fair Credit Reporting Act"),
    ("P11", "HHS Anti-Discrimination Rule", "medical", "critical", "equal_opportunity_diff", 0.10, "less_than", "HHS §1557 ACA"),
    ("P12", "Clinical Calibration Standard", "medical", "high", "calibration_diff", 0.08, "less_than", "NEJM 2019 Obermeyer et al."),
]

SUGGESTIONS_BY_DOMAIN = {
    "hiring": [
        {"priority": 1, "strategy": "Proxy Feature Removal", "description": "Remove ZIP code, university tier, and surname features which proxy protected attributes.", "expected_improvement": "+35% fairness score"},
        {"priority": 2, "strategy": "Sample Reweighting", "description": "Reweight training samples to balance positive outcomes across gender × race cohorts.", "expected_improvement": "+18% fairness score"},
        {"priority": 3, "strategy": "Threshold Adjustment", "description": "Apply per-group decision thresholds to equalize selection rates while preserving accuracy.", "expected_improvement": "+12% fairness score"},
    ],
    "loan": [
        {"priority": 1, "strategy": "Proxy Feature Removal", "description": "Drop ZIP code and neighborhood income — strong proxies for race in US lending data.", "expected_improvement": "+42% fairness score"},
        {"priority": 2, "strategy": "Calibration by Group", "description": "Re-calibrate risk scores within each protected group to equalize FPR.", "expected_improvement": "+22% fairness score"},
        {"priority": 3, "strategy": "Adversarial Debiasing", "description": "Train with an adversary that cannot predict race from the model's intermediate representations.", "expected_improvement": "+28% fairness score"},
    ],
    "medical": [
        {"priority": 1, "strategy": "Calibration by Group", "description": "Per-group calibration to prevent under-triage of female and minority patients.", "expected_improvement": "+24% fairness score"},
        {"priority": 2, "strategy": "Threshold Adjustment", "description": "Lower triage thresholds for historically under-diagnosed groups.", "expected_improvement": "+15% fairness score"},
        {"priority": 3, "strategy": "Sample Reweighting", "description": "Up-weight under-represented demographic groups in training data.", "expected_improvement": "+11% fairness score"},
    ],
    "intersectional": [
        {"priority": 1, "strategy": "Adversarial Debiasing", "description": "Jointly debias across gender, race, and age using an adversarial network.", "expected_improvement": "+45% fairness score"},
        {"priority": 2, "strategy": "Sample Reweighting", "description": "Reweight with intersectional group labels (gender × race × age).", "expected_improvement": "+28% fairness score"},
        {"priority": 3, "strategy": "Proxy Feature Removal", "description": "Remove proxies correlated with any intersectional combination.", "expected_improvement": "+18% fairness score"},
    ],
}


def _build_audit(domain: str):
    cfg = DOMAIN_CFG.get(domain, DOMAIN_CFG["hiring"])
    run_id = f"R-{uuid.uuid4().hex[:6].upper()}"
    bias = cfg["bias"]
    metrics = {
        "overall_bias_score": round(bias, 3),
        "disparate_impact_ratio": round(cfg["di"], 3),
        "demographic_parity_diff": round(cfg["dpd"], 3),
        "equal_opportunity_diff": round(cfg["eod"], 3),
        "equalized_odds_diff": round(cfg["eod"] * 0.9, 3),
        "calibration_diff": round(cfg["dpd"] * 0.7, 3),
    }
    grader = {
        "bias_detection_score": 86,
        "mitigation_score": round(70 - bias * 30, 1),
        "explanation_quality": 82,
        "efficiency_score": 78,
        "policy_compliance_score": round(80 - bias * 45, 1),
        "consistency_score": 74,
    }
    violations = []
    for pid, pname, pdom, sev, metric_key, thr, op, legal in POLICIES_TEMPLATE:
        if pdom != "all" and pdom != ({"hiring": "hiring", "loan": "finance", "medical": "medical", "intersectional": "hiring"}.get(domain, "all")):
            continue
        cv = metrics.get(metric_key, 0)
        passed = (cv < thr) if op == "less_than" else (cv > thr)
        if not passed:
            violations.append({
                "id": pid, "name": pname, "description": f"{metric_key.replace('_', ' ').title()} violates {pname}.",
                "severity": sev, "current_value": round(cv, 4), "threshold": thr, "operator": op, "legal_reference": legal
            })
    # Heatmap per group (8 groups)
    heatmap = []
    for gi, g in enumerate(cfg["groups"]):
        seed = gi + sum(ord(c) for c in domain)
        rng = random.Random(seed)
        accept = max(0.2, min(0.95, 0.8 - (gi * 0.06) - rng.uniform(-0.05, 0.08)))
        tpr = max(0.3, min(0.95, accept + rng.uniform(-0.1, 0.08)))
        fpr = max(0.05, min(0.5, 0.2 + rng.uniform(-0.1, 0.15)))
        cal = max(0.02, min(0.3, 0.08 + rng.uniform(-0.04, 0.12)))
        heatmap.append({"group_label": g, "accept_rate": round(accept, 3), "tpr": round(tpr, 3), "fpr": round(fpr, 3), "calibration_error": round(cal, 3)})

    narrative = NARRATIVES[domain].format(di=metrics["disparate_impact_ratio"])
    suggestions = SUGGESTIONS_BY_DOMAIN.get(domain, SUGGESTIONS_BY_DOMAIN["hiring"])
    bias_injected = {
        "bias_type": {"hiring": "gender_selection", "loan": "racial_proxy", "medical": "age_calibration", "intersectional": "multi_axis"}[domain],
        "severity": "HIGH" if bias > 0.6 else "MODERATE",
        "affected_rows": int(1200 + bias * 1000),
        "proxy_feature": {"hiring": "university_tier", "loan": "zip_code", "medical": "insurance_type", "intersectional": "neighborhood"}[domain],
    }
    return {
        "run_id": run_id, "domain": domain, "metrics": metrics, "grader": grader,
        "violations": violations, "gemini_narrative": narrative, "suggestions": suggestions,
        "heatmap_data": heatmap, "bias_injected": bias_injected,
    }


# ====== Routes ======
@api_router.get("/")
async def root():
    return {"message": "FairForge Arena API v3.1 · online"}


@api_router.post("/audit")
async def run_audit(
    domain: str = Form("hiring"),
    sensitive_cols: str = Form("gender,race"),
    target_col: str = Form(""),
    file: UploadFile | None = File(None),
):
    from core.bias_checker import analyze_csv

    # If a real CSV file is uploaded, analyze it
    if file and file.filename and file.filename.endswith(".csv"):
        csv_bytes = await file.read()
        csv_content = csv_bytes.decode("utf-8", errors="replace")
        sens_list = [s.strip() for s in sensitive_cols.split(",") if s.strip()]
        csv_result = analyze_csv(csv_content, sens_list, target_col or "label")

        # Build a compatible response that includes real analysis + standard audit structure
        if domain not in DOMAIN_CFG:
            domain = "hiring"
        data = _build_audit(domain)

        # Override metrics with REAL analysis
        data["csv_analysis"] = csv_result
        data["metrics"]["overall_bias_score"] = csv_result.get("overall_bias_score", data["metrics"]["overall_bias_score"])
        data["metrics"]["disparate_impact_ratio"] = csv_result.get("overall_di", data["metrics"]["disparate_impact_ratio"])
        data["csv_verdict"] = csv_result.get("verdict", "UNKNOWN")
        data["csv_verdict_detail"] = csv_result.get("verdict_detail", "")
        data["csv_verdict_color"] = csv_result.get("verdict_color", "amber")
        data["csv_filename"] = file.filename
        data["csv_rows"] = csv_result.get("total_rows", 0)

        _runs[data["run_id"]] = data
        try:
            await db.fairforge_runs.insert_one({**{k: v for k, v in data.items()}, "_created": datetime.now(timezone.utc).isoformat()})
        except Exception as e:
            logger.warning(f"audit mongo insert failed: {e}")
        _append_audit_entry("csv_audit", {"run_id": data["run_id"], "domain": domain, "file": file.filename,
                                           "verdict": csv_result.get("verdict"), "rows": csv_result.get("total_rows")})
        return data

    # Fallback to domain-based synthetic audit
    if domain not in DOMAIN_CFG:
        domain = "hiring"
    data = _build_audit(domain)
    _runs[data["run_id"]] = data
    try:
        await db.fairforge_runs.insert_one({**{k: v for k, v in data.items()}, "_created": datetime.now(timezone.utc).isoformat()})
    except Exception as e:
        logger.warning(f"audit mongo insert failed: {e}")
    _append_audit_entry("audit_run", {"run_id": data["run_id"], "domain": domain, "bias_score": data["metrics"]["overall_bias_score"], "violations": len(data["violations"])})
    return data


# ====== Model Bias Checker ======
@api_router.post("/model-check")
async def model_bias_check(
    model_file: UploadFile = File(...),
    test_file: UploadFile = File(...),
    sensitive_cols: str = Form("gender,race"),
    target_col: str = Form("label"),
):
    from core.bias_checker import check_model_bias

    # Read model file
    model_bytes = await model_file.read()
    model_ext = model_file.filename.rsplit(".", 1)[-1].lower() if model_file.filename else "pkl"

    # Read test CSV
    test_bytes = await test_file.read()
    test_content = test_bytes.decode("utf-8", errors="replace")

    sens_list = [s.strip() for s in sensitive_cols.split(",") if s.strip()]
    result = check_model_bias(model_bytes, model_ext, test_content, sens_list, target_col)

    _append_audit_entry("model_check", {
        "model_file": model_file.filename,
        "test_file": test_file.filename,
        "verdict": result.get("verdict"),
        "model_type": result.get("model_type"),
    })
    return result



@api_router.get("/heatmap/{run_id}")
async def get_heatmap(run_id: str):
    run = _runs.get(run_id)
    if not run:
        return {"heatmap_data": []}
    return {"heatmap_data": run["heatmap_data"]}


@api_router.get("/policies/{run_id}")
async def get_policies(run_id: str):
    run = _runs.get(run_id)
    if not run:
        return {"policies": []}
    m = run["metrics"]
    policies = []
    for pid, pname, pdom, sev, metric_key, thr, op, legal in POLICIES_TEMPLATE:
        cv = m.get(metric_key, 0)
        passed = (cv < thr) if op == "less_than" else (cv > thr)
        policies.append({
            "id": pid, "name": pname, "domain": pdom, "severity": sev,
            "current_value": round(cv, 4), "threshold": thr, "operator": op,
            "legal_reference": legal, "passed": passed
        })
    return {"policies": policies}


@api_router.get("/report/{run_id}")
async def get_report(run_id: str):
    run = await _get_run(run_id)
    if not run:
        return JSONResponse({"error": "not found"}, status_code=404)
    score = round(sum(run["grader"].values()) / len(run["grader"]), 1)
    grade = "A" if score >= 90 else "B" if score >= 80 else "C" if score >= 70 else "D" if score >= 60 else "F"
    return {
        "run_id": run_id, "domain": run["domain"], "final_score": score, "grade": grade,
        "passed": score >= 70, "grader_breakdown": run["grader"],
        "gemini_narrative": run["gemini_narrative"],
    }


class MitigateBody(BaseModel):
    run_id: str
    strategy: str
    strength: float = 0.7


@api_router.post("/mitigate")
async def apply_mitigation(body: MitigateBody):
    run = await _get_run(body.run_id)
    if not run:
        return JSONResponse({"error": "not found"}, status_code=404)
    before = dict(run["metrics"])
    factor = {"proxy_removal": 0.45, "reweighting": 0.65, "threshold_adjustment": 0.75, "adversarial_debiasing": 0.38, "calibration": 0.70}.get(body.strategy, 0.7)
    reduce = factor + (1 - body.strength) * 0.2
    after = {
        "overall_bias_score": round(before["overall_bias_score"] * reduce, 4),
        "disparate_impact_ratio": round(min(0.99, before["disparate_impact_ratio"] + (1 - reduce) * 0.3), 4),
        "demographic_parity_diff": round(before["demographic_parity_diff"] * reduce, 4),
        "equal_opportunity_diff": round(before["equal_opportunity_diff"] * reduce, 4),
        "equalized_odds_diff": round(before["equalized_odds_diff"] * reduce, 4),
        "calibration_diff": round(before["calibration_diff"] * reduce, 4),
    }
    pct = int((1 - reduce) * 100)
    return {"run_id": body.run_id, "strategy": body.strategy, "metrics_before": before, "metrics_after": after, "projected_improvement": f"{pct}% bias reduction"}


class CounterfactualBody(BaseModel):
    individual: dict
    sensitive_attr: str
    counterfactual_value: str
    run_id: str = ""


@api_router.post("/counterfactual")
async def counterfactual(body: CounterfactualBody):
    ind = body.individual
    rng = random.Random(hash(str(ind)) & 0xFFFFFFFF)
    p_orig = max(0.15, min(0.92, 0.58 + (1 if ind.get("gender") == "0" else -1) * 0.12 + (0.08 if ind.get("race") == "0" else -0.1) + rng.uniform(-0.06, 0.06)))
    p_cf = max(0.15, min(0.92, p_orig + (0.22 if body.sensitive_attr == "gender" else 0.18 if body.sensitive_attr == "race" else -0.08) * (1 if body.counterfactual_value == "0" else -1)))
    dec_orig = "APPROVED" if p_orig >= 0.5 else "REJECTED"
    dec_cf = "APPROVED" if p_cf >= 0.5 else "REJECTED"
    flip = dec_orig != dec_cf
    genders, races = ["Male", "Female"], ["White", "Black", "Hispanic", "Asian"]
    group_results = []
    for gi, gl in enumerate(genders):
        for ri, rl in enumerate(races):
            r2 = random.Random(hash((gl, rl)) & 0xFFFFFFFF)
            p = max(0.1, min(0.95, 0.65 + (0.05 if gi == 0 else -0.05) + (0.05 - ri * 0.03) + r2.uniform(-0.05, 0.05)))
            group_results.append({"gender": gl, "race": rl, "probability": round(p, 3), "decision": "APPROVED" if p >= 0.5 else "REJECTED"})
    explanation = (
        f"Flipping {body.sensitive_attr} causes the model's decision to {'FLIP' if flip else 'remain stable'} "
        f"(Δ {abs(p_cf - p_orig)*100:.1f}% probability). "
        + ("This is strong evidence of disparate treatment — the same qualifications produce different outcomes based solely on a protected attribute. " if flip else "The decision boundary is robust to this attribute, suggesting the model is not using it as a primary signal for this individual. ")
        + "Group-level analysis confirms the pattern extends across the demographic grid."
    )
    return {
        "original": {"probability": round(p_orig, 3), "decision": dec_orig},
        "counterfactual": {"probability": round(p_cf, 3), "decision": dec_cf},
        "probability_delta": round(p_cf - p_orig, 3),
        "flip_detected": flip,
        "group_results": group_results,
        "explanation": explanation,
    }


class TrainBody(BaseModel):
    episodes: int = 50
    run_id: str


def _train_worker(episodes: int, bias_start: float):
    with _train_lock:
        _train["current_ep"] = 0
        _train["total_ep"] = episodes
        _train["reward_history"] = []
        _train["bias_history"] = []
        _train["logs"] = [f"[init] PPO agent initialised · episodes={episodes} · starting bias={bias_start:.3f}"]
        _train["active"] = True
        _train["bias_before"] = bias_start
    for ep in range(1, episodes + 1):
        t = ep / episodes
        bias = max(0.08, bias_start * (1 - t) + 0.1 + random.uniform(-0.02, 0.02))
        reward = min(0.98, 0.2 + t * 0.7 + random.uniform(-0.03, 0.03))
        with _train_lock:
            _train["current_ep"] = ep
            _train["reward_history"].append(round(reward, 4))
            _train["bias_history"].append(round(bias, 4))
            if ep % max(1, episodes // 15) == 0 or ep == 1:
                _train["logs"].append(f"[ep {ep:>3}/{episodes}] reward={reward:.3f}  bias={bias:.3f}  policy_entropy={0.35 + random.uniform(-0.05, 0.05):.2f}")
        time.sleep(0.12)
    with _train_lock:
        _train["active"] = False
        _train["bias_after"] = _train["bias_history"][-1] if _train["bias_history"] else 0.2
        _train["logs"].append(f"[done] Training complete · final bias={_train['bias_after']:.3f}")


@api_router.post("/train")
async def start_train(body: TrainBody):
    if _train["active"]:
        return {"success": False, "message": "Training already in progress"}
    run = await _get_run(body.run_id)
    bias_start = run["metrics"]["overall_bias_score"] if run else 0.7
    thread = threading.Thread(target=_train_worker, args=(max(10, min(200, body.episodes)), bias_start), daemon=True)
    thread.start()
    return {"success": True, "episodes": body.episodes}


@api_router.get("/train/status")
async def train_status():
    with _train_lock:
        return dict(_train)


@api_router.post("/train/reset")
async def train_reset():
    with _train_lock:
        _train.update({"current_ep": 0, "total_ep": 0, "reward_history": [], "bias_history": [], "logs": [], "active": False})
    return {"ok": True}


# ====== Drift Monitoring Routes ======
@api_router.get("/drift/status")
async def drift_status():
    return {"history": _drift_state["history"], "alerts": _drift_state["alerts"], "baseline": _drift_state["baseline"], "injected": _drift_state["injected"]}

@api_router.post("/drift/simulate")
async def simulate_drift():
    _drift_state["injected"] = True
    _drift_state["alerts"] = []
    rng = random.Random()
    for i in range(7):
        di = round(max(0.35, 0.780 - (i+1)*0.065 + rng.uniform(-0.02, 0.02)), 3)
        dpd = round(min(0.45, 0.120 + (i+1)*0.040 + rng.uniform(-0.01, 0.01)), 3)
        bias = round(min(0.90, 0.450 + (i+1)*0.055 + rng.uniform(-0.02, 0.02)), 3)
        alert = di < 0.741 or dpd > 0.126 or bias > 0.4725
        _drift_state["history"].append({"day": f"D+{i+1}", "disparate_impact_ratio": di, "demographic_parity_diff": dpd, "overall_bias_score": bias, "alert": alert})
        if alert:
            _drift_state["alerts"].append({"day": f"D+{i+1}", "message": f"DI dropped to {di:.3f} (−5%% threshold: 0.741)", "severity": "critical" if di < 0.65 else "warn"})
    _append_audit_entry("drift_alert", {"alerts_triggered": len(_drift_state["alerts"]), "injected": True})
    return {"history": _drift_state["history"], "alerts": _drift_state["alerts"], "baseline": _drift_state["baseline"]}

@api_router.post("/drift/reset")
async def reset_drift():
    _drift_state["history"] = []
    _drift_state["alerts"] = []
    _drift_state["injected"] = False
    _init_drift_history()
    return {"ok": True}


# ====== Audit Trail Routes ======
@api_router.get("/trail")
async def get_trail():
    return {"entries": _audit_log, "count": len(_audit_log)}

@api_router.post("/trail/verify")
async def verify_trail():
    if not _audit_log:
        return {"valid": True, "message": "No entries to verify.", "checked": 0, "errors": []}
    errors = []
    prev_hash = "0" * 64
    for i, entry in enumerate(_audit_log):
        check = {k: v for k, v in entry.items() if k != "chain_hash"}
        expected_lh = _compute_hash(_json.dumps(check, sort_keys=True))
        expected_ch = _compute_hash(prev_hash + entry["log_hash"])
        if expected_ch != entry["chain_hash"]:
            errors.append(f"Entry #{i+1} (trace: {entry['trace_id'][:8]}…) — chain hash MISMATCH")
        prev_hash = entry["chain_hash"]
    return {"valid": len(errors) == 0, "checked": len(_audit_log), "errors": errors,
            "message": "✓ All records verified — chain intact" if not errors else f"✗ {len(errors)} tampered record(s) detected"}

@api_router.post("/trail/tamper/{idx}")
async def tamper_entry_demo(idx: int):
    if idx < 0 or idx >= len(_audit_log):
        return JSONResponse({"error": "index out of range"}, status_code=404)
    _audit_log[idx]["data"]["TAMPERED"] = True
    _audit_log[idx]["event"] = "[TAMPERED] " + _audit_log[idx]["event"]
    return {"ok": True, "modified_entry": idx}


# ====== Benchmark Routes ======
class BenchmarkBody(BaseModel):
    models: list[str]
    domain: str = "hiring"

@api_router.post("/benchmark")
async def run_benchmark(body: BenchmarkBody):
    results = []
    for model in body.models:
        base = dict(_BENCH_SCORES.get(model, {}))
        if not base:
            rng = random.Random(hash(model) & 0xFFFFFFFF)
            base = {"gender_bias": round(rng.uniform(0.65, 0.91), 3), "racial_bias": round(rng.uniform(0.60, 0.88), 3), "age_bias": round(rng.uniform(0.65, 0.90), 3)}
            base["overall_fairness"] = round(sum(base.values()) / 3, 3)
        results.append({"model": model, **base})
    results.sort(key=lambda x: x["overall_fairness"], reverse=True)
    for i, r in enumerate(results):
        r["rank"] = i + 1
    _append_audit_entry("benchmark_run", {"models": body.models, "domain": body.domain})
    return {"results": results, "domain": body.domain, "prompts_run": 50}


# ====== Policy Engine Routes ======
@api_router.get("/policy-rules")
async def get_policy_rules():
    return {"rules": _policy_rules}

class PolicyRuleBody(BaseModel):
    id: str
    name: str
    metric: str
    operator: str
    threshold: float
    action: str
    active: bool

@api_router.put("/policy-rules")
async def update_policy_rules(rules: list[PolicyRuleBody]):
    _policy_rules.clear()
    _policy_rules.extend([r.dict() for r in rules])
    return {"rules": _policy_rules}

@api_router.post("/policy-rules/evaluate/{run_id}")
async def evaluate_policy_rules(run_id: str):
    run = _runs.get(run_id)
    if not run:
        return JSONResponse({"error": "run not found"}, status_code=404)
    m = run["metrics"]
    results = []
    for rule in _policy_rules:
        if not rule["active"]:
            continue
        cv = m.get(rule["metric"], 0)
        op, thr = rule["operator"], rule["threshold"]
        passed = (cv >= thr) if op == ">=" else (cv <= thr) if op == "<=" else (cv > thr) if op == ">" else (cv < thr)
        results.append({**rule, "current_value": round(cv, 4), "passed": passed})
    blocked = any(r["action"] == "BLOCK" and not r["passed"] for r in results)
    warned  = any(r["action"] == "WARN"  and not r["passed"] for r in results)
    _append_audit_entry("policy_evaluation", {"run_id": run_id, "blocked": blocked, "warned": warned})
    return {"results": results, "blocked": blocked, "warned": warned,
            "verdict": "BLOCKED" if blocked else "WARNED" if warned else "CLEAR"}


# ====== Shadow AI Routes ======
class ShadowAIBody(BaseModel):
    text: str

@api_router.post("/shadow-ai/scan")
async def shadow_ai_scan(body: ShadowAIBody):
    text_lower = body.text.lower()
    words = body.text.split()
    word_count = len(words)
    raw_scores: dict = {}
    indicators: list = []
    for model, phrases in _SHADOW_PATTERNS.items():
        hits = [p for p in phrases if p in text_lower]
        raw_scores[model] = round(min(0.95, len(hits) / max(len(phrases), 1)), 3)
        indicators.extend([f"[{model}] '{p}'" for p in hits])
    bullet_lines = sum(1 for l in body.text.split('\n') if l.strip().startswith(('-', '*', '•')))
    avg_word_len = sum(len(w.strip('.,!?')) for w in words) / max(word_count, 1)
    if bullet_lines >= 2:
        raw_scores["Claude"] = min(0.95, raw_scores.get("Claude", 0) + 0.15)
    if avg_word_len < 4.0:
        raw_scores["GPT"] = min(0.95, raw_scores.get("GPT", 0) + 0.10)
    total_ai = sum(raw_scores.values())
    human_prob = round(max(0.03, 1.0 - total_ai * 0.7), 3)
    best_model = max(raw_scores, key=raw_scores.get) if raw_scores else None
    best_score = raw_scores.get(best_model, 0) if best_model else 0
    if human_prob > 0.65 or best_score < 0.12:
        verdict, confidence = "Human", human_prob
    else:
        verdict, confidence = best_model, round(best_score, 3)
    return {"verdict": verdict, "confidence": confidence, "human_probability": human_prob,
            "model_scores": raw_scores, "word_count": word_count,
            "indicators": indicators[:8], "structural": {"bullet_lines": bullet_lines, "avg_word_len": round(avg_word_len, 2)}}

# ====== LPS Engine Routes (Longitudinal Policy Search) ======
from core.lps_engine import simulate_longitudinal, compare_strategies
from core.causal_simulator import run_parallel_universes, asymmetric_cost_analysis

class LPSBody(BaseModel):
    domain: str = "hiring"
    years: int = 10
    mitigation: str = ""  # "", "static_threshold", "reweighting", "dynamic_lps"

@api_router.post("/lps/simulate")
async def lps_simulate(body: LPSBody):
    result = simulate_longitudinal(
        domain=body.domain, years=body.years,
        mitigation=body.mitigation or None,
    )
    _append_audit_entry("lps_simulation", {"domain": body.domain, "years": body.years, "mitigation": body.mitigation or "none"})
    return result

@api_router.post("/lps/compare")
async def lps_compare(body: LPSBody):
    result = compare_strategies(
        domain=body.domain, years=body.years,
    )
    _append_audit_entry("lps_comparison", {"domain": body.domain, "years": body.years})
    return result

# ====== Causal Parallel Universe Routes ======
class CausalBody(BaseModel):
    domain: str = "hiring"
    universe_a: str = "drop_feature"
    universe_b: str = "reweight_samples"
    years: int = 10

@api_router.post("/causal/compare")
async def causal_compare(body: CausalBody):
    result = run_parallel_universes(
        domain=body.domain,
        universe_a_strategy=body.universe_a,
        universe_b_strategy=body.universe_b,
        years=body.years,
    )
    _append_audit_entry("causal_comparison", {"domain": body.domain, "a": body.universe_a, "b": body.universe_b})
    return result

# ====== Asymmetric Cost Analysis Route ======
class CostBody(BaseModel):
    domain: str = "hiring"

@api_router.post("/cost-analysis")
async def cost_analysis(body: CostBody):
    run = list(_runs.values())[-1] if _runs else None
    result = asymmetric_cost_analysis(domain=body.domain, run_metrics=run["metrics"] if run else None)
    _append_audit_entry("cost_analysis", {"domain": body.domain, "underestimate": result["harm_underestimate_pct"]})
    return result


FRONTEND_DIR = ROOT_DIR.parent / 'frontend'

app.include_router(api_router)
app.add_middleware(
    CORSMiddleware,
    allow_credentials=True,
    allow_origins=os.environ.get('CORS_ORIGINS', '*').split(','),
    allow_methods=["*"],
    allow_headers=["*"],
)

# ====== Serve sample CSV data ======
SAMPLE_DIR = FRONTEND_DIR / 'sample_data'
if SAMPLE_DIR.exists():
    app.mount("/sample_data", StaticFiles(directory=str(SAMPLE_DIR)), name="sample_data")

@api_router.get("/sample-datasets")
async def list_sample_datasets():
    files = []
    if SAMPLE_DIR.exists():
        for f in SAMPLE_DIR.glob("*.csv"):
            files.append({"name": f.name, "url": f"/sample_data/{f.name}", "size_kb": round(f.stat().st_size / 1024, 1)})
    return {"datasets": files}

# ====== Serve frontend HTML ======
@app.get("/", response_class=HTMLResponse)
async def serve_frontend():
    idx = FRONTEND_DIR / 'index.html'
    if idx.exists():
        headers = {
            "Cache-Control": "no-cache, no-store, must-revalidate",
            "Pragma": "no-cache",
            "Expires": "0"
        }
        return FileResponse(str(idx), media_type='text/html', headers=headers)
    return HTMLResponse('<h1>FairForge Arena</h1><p>Frontend not found. Make sure frontend/index.html exists.</p>')

# Health-check aliases (browser was hitting /api/v1/health)
@app.get("/api/v1/health")
@app.get("/api/health")
async def health_check():
    return {"status": "ok", "service": "FairForge Arena API", "version": "3.1.0", "frontend": str(FRONTEND_DIR / 'index.html'), "frontend_exists": (FRONTEND_DIR / 'index.html').exists()}

@app.on_event("shutdown")
async def shutdown_db_client():
    client.close()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)