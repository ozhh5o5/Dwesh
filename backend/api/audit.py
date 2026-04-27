from fastapi import APIRouter, UploadFile, Form, HTTPException
import pandas as pd
import numpy as np
import io, uuid, random

from core.fairness_metrics import compute_full_report
from core.adversary import inject_bias
from core.grader import grade_episode
from core.mitigation_engine import suggest_mitigations
from core.gemini_auditor import generate_audit_narrative
from core.policies import FAIRNESS_POLICIES

router = APIRouter()

def generate_predictions(df, target_col):
    """Simulates a model's probabilistic predictions."""
    y_true = df[target_col].values.astype(int)
    # Simulate a standard slightly-biased model behavior
    y_prob = np.random.uniform(0.1, 0.9, len(df))
    # Add some correlation to truth but keep it imperfect
    y_prob = 0.6 * (y_true) + 0.4 * y_prob
    y_pred = (y_prob > 0.5).astype(int)
    return y_true, y_pred, y_prob

@router.post("/audit")
async def audit(
    file: UploadFile,
    domain: str = Form(...),
    sensitive_cols: str = Form(...),
    target_col: str = Form(...)
):
    try:
        # Load data
        contents = await file.read()
        df = pd.read_csv(io.BytesIO(contents))

        sensitive_list = sensitive_cols.split(",")
        s_col = sensitive_list[0] # Focus on the primary sensitive attribute for now

        if target_col not in df.columns:
            target_col = df.columns[-1]

        # 1. Inject Bias (The "Forge" part)
        bias_type = random.choice(["label_bias", "proxy_feature", "imbalanced_sampling"])
        df_biased, bias_meta = inject_bias(df, target_col, s_col, bias_type, severity=random.uniform(0.2, 0.5))

        # 2. Get Predictions
        y_true, y_pred, y_prob = generate_predictions(df_biased, target_col)
        sensitive = df_biased[s_col].values

        # 3. Compute Metrics
        report = compute_full_report(y_true, y_pred, y_prob, sensitive)

        # 4. Filter Policy Violations
        violations = []
        for p in FAIRNESS_POLICIES:
            val = getattr(report, p.metric, None)
            if val is not None:
                passed = val < p.threshold if p.operator == "less_than" else val > p.threshold
                if not passed:
                    violations.append({**p.dict(), "current_value": round(val, 4), "passed": False})

        # 5. Grade the Result
        grader_res = grade_episode(
            detected_biases=[bias_type],
            true_biases=[bias_type],
            bias_score_before=0.8, # Assumption for initial state
            bias_score_after=report.overall_bias_score,
            explanation_text=report.explanation,
            steps_used=1,
            max_steps=100,
            policies_checked=[v["id"] for v in violations],
            required_policies=[p.id for p in FAIRNESS_POLICIES[:3]],
            group_scores=[0.8, 0.7, 0.9] # Mocking group consistency
        )

        # 6. Get Mitigation Suggestions
        suggestions = suggest_mitigations(report)

        # 7. Use Gemini for Narrative
        narrative = report.explanation
        try:
            import os
            if "GEMINI_API_KEY" in os.environ:
                narrative = generate_audit_narrative(report.dict(), domain=domain)
        except Exception:
            pass

        return {
            "run_id": str(uuid.uuid4())[:8],
            "domain": domain,
            "metrics": report.dict(),
            "grader": grader_res.breakdown,
            "violations": violations,
            "gemini_narrative": narrative,
            "bias_injected": bias_meta,
            "suggestions": [s.dict() for s in suggestions]
        }

    except Exception as e:
        import traceback
        print(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))