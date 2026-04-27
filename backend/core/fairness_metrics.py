"""
FairForge Arena v3.0 — Core Fairness Metrics Engine
All functions take y_true, y_pred, sensitive_attr as numpy arrays
"""
import numpy as np
from typing import Dict, List
from pydantic import BaseModel


class FairnessReport(BaseModel):
    demographic_parity_diff: float
    equal_opportunity_diff: float
    disparate_impact_ratio: float
    equalized_odds_diff: float
    calibration_diff: float
    overall_bias_score: float  # 0 = fair, 1 = fully biased
    flagged: bool
    explanation: str


def demographic_parity_difference(y_pred: np.ndarray, sensitive: np.ndarray) -> float:
    """
    DPD = |P(ŷ=1|A=0) - P(ŷ=1|A=1)|
    Should be < 0.1 for fairness. Red flag > 0.2
    """
    groups = np.unique(sensitive)
    rates = [np.mean(y_pred[sensitive == g]) for g in groups]
    return float(max(rates) - min(rates))


def equal_opportunity_difference(y_true: np.ndarray, y_pred: np.ndarray, sensitive: np.ndarray) -> float:
    """
    EOD = |TPR_group0 - TPR_group1|
    True positive rate difference across groups
    """
    groups = np.unique(sensitive)
    tprs = []
    for g in groups:
        mask = (sensitive == g) & (y_true == 1)
        if mask.sum() == 0:
            tprs.append(0.0)
        else:
            tprs.append(float(np.mean(y_pred[mask])))
    return float(max(tprs) - min(tprs))


def disparate_impact_ratio(y_pred: np.ndarray, sensitive: np.ndarray) -> float:
    """
    DIR = P(ŷ=1|A=minority) / P(ŷ=1|A=majority)
    Must be >= 0.8 (80% rule). Below = illegal discrimination.
    """
    groups = np.unique(sensitive)
    rates = {g: np.mean(y_pred[sensitive == g]) for g in groups}
    min_rate = min(rates.values())
    max_rate = max(rates.values())
    if max_rate == 0:
        return 1.0
    return float(min_rate / max_rate)


def equalized_odds_diff(y_true: np.ndarray, y_pred: np.ndarray, sensitive: np.ndarray) -> float:
    """
    Max of |TPR diff| and |FPR diff| across groups
    """
    groups = np.unique(sensitive)
    tprs, fprs = [], []
    for g in groups:
        mask_pos = (sensitive == g) & (y_true == 1)
        mask_neg = (sensitive == g) & (y_true == 0)
        tprs.append(np.mean(y_pred[mask_pos]) if mask_pos.sum() > 0 else 0.0)
        fprs.append(np.mean(y_pred[mask_neg]) if mask_neg.sum() > 0 else 0.0)
    return float(max(max(tprs) - min(tprs), max(fprs) - min(fprs)))


def calibration_difference(y_true: np.ndarray, y_prob: np.ndarray, sensitive: np.ndarray) -> float:
    """
    |mean(y_true - y_prob) per group| difference
    Checks if confidence scores are equally calibrated
    """
    groups = np.unique(sensitive)
    errors = [float(np.mean(y_true[sensitive == g] - y_prob[sensitive == g])) for g in groups]
    return float(max(errors) - min(errors))


def compute_full_report(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray,
    sensitive: np.ndarray,
    threshold: float = 0.2
) -> FairnessReport:
    dpd = demographic_parity_difference(y_pred, sensitive)
    eod = equal_opportunity_difference(y_true, y_pred, sensitive)
    dir_ = disparate_impact_ratio(y_pred, sensitive)
    eqo = equalized_odds_diff(y_true, y_pred, sensitive)
    cal = abs(calibration_difference(y_true, y_prob, sensitive))

    # Composite bias score (0=fair, 1=biased)
    bias_score = np.mean([
        min(dpd / 0.4, 1.0),
        min(eod / 0.4, 1.0),
        min((1 - dir_) / 0.5, 1.0),
        min(eqo / 0.4, 1.0),
        min(cal / 0.4, 1.0),
    ])

    flagged = dpd > threshold or dir_ < 0.8 or eod > threshold

    explanation = _generate_explanation(dpd, eod, dir_, eqo, cal, flagged)

    return FairnessReport(
        demographic_parity_diff=round(dpd, 4),
        equal_opportunity_diff=round(eod, 4),
        disparate_impact_ratio=round(dir_, 4),
        equalized_odds_diff=round(eqo, 4),
        calibration_diff=round(cal, 4),
        overall_bias_score=round(float(bias_score), 4),
        flagged=flagged,
        explanation=explanation
    )


def _generate_explanation(dpd, eod, dir_, eqo, cal, flagged) -> str:
    issues = []
    if dpd > 0.2:
        issues.append(f"Demographic parity gap of {dpd:.2f} detected — selection rates differ significantly across groups.")
    if dir_ < 0.8:
        issues.append(f"Disparate Impact Ratio {dir_:.2f} violates the 80% legal rule.")
    if eod > 0.2:
        issues.append(f"Equal opportunity gap of {eod:.2f} — true positive rates differ by group.")
    if not issues:
        return "Model appears fair across measured dimensions. Continue monitoring."
    return " ".join(issues)