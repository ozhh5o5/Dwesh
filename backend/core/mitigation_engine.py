"""
FairForge Mitigation Engine — Recommends and applies bias fixes
"""
import numpy as np
import pandas as pd
from typing import List
from pydantic import BaseModel

class MitigationSuggestion(BaseModel):
    strategy: str
    description: str
    expected_improvement: str
    code_snippet: str
    priority: int  # 1=highest

def suggest_mitigations(report) -> List[MitigationSuggestion]:
    suggestions = []

    if report.demographic_parity_diff > 0.2:
        suggestions.append(MitigationSuggestion(
            strategy="Reweighting",
            description="Assign higher sample weights to underrepresented groups during training.",
            expected_improvement="Reduces demographic parity gap by ~40-60%",
            code_snippet="""
# Compute sample weights inversely proportional to group frequency
from sklearn.utils.class_weight import compute_sample_weight
weights = compute_sample_weight('balanced', y=sensitive_attr)
model.fit(X_train, y_train, sample_weight=weights)
""",
            priority=1
        ))

    if report.disparate_impact_ratio < 0.8:
        suggestions.append(MitigationSuggestion(
            strategy="Remove Proxy Features",
            description="Remove features that act as proxies for protected attributes (ZIP code, school name).",
            expected_improvement="Eliminates proxy-driven disparate impact",
            code_snippet="""
# Identify and drop proxy features
proxy_features = ['zip_code', 'neighborhood', 'school_name']
X_train = X_train.drop(columns=[c for c in proxy_features if c in X_train.columns])
""",
            priority=1
        ))

    if report.equal_opportunity_diff > 0.15:
        suggestions.append(MitigationSuggestion(
            strategy="Threshold Adjustment",
            description="Use different decision thresholds per group to equalize true positive rates.",
            expected_improvement="Equalizes TPR across groups within 0.05",
            code_snippet="""
# Set group-specific thresholds
thresholds = {'majority': 0.5, 'minority': 0.35}
y_pred = np.where(
    sensitive == 'minority',
    (y_prob >= thresholds['minority']).astype(int),
    (y_prob >= thresholds['majority']).astype(int)
)
""",
            priority=2
        ))

    if report.equalized_odds_diff > 0.15:
        suggestions.append(MitigationSuggestion(
            strategy="Adversarial Debiasing",
            description="Train a secondary network to predict sensitive attribute — penalize if it succeeds.",
            expected_improvement="Reduces equalized odds gap by 50-70%",
            code_snippet="""
# Use fairlearn's AdversarialFairnessClassifier
from fairlearn.adversarial import AdversarialFairnessClassifier
mitigator = AdversarialFairnessClassifier(
    backend='torch', predictor_model=[50, 'leaky_relu'],
    adversary_model=[3, 'leaky_relu'], batch_size=2**8,
    progress_updates=0.5, random_state=42
)
mitigator.fit(X_train, y_train, sensitive_features=sensitive_train)
""",
            priority=2
        ))

    if report.calibration_diff > 0.1:
        suggestions.append(MitigationSuggestion(
            strategy="Calibration by Group",
            description="Apply Platt scaling or isotonic regression separately per demographic group.",
            expected_improvement="Equalizes confidence scores across groups",
            code_snippet="""
from sklearn.calibration import CalibratedClassifierCV
calibrated = CalibratedClassifierCV(base_model, method='isotonic', cv='prefit')
calibrated.fit(X_cal[group_mask], y_cal[group_mask])
""",
            priority=3
        ))

    return sorted(suggestions, key=lambda x: x.priority)


def apply_reweighting(X: np.ndarray, y: np.ndarray, sensitive: np.ndarray) -> np.ndarray:
    """Returns sample weights for reweighting mitigation."""
    n = len(y)
    weights = np.ones(n)
    for g in np.unique(sensitive):
        for label in [0, 1]:
            mask = (sensitive == g) & (y == label)
            if mask.sum() > 0:
                weights[mask] = n / (len(np.unique(sensitive)) * 2 * mask.sum())
    return weights