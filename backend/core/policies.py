"""
FairForge Arena v3.0 — 12 Fairness Policies
Each policy defines a rule, threshold, severity, and domain.
"""
from pydantic import BaseModel
from typing import Literal

class FairnessPolicy(BaseModel):
    id: str
    name: str
    description: str
    metric: str
    threshold: float
    operator: Literal["less_than", "greater_than"]
    severity: Literal["critical", "high", "medium", "low"]
    domain: str
    legal_reference: str | None = None

FAIRNESS_POLICIES: list[FairnessPolicy] = [
    FairnessPolicy(
        id="FP-01",
        name="Demographic Parity",
        description="Selection rate must not differ by more than 20% across protected groups.",
        metric="demographic_parity_diff",
        threshold=0.2,
        operator="less_than",
        severity="critical",
        domain="all",
        legal_reference="EEOC Uniform Guidelines"
    ),
    FairnessPolicy(
        id="FP-02",
        name="80% Disparate Impact Rule",
        description="Minority group selection rate must be >= 80% of majority group.",
        metric="disparate_impact_ratio",
        threshold=0.8,
        operator="greater_than",
        severity="critical",
        domain="hiring",
        legal_reference="US Title VII, EEOC"
    ),
    FairnessPolicy(
        id="FP-03",
        name="Equal Opportunity",
        description="True positive rate gap across groups must be < 0.15.",
        metric="equal_opportunity_diff",
        threshold=0.15,
        operator="less_than",
        severity="high",
        domain="all",
    ),
    FairnessPolicy(
        id="FP-04",
        name="Equalized Odds",
        description="Both TPR and FPR must be within 0.15 across groups.",
        metric="equalized_odds_diff",
        threshold=0.15,
        operator="less_than",
        severity="high",
        domain="all",
    ),
    FairnessPolicy(
        id="FP-05",
        name="Calibration Fairness",
        description="Confidence score calibration error must not differ > 0.1 by group.",
        metric="calibration_diff",
        threshold=0.1,
        operator="less_than",
        severity="medium",
        domain="medical",
    ),
    FairnessPolicy(
        id="FP-06",
        name="No Proxy Discrimination",
        description="ZIP code, school name, or neighborhood must not serve as race proxy.",
        metric="proxy_score",
        threshold=0.3,
        operator="less_than",
        severity="critical",
        domain="all",
        legal_reference="Fair Housing Act"
    ),
    FairnessPolicy(
        id="FP-07",
        name="Intersectional Fairness",
        description="Fairness must hold at intersections (e.g. Black women, not just Black OR women).",
        metric="intersectional_dpd",
        threshold=0.25,
        operator="less_than",
        severity="high",
        domain="all",
    ),
    FairnessPolicy(
        id="FP-08",
        name="Age Non-Discrimination",
        description="Decision rates for applicants 40+ must not differ by > 15%.",
        metric="age_parity_diff",
        threshold=0.15,
        operator="less_than",
        severity="high",
        domain="hiring",
        legal_reference="ADEA 1967"
    ),
    FairnessPolicy(
        id="FP-09",
        name="Loan Approval Parity",
        description="Loan denial rate gap across race/gender must be < 10%.",
        metric="demographic_parity_diff",
        threshold=0.1,
        operator="less_than",
        severity="critical",
        domain="finance",
        legal_reference="Equal Credit Opportunity Act"
    ),
    FairnessPolicy(
        id="FP-10",
        name="Medical Diagnostic Equity",
        description="Sensitivity (recall) gap in medical diagnosis must be < 0.1 across groups.",
        metric="equal_opportunity_diff",
        threshold=0.1,
        operator="less_than",
        severity="critical",
        domain="medical",
    ),
    FairnessPolicy(
        id="FP-11",
        name="Counterfactual Fairness",
        description="Changing only protected attribute should not change decision > 10% of cases.",
        metric="counterfactual_flip_rate",
        threshold=0.1,
        operator="less_than",
        severity="high",
        domain="all",
    ),
    FairnessPolicy(
        id="FP-12",
        name="Consistency Score",
        description="Similar individuals (by non-protected attributes) should get same decision > 90%.",
        metric="consistency_score",
        threshold=0.9,
        operator="greater_than",
        severity="medium",
        domain="all",
    ),
]