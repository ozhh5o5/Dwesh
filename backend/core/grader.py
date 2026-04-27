"""
FairForge Grader — Scores the PPO agent's fairness audit quality
Weights: Bias Detection 30% | Mitigation 25% | Explanation 20% | Efficiency 10% | Policy 10% | Consistency 5%
"""
from dataclasses import dataclass
from typing import Dict
import numpy as np

@dataclass
class GraderResult:
    bias_detection_score: float      # 0-100
    mitigation_score: float          # 0-100
    explanation_quality: float       # 0-100
    efficiency_score: float          # 0-100
    policy_compliance_score: float   # 0-100
    consistency_score: float         # 0-100
    final_score: float               # weighted composite
    passed: bool
    breakdown: Dict[str, float]

WEIGHTS = {
    "bias_detection": 0.30,
    "mitigation": 0.25,
    "explanation": 0.20,
    "efficiency": 0.10,
    "policy_compliance": 0.10,
    "consistency": 0.05,
}

def grade_episode(
    detected_biases: list,          # biases the agent flagged
    true_biases: list,              # ground truth injected biases
    bias_score_before: float,       # overall_bias_score before mitigation
    bias_score_after: float,        # overall_bias_score after mitigation
    explanation_text: str,          # agent's explanation string
    steps_used: int,                # how many env steps
    max_steps: int,
    policies_checked: list,         # policy IDs the agent validated
    required_policies: list,
    group_scores: list[float],      # per-group consistency
) -> GraderResult:

    # 1. Bias Detection (precision + recall F1)
    tp = len(set(detected_biases) & set(true_biases))
    precision = tp / len(detected_biases) if detected_biases else 0
    recall = tp / len(true_biases) if true_biases else 0
    f1 = 2 * precision * recall / (precision + recall + 1e-9)
    bias_detection = f1 * 100

    # 2. Mitigation Effectiveness
    reduction = max(0, bias_score_before - bias_score_after) / (bias_score_before + 1e-9)
    mitigation = min(reduction * 100, 100)

    # 3. Explanation Quality (heuristic: length, keyword presence)
    keywords = ["demographic", "parity", "disparate", "group", "bias", "fairness", "protected"]
    kw_hits = sum(1 for kw in keywords if kw.lower() in explanation_text.lower())
    explanation = min((len(explanation_text) / 200) * 50 + (kw_hits / len(keywords)) * 50, 100)

    # 4. Efficiency
    efficiency = max(0, (1 - steps_used / max_steps)) * 100

    # 5. Policy Compliance
    covered = len(set(policies_checked) & set(required_policies))
    policy_compliance = (covered / len(required_policies)) * 100 if required_policies else 100

    # 6. Consistency
    consistency = (1 - np.std(group_scores)) * 100 if group_scores else 100

    final = (
        bias_detection * WEIGHTS["bias_detection"]
        + mitigation * WEIGHTS["mitigation"]
        + explanation * WEIGHTS["explanation"]
        + efficiency * WEIGHTS["efficiency"]
        + policy_compliance * WEIGHTS["policy_compliance"]
        + consistency * WEIGHTS["consistency"]
    )

    return GraderResult(
        bias_detection_score=round(bias_detection, 2),
        mitigation_score=round(mitigation, 2),
        explanation_quality=round(explanation, 2),
        efficiency_score=round(efficiency, 2),
        policy_compliance_score=round(policy_compliance, 2),
        consistency_score=round(consistency, 2),
        final_score=round(final, 2),
        passed=final >= 70,
        breakdown={k: round(v, 2) for k, v in {
            "bias_detection": bias_detection,
            "mitigation": mitigation,
            "explanation": explanation,
            "efficiency": efficiency,
            "policy_compliance": policy_compliance,
            "consistency": consistency,
        }.items()}
    )