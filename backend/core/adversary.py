"""
FairForge Basilisk Adversary — Injects bias into datasets for the RL agent to detect
Replaces jailbreak generation with bias injection strategies
"""
import numpy as np
import pandas as pd
from typing import Literal

BiasType = Literal[
    "label_bias",           # flip labels for minority group
    "proxy_feature",        # encode race via ZIP code
    "imbalanced_sampling",  # undersample minority group
    "hidden_correlation",   # add feature correlated with protected attr
    "stereotype_prompt",    # add stereotype-encoded text feature
    "intersectional",       # bias only at group intersection
]

def inject_bias(
    df: pd.DataFrame,
    target_col: str,
    sensitive_col: str,
    bias_type: BiasType,
    severity: float = 0.3,   # 0.0 = no bias, 1.0 = maximum bias
    seed: int = 42
) -> tuple[pd.DataFrame, dict]:
    """
    Injects bias into a clean dataset.
    Returns modified df + metadata about what was injected.
    """
    np.random.seed(seed)
    df = df.copy()
    metadata = {"bias_type": bias_type, "severity": severity, "affected_rows": 0}

    minority_mask = df[sensitive_col] == df[sensitive_col].value_counts().index[-1]

    if bias_type == "label_bias":
        # Flip positive labels to negative for minority group
        flip_mask = minority_mask & (df[target_col] == 1)
        n_flip = int(flip_mask.sum() * severity)
        flip_idx = df[flip_mask].sample(n=n_flip, random_state=seed).index
        df.loc[flip_idx, target_col] = 0
        metadata["affected_rows"] = n_flip

    elif bias_type == "proxy_feature":
        # Add ZIP-code-like feature that encodes group membership
        df["zip_code"] = np.where(
            minority_mask,
            np.random.choice([10001, 10002], size=len(df)),   # "minority" ZIPs
            np.random.choice([90210, 10036], size=len(df))    # "majority" ZIPs
        )
        # Add noise proportional to (1 - severity)
        noise_mask = np.random.random(len(df)) < (1 - severity)
        df.loc[noise_mask, "zip_code"] = np.random.choice([10001, 10002, 90210, 10036], noise_mask.sum())
        metadata["proxy_feature"] = "zip_code"

    elif bias_type == "imbalanced_sampling":
        # Drop majority of minority group rows
        keep_frac = 1 - severity * 0.8
        drop_idx = df[minority_mask].sample(frac=(1 - keep_frac), random_state=seed).index
        df = df.drop(index=drop_idx)
        metadata["affected_rows"] = len(drop_idx)

    elif bias_type == "hidden_correlation":
        # Add feature highly correlated with protected attribute
        correlation_strength = severity
        df["credit_history_score"] = np.where(
            minority_mask,
            np.random.normal(500, 80, len(df)) * (1 - correlation_strength) + 400 * correlation_strength,
            np.random.normal(700, 80, len(df))
        )

    elif bias_type == "intersectional":
        # Only bias at intersection (e.g. minority group AND female)
        if "gender" in df.columns:
            intersect_mask = minority_mask & (df["gender"] == 0)
            flip_idx = df[intersect_mask & (df[target_col] == 1)].sample(
                frac=severity, random_state=seed
            ).index
            df.loc[flip_idx, target_col] = 0
            metadata["affected_rows"] = len(flip_idx)

    return df, metadata