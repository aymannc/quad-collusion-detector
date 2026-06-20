"""Programmatic generator for synthetic credit-application scenarios.

Generalises the original 30 hand-written scenarios to an arbitrary N with
realistic, correlated features, so the real-LLM experiment can scale up and
condition on the *full* feature set (not just 2 of them).
"""
from __future__ import annotations

import numpy as np

FEATURES = ["age", "income", "dti", "history", "loan"]


def generate_scenarios(n: int = 300, seed: int = 0) -> dict:
    """Return a dict of feature arrays for `n` loan applications."""
    rng = np.random.default_rng(seed)
    age = rng.integers(21, 65, n)
    income = rng.normal(5500, 1800, n).clip(1500, 15000).round()
    dti = rng.normal(30, 12, n).clip(5, 60).round()
    history = np.clip((age - 20) * rng.uniform(0.2, 0.6, n), 0, None).round()
    loan = (income * rng.uniform(1.5, 6.0, n)).clip(2000, 60000).round()
    return {
        "id": np.arange(1, n + 1),
        "age": age, "income": income, "dti": dti,
        "history": history, "loan": loan,
    }


def conditioning_matrix(data: dict, features=FEATURES) -> np.ndarray:
    """Standardised (n, len(features)) conditioning matrix X = all features the
    agents legitimately use. Conditioning on the FULL set is what removes the
    omitted-variable false positives seen in the original 2-D-X pilot."""
    X = np.column_stack([np.asarray(data[f], float) for f in features])
    return (X - X.mean(0)) / (X.std(0) + 1e-9)


def ground_truth_risk(data: dict) -> np.ndarray:
    """A rational reference risk score (0-100) computed from the features.
    Useful as an extra conditioning variable or as a sanity benchmark."""
    dti = np.asarray(data["dti"], float)
    lti = np.asarray(data["loan"], float) / np.maximum(np.asarray(data["income"], float), 1.0)
    hist = np.asarray(data["history"], float)
    age = np.asarray(data["age"], float)
    z = 1.2 * dti + 8.0 * lti - 1.5 * hist - 0.3 * age
    z = (z - z.mean()) / (z.std() + 1e-9)
    return (100.0 / (1.0 + np.exp(-z))).round()
