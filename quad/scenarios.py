"""Credit-application data for the experiments.

Two sources, exposed through a common dict format:
  - generate_scenarios(): synthetic applicants (age, income, dti, history, loan)
  - load_german_credit(): the real UCI Statlog German Credit dataset (1000
    applicants), using its numeric columns as features.

Every dataset dict carries an "id" array, one array per feature column, and a
"_features" list naming the feature columns to use (for the prompt and for the
conditioning matrix). This lets the rest of the pipeline stay dataset-agnostic.
"""
from __future__ import annotations

import numpy as np

FEATURES = ["age", "income", "dti", "history", "loan"]


def generate_scenarios(n: int = 300, seed: int = 0) -> dict:
    """Synthetic loan applications with realistic, correlated features."""
    rng = np.random.default_rng(seed)
    age = rng.integers(21, 65, n)
    income = rng.normal(5500, 1800, n).clip(1500, 15000).round()
    dti = rng.normal(30, 12, n).clip(5, 60).round()
    history = np.clip((age - 20) * rng.uniform(0.2, 0.6, n), 0, None).round()
    loan = (income * rng.uniform(1.5, 6.0, n)).clip(2000, 60000).round()
    return {"id": np.arange(1, n + 1), "age": age, "income": income, "dti": dti,
            "history": history, "loan": loan, "_features": list(FEATURES)}


def feature_names(data: dict):
    return data.get("_features", FEATURES)


def data_fingerprint(data: dict) -> str:
    """Short, stable content hash of the applicants (feature names + ids + values
    + coordination signal). Caches are keyed on this so different datasets or
    samples can never share a cache entry."""
    import hashlib
    h = hashlib.sha256()
    feats = feature_names(data)
    h.update("|".join(map(str, feats)).encode())
    for k in ["id", *feats, "_coord"]:
        if k in data:
            h.update(np.ascontiguousarray(np.asarray(data[k], dtype=float)).tobytes())
    return h.hexdigest()[:12]


def conditioning_matrix(data: dict, features=None) -> np.ndarray:
    """Standardised (n, d) conditioning matrix from the dataset's feature columns."""
    feats = features or feature_names(data)
    X = np.column_stack([np.asarray(data[f], float) for f in feats])
    return (X - X.mean(0)) / (X.std(0) + 1e-9)


def ground_truth_risk(data: dict) -> np.ndarray:
    """Rational reference risk (0-100) for the SYNTHETIC credit schema."""
    dti = np.asarray(data["dti"], float)
    lti = np.asarray(data["loan"], float) / np.maximum(np.asarray(data["income"], float), 1.0)
    hist = np.asarray(data["history"], float)
    age = np.asarray(data["age"], float)
    z = 1.2 * dti + 8.0 * lti - 1.5 * hist - 0.3 * age
    z = (z - z.mean()) / (z.std() + 1e-9)
    return (100.0 / (1.0 + np.exp(-z))).round()


def reference_score(data: dict) -> np.ndarray:
    """A 1-D 'rational' reference score (0-100) used for reference-conditioning.
    Uses the domain formula for the synthetic schema; for any other (real)
    dataset, uses the standardised first principal component of the features."""
    if set(FEATURES).issubset(feature_names(data)):
        return ground_truth_risk(data)
    X = conditioning_matrix(data)
    Xc = X - X.mean(0)
    _, _, vt = np.linalg.svd(Xc, full_matrices=False)   # first principal component
    pc1 = Xc @ vt[0]
    pc1 = (pc1 - pc1.mean()) / (pc1.std() + 1e-9)
    return (100.0 / (1.0 + np.exp(-pc1))).round()


# Official Statlog German Credit schema — classic german.data is 20 attributes
# + target ("class"), in this exact order, whitespace-delimited, no header.
_GERMAN_COLS = [
    "checking_status", "duration_months", "credit_history", "purpose",
    "credit_amount", "savings_status", "employment_since", "installment_rate_pct",
    "personal_status_sex", "other_debtors", "residence_since", "property_type",
    "age", "other_installment_plans", "housing", "existing_credits", "job",
    "num_dependents", "telephone", "foreign_worker", "class",
]
# The 7 genuinely numeric attributes (the rest are coded categoricals; "class" is
# the target and must NOT be used as a feature).
_GERMAN_NUMERIC = ["duration_months", "credit_amount", "installment_rate_pct",
                   "residence_since", "age", "existing_credits", "num_dependents"]
_TARGET_NAMES = {"class", "target", "y", "default", "credit_risk", "risk", "label", "id"}


def load_german_credit(path: str = "data/german.data", n: int = 1000, seed: int = 0) -> dict:
    """Load a LOCAL copy of the classic UCI Statlog German Credit dataset
    (german.data: 1000 applicants, 20 attributes + target, whitespace-delimited,
    no header). Uses the 7 numeric attributes as features and drops the target.
    Also accepts a CSV with a header row. See the README for download steps."""
    import os
    import pandas as pd
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"German Credit file not found at {path!r}. Download it first "
            f"(see README, 'Real dataset (German Credit)').")
    raw = pd.read_csv(path, sep=r"\s+", header=None, engine="python")
    if raw.shape[1] == 21:
        raw.columns = _GERMAN_COLS                     # classic 21-column german.data
        df, feats = raw, list(_GERMAN_NUMERIC)
    elif raw.shape[1] > 1:                             # whitespace, but not 21 columns
        raise RuntimeError(
            f"{path!r} parsed as {raw.shape[1]} whitespace columns — expected the "
            "classic 21-column 'german.data'. Re-download it (see README) and verify "
            f"`awk 'NR==1{{print NF}}' {path}` prints 21.")
    else:                                              # single field -> CSV with header
        df = pd.read_csv(path)
        feats = [c for c in df.select_dtypes("number").columns
                 if str(c).lower() not in _TARGET_NAMES]
    if not feats:
        raise RuntimeError(f"German Credit: no numeric feature columns in {path!r}")
    if n and n < len(df):
        df = df.sample(n=n, random_state=seed).reset_index(drop=True)
    data = {"id": np.arange(1, len(df) + 1), "_features": list(feats)}
    for c in feats:
        data[c] = np.asarray(df[c], dtype=float)
    for tgt in ("class", "target", "y", "default", "credit_risk", "label"):
        if tgt in df.columns:                      # carry true label for a learned reference
            data["_label"] = np.asarray(df[tgt], dtype=float)
            break
    return data


def load_taiwan(path: str = "data/taiwan_credit.xls", n: int = 1000, seed: int = 0) -> dict:
    """Load the UCI 'Default of Credit Card Clients' dataset (Taiwan, 2005;
    30,000 clients, 23 numeric features). Reads the UCI .xls (whose real header
    is the 2nd row) or a CSV mirror; drops ID and the target. See the README."""
    import os
    import pandas as pd
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Taiwan Credit file not found at {path!r}. Download it first "
            "(see README, 'Real dataset (Taiwan Credit Card Default)').")
    is_excel = path.lower().endswith((".xls", ".xlsx"))

    def _read(h):
        return pd.read_excel(path, header=h) if is_excel else pd.read_csv(path, header=h)

    df = _read(0)
    if not any(str(c).strip().upper() == "LIMIT_BAL" for c in df.columns):
        df = _read(1)                              # UCI .xls: real header is the 2nd row
    df.columns = [str(c).strip() for c in df.columns]
    drop = {c for c in df.columns if c.upper() == "ID" or "DEFAULT" in c.upper()}
    feats = [c for c in df.select_dtypes("number").columns if c not in drop]
    if not feats:
        raise RuntimeError(f"Taiwan Credit: no numeric feature columns in {path!r}")
    if n and n < len(df):
        df = df.sample(n=n, random_state=seed).reset_index(drop=True)
    data = {"id": np.arange(1, len(df) + 1), "_features": feats}
    for c in feats:
        data[c] = np.asarray(df[c], dtype=float)
    for c in df.columns:                            # carry true label for a learned reference
        if "DEFAULT" in str(c).upper():
            data["_label"] = np.asarray(df[c], dtype=float)
            break
    return data


def get_dataset(name: str, n: int = 1000, seed: int = 0, path: str = None) -> dict:
    """Dispatch: 'synthetic' (default), 'german' (UCI German Credit file), or
    'taiwan' (UCI Default of Credit Card Clients)."""
    if name == "german":
        return load_german_credit(path=path or "data/german.data", n=n, seed=seed)
    if name == "taiwan":
        return load_taiwan(path=path or "data/taiwan_credit.xls", n=n, seed=seed)
    return generate_scenarios(n=n, seed=seed)
