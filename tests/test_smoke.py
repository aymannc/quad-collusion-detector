"""Fast smoke tests: the detector must separate collusion from independence,
the baselines must run, and the mock real-LLM pipeline must produce the
expected fire pattern (M2 fires, M1/M3 do not)."""
import numpy as np
import re

from quad import dgp
from quad.cmi import cmi_permutation_test
from quad import baselines as B
from quad import scenarios as S
from quad import llm_agents as L
from quad.analysis import run_all, fires


def test_cmi_separates():
    rng = np.random.default_rng(0)
    _, y1, y2 = dgp.independent(1, n=300)
    X, c1, c2 = dgp.colluding(2, signal_strength=0.5, n=300)
    Xi, *_ = dgp.independent(1, n=300)
    p_indep = cmi_permutation_test(*dgp.independent(1, n=300)[1:], dgp.independent(1, n=300)[0],
                                   n_perm=100, rng=rng)[1]
    p_collude = cmi_permutation_test(c1, c2, X, n_perm=100, rng=rng)[1]
    assert p_collude < 0.05 < p_indep or (p_collude < p_indep)


def test_partial_corr_matches_on_linear():
    X, y1, y2 = dgp.colluding(3, signal_strength=0.5, n=300)
    _, p = B.partial_corr_test(y1, y2, X)
    assert p < 0.05  # linear test catches linear collusion


def test_nonlinear_needs_nonlinear_test():
    # Over several seeds the nonlinear conditional test (residual dCor) detects
    # nonlinear collusion that linear partial correlation largely misses.
    lin = dcor = 0
    for s in range(8):
        X, y1, y2 = dgp.nonlinear_collusion(s, signal_strength=1.0, n=400)
        rng = np.random.default_rng(1000 + s)          # seed -> deterministic
        lin += B.partial_corr_test(y1, y2, X)[1] < 0.05
        dcor += B.residual_dcor_test(y1, y2, X, n_perm=100, rng=rng)[1] < 0.05
    assert dcor >= lin and dcor >= 6  # dCor catches most; linear test is weaker


def test_mock_pipeline_pattern():
    data = S.generate_scenarios(n=300, seed=0)
    ref = S.ground_truth_risk(data).astype(float).reshape(-1, 1)
    X = (ref - ref.mean()) / (ref.std() + 1e-9)   # condition on the reference statistic
    rng = np.random.default_rng(0)
    out = {}
    for cond in ("M1", "M2", "M3"):
        y1 = L.score_scenarios(cond, "A", data, provider="mock")[0]
        y2 = L.score_scenarios(cond, "B", data, provider="mock")[0]
        y1 = y1 + rng.normal(0, 0.5, len(y1)); y2 = y2 + rng.normal(0, 0.5, len(y2))
        out[cond] = fires(run_all(y1, y2, X, n_perm=100, rng=rng))["CMI"]
    assert out["M2"] and not out["M1"] and not out["M3"]


def test_chunked_keeps_features_in_every_chunk():
    # Real providers go through the chunking path (mock returns before it). Every
    # chunk must still carry the feature columns — regression for the bug where
    # the "_features" name-list was sliced like a per-row column, leaving the 2nd
    # chunk onward as bare IDs (and the agent refusing for lack of data).
    data = S.get_dataset("synthetic", n=130, seed=0)        # 3 chunks at chunk=50
    tally = {"numbered": 0, "withfeat": 0}
    orig = L._call_openai
    def fake(model, prompt):
        numbered = [r for r in prompt.splitlines() if re.match(r"^\s*\d+\.", r)]
        tally["numbered"] += len(numbered)
        tally["withfeat"] += sum("=" in r for r in numbered)
        return ",".join(["50"] * len(numbered)), {"in": 1, "out": 1}
    L._call_openai = fake
    try:
        scores = L.score_scenarios("M1", "A", data, provider="openai", model="x", chunk=50)[0]
    finally:
        L._call_openai = orig
    assert len(scores) == 130
    assert tally["numbered"] == tally["withfeat"] == 130    # every chunk carried features


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn(); print(f"PASS {name}")
    print("all smoke tests passed")
