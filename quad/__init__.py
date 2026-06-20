"""QUAD collusion detector — conditional-mutual-information test for hidden
coordination among multi-agent (LLM) decision systems, with baseline
comparators.

Public API:
    cmi_estimate, cmi_permutation_test   - the detector (quad.cmi)
    baselines                            - Pearson, partial corr, dCor, HSIC, KCI
    dgp                                  - synthetic data-generating regimes
    scenarios                            - programmatic credit scenarios
    analysis                             - run detector + baselines, build tables
"""
from .cmi import cmi_estimate, cmi_permutation_test
from . import baselines, dgp, scenarios, analysis

__all__ = [
    "cmi_estimate", "cmi_permutation_test",
    "baselines", "dgp", "scenarios", "analysis",
]
__version__ = "0.1.0"
