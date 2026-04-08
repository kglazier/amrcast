"""Explainability module — SHAP attributions + clinical interpretation."""

from amrcast.explain.clinical import (
    BREAKPOINTS,
    annotate_gene,
    classify_mic,
    get_breakpoint,
)
from amrcast.explain.shap_explainer import (
    FeatureContribution,
    PredictionExplanation,
    explain_prediction,
)

__all__ = [
    "BREAKPOINTS",
    "annotate_gene",
    "classify_mic",
    "get_breakpoint",
    "explain_prediction",
    "FeatureContribution",
    "PredictionExplanation",
]
