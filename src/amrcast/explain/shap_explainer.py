"""SHAP-based model explanations for MIC predictions.

Turns raw XGBoost predictions into interpretable reports:
- Which genes/features drive the prediction up or down
- Clinical context (gene annotations, resistance mechanisms)
- Structured output for downstream reporting
"""

import logging
from dataclasses import dataclass, field

import numpy as np

from amrcast.explain.clinical import annotate_gene, classify_mic, get_breakpoint

logger = logging.getLogger(__name__)


@dataclass
class FeatureContribution:
    """A single feature's contribution to the prediction."""
    feature_name: str
    shap_value: float
    feature_value: float
    gene_symbol: str | None = None
    annotation: str | None = None

    @property
    def direction(self) -> str:
        if self.shap_value > 0.05:
            return "increases resistance"
        elif self.shap_value < -0.05:
            return "decreases resistance"
        return "minimal effect"

    def to_dict(self) -> dict:
        d = {
            "feature": self.feature_name,
            "shap_value": round(self.shap_value, 3),
            "feature_value": round(self.feature_value, 3),
            "direction": self.direction,
        }
        if self.gene_symbol:
            d["gene"] = self.gene_symbol
        if self.annotation:
            d["annotation"] = self.annotation
        return d


@dataclass
class PredictionExplanation:
    """Complete explanation for one antibiotic prediction."""
    antibiotic: str
    predicted_mic_ug_ml: float
    predicted_log2_mic: float
    clinical_category: str
    base_value: float
    top_contributors: list[FeatureContribution] = field(default_factory=list)
    breakpoint_info: dict | None = None

    def to_dict(self) -> dict:
        d = {
            "antibiotic": self.antibiotic,
            "predicted_mic_ug_ml": round(self.predicted_mic_ug_ml, 4),
            "predicted_log2_mic": round(self.predicted_log2_mic, 2),
            "clinical_category": self.clinical_category,
            "base_value": round(self.base_value, 2),
            "top_features": [c.to_dict() for c in self.top_contributors],
        }
        if self.breakpoint_info:
            d["breakpoint"] = self.breakpoint_info
        return d

    def summary(self) -> str:
        """Human-readable one-line summary."""
        mic_str = f"{self.predicted_mic_ug_ml:.1f}" if self.predicted_mic_ug_ml >= 1 else f"{self.predicted_mic_ug_ml:.3f}"
        return (
            f"{self.antibiotic}: {mic_str} µg/mL → {self.clinical_category}"
        )

    def detailed_report(self) -> str:
        """Multi-line human-readable report."""
        lines = [self.summary(), ""]

        if self.breakpoint_info:
            bp = self.breakpoint_info
            lines.append(
                f"  CLSI breakpoints: S ≤ {bp['susceptible_lte']} | R ≥ {bp['resistant_gte']} {bp['unit']}"
            )
            lines.append("")

        if self.top_contributors:
            lines.append("  Top contributing features:")
            for c in self.top_contributors:
                sign = "+" if c.shap_value >= 0 else ""
                feat_info = f"= {c.feature_value:.1f}" if c.feature_value != 0 else "= 0"
                lines.append(
                    f"    {c.feature_name:<30} ({sign}{c.shap_value:.2f} log2)  [{feat_info}]"
                )
                if c.annotation:
                    lines.append(f"      → {c.annotation}")
            lines.append("")

        return "\n".join(lines)


def explain_prediction(
    model,
    X: np.ndarray,
    feature_names: list[str],
    antibiotic: str,
    predicted_log2_mic: float,
    top_n: int = 8,
) -> PredictionExplanation:
    """Generate a structured SHAP explanation for a single prediction.

    Args:
        model: Trained XGBoost model (or any tree model).
        X: Feature vector for one sample, shape (1, n_features).
        feature_names: Names corresponding to feature columns.
        antibiotic: Antibiotic name.
        predicted_log2_mic: The model's prediction (log2 scale).
        top_n: Number of top features to include.

    Returns:
        PredictionExplanation with structured attribution data.
    """
    import shap

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)

    if shap_values.ndim == 1:
        sv = shap_values
    else:
        sv = shap_values[0]

    base_value = float(explainer.expected_value)
    mic_ug_ml = float(2 ** predicted_log2_mic)
    clinical_cat = classify_mic(antibiotic, mic_ug_ml)

    # Get top features by absolute SHAP value
    abs_shap = np.abs(sv)
    top_idx = abs_shap.argsort()[-top_n:][::-1]

    contributors = []
    for i in top_idx:
        fname = feature_names[i] if i < len(feature_names) else f"feature_{i}"
        fval = float(X[0, i]) if X.ndim == 2 else float(X[i])

        # Extract gene symbol from feature name
        gene_sym = _extract_gene_symbol(fname)
        annotation = annotate_gene(gene_sym) if gene_sym else None

        contributors.append(FeatureContribution(
            feature_name=fname,
            shap_value=float(sv[i]),
            feature_value=fval,
            gene_symbol=gene_sym,
            annotation=annotation,
        ))

    # Breakpoint info
    bp = get_breakpoint(antibiotic)
    bp_info = None
    if bp:
        bp_info = {
            "susceptible_lte": bp.susceptible_lte,
            "resistant_gte": bp.resistant_gte,
            "unit": bp.unit,
            "source": bp.source,
        }

    return PredictionExplanation(
        antibiotic=antibiotic,
        predicted_mic_ug_ml=mic_ug_ml,
        predicted_log2_mic=predicted_log2_mic,
        clinical_category=clinical_cat,
        base_value=base_value,
        top_contributors=contributors,
        breakpoint_info=bp_info,
    )


def _extract_gene_symbol(feature_name: str) -> str | None:
    """Extract a gene symbol from a feature column name.

    Examples:
        "blaTEM-1_present" -> "blaTEM-1"
        "blaTEM-1_identity" -> "blaTEM-1"
        "BETA-LACTAM_class" -> None
        "n_amr_genes" -> None
        "esm_42" -> None
    """
    for suffix in ("_present", "_identity"):
        if feature_name.endswith(suffix):
            return feature_name[: -len(suffix)]
    return None
