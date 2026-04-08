"""Tests for the explainability module."""

import numpy as np
import pytest

from amrcast.explain.clinical import (
    annotate_gene,
    classify_mic,
    get_breakpoint,
)
from amrcast.explain.shap_explainer import (
    PredictionExplanation,
    explain_prediction,
    _extract_gene_symbol,
)


class TestClinicalBreakpoints:
    def test_ampicillin_susceptible(self):
        assert classify_mic("ampicillin", 4) == "Susceptible"
        assert classify_mic("ampicillin", 8) == "Susceptible"

    def test_ampicillin_intermediate(self):
        assert classify_mic("ampicillin", 16) == "Intermediate"

    def test_ampicillin_resistant(self):
        assert classify_mic("ampicillin", 32) == "Resistant"
        assert classify_mic("ampicillin", 256) == "Resistant"

    def test_ciprofloxacin_breakpoints(self):
        assert classify_mic("ciprofloxacin", 0.125) == "Susceptible"
        assert classify_mic("ciprofloxacin", 0.25) == "Susceptible"
        assert classify_mic("ciprofloxacin", 0.5) == "Intermediate"
        assert classify_mic("ciprofloxacin", 1) == "Resistant"
        assert classify_mic("ciprofloxacin", 4) == "Resistant"

    def test_unknown_antibiotic(self):
        assert classify_mic("madeup_drug", 16) == "Unknown"

    def test_case_insensitive(self):
        assert classify_mic("Ampicillin", 4) == "Susceptible"
        assert classify_mic("CIPROFLOXACIN", 4) == "Resistant"

    def test_get_breakpoint(self):
        bp = get_breakpoint("ampicillin")
        assert bp is not None
        assert bp.susceptible_lte == 8
        assert bp.resistant_gte == 32

    def test_get_breakpoint_missing(self):
        assert get_breakpoint("unknowndrug") is None


class TestGeneAnnotation:
    def test_exact_match(self):
        ann = annotate_gene("gyrA")
        assert "gyrase" in ann.lower() or "quinolone" in ann.lower()

    def test_prefix_match(self):
        ann = annotate_gene("blaTEM-1")
        assert "beta-lactamase" in ann.lower()

    def test_prefix_match_variant(self):
        ann = annotate_gene("blaCTX-M-15")
        assert "extended-spectrum" in ann.lower() or "CTX-M" in ann

    def test_unknown_gene(self):
        ann = annotate_gene("completely_unknown_xyz")
        assert ann == "AMR-associated gene"

    def test_tet_genes(self):
        ann = annotate_gene("tet(A)")
        assert "tetracycline" in ann.lower()


class TestFeatureSymbolExtraction:
    def test_present_suffix(self):
        assert _extract_gene_symbol("blaTEM-1_present") == "blaTEM-1"

    def test_identity_suffix(self):
        assert _extract_gene_symbol("gyrA_S83L_identity") == "gyrA_S83L"

    def test_class_feature(self):
        assert _extract_gene_symbol("BETA-LACTAM_class") is None

    def test_summary_feature(self):
        assert _extract_gene_symbol("n_amr_genes") is None

    def test_esm_feature(self):
        assert _extract_gene_symbol("esm_42") is None


class TestSHAPExplainer:
    def test_explain_prediction(self):
        from amrcast.ml.xgboost_model import MICPredictor

        rng = np.random.RandomState(42)
        X_train = rng.rand(80, 10)
        y_train = rng.uniform(-2, 8, 80)

        predictor = MICPredictor(antibiotic="ampicillin")
        predictor.train(X_train, y_train)

        X_test = rng.rand(1, 10)
        log2_pred = float(predictor.predict(X_test)[0])

        feature_names = [
            "blaTEM-1_present", "blaTEM-1_identity",
            "tet(A)_present", "tet(A)_identity",
            "BETA-LACTAM_class", "TETRACYCLINE_class",
            "n_amr_genes", "n_point_mutations",
            "n_drug_classes", "sul1_present",
        ]

        explanation = explain_prediction(
            model=predictor.model,
            X=X_test,
            feature_names=feature_names,
            antibiotic="ampicillin",
            predicted_log2_mic=log2_pred,
            top_n=5,
        )

        assert explanation.antibiotic == "ampicillin"
        assert explanation.clinical_category in ("Susceptible", "Intermediate", "Resistant")
        assert len(explanation.top_contributors) == 5
        assert explanation.breakpoint_info is not None
        assert explanation.breakpoint_info["susceptible_lte"] == 8

        # Check serialization
        d = explanation.to_dict()
        assert "predicted_mic_ug_ml" in d
        assert "top_features" in d
        assert len(d["top_features"]) == 5

        # Check report generation
        report = explanation.detailed_report()
        assert "ampicillin" in report
        assert "ug/mL" in report

    def test_explanation_with_gene_annotations(self):
        from amrcast.ml.xgboost_model import MICPredictor

        rng = np.random.RandomState(123)
        X = rng.rand(80, 4)
        y = rng.uniform(0, 6, 80)

        predictor = MICPredictor(antibiotic="ciprofloxacin")
        predictor.train(X, y)

        X_test = rng.rand(1, 4)
        log2_pred = float(predictor.predict(X_test)[0])

        explanation = explain_prediction(
            model=predictor.model,
            X=X_test,
            feature_names=["gyrA_S83L_present", "gyrA_S83L_identity", "n_amr_genes", "QUINOLONE_class"],
            antibiotic="ciprofloxacin",
            predicted_log2_mic=log2_pred,
            top_n=4,
        )

        # gyrA should get an annotation
        gene_features = [c for c in explanation.top_contributors if c.gene_symbol]
        for c in gene_features:
            if c.gene_symbol and c.gene_symbol.startswith("gyrA"):
                assert c.annotation is not None
                assert "gyrase" in c.annotation.lower() or "quinolone" in c.annotation.lower()
