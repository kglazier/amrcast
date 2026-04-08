"""Basic tests for core AMRCast components."""

import numpy as np
import pytest

from amrcast.config.settings import Settings, get_settings
from amrcast.data.harmonize import harmonize_mic_data, mic_to_log2, parse_mic_value
from amrcast.genome.models import AMRFinderHit, GenomeAMRProfile
from amrcast.features.gene_features import build_feature_matrix
from amrcast.ml.xgboost_model import MICPredictor


class TestMICParsing:
    def test_parse_numeric(self):
        assert parse_mic_value("4") == 4.0
        assert parse_mic_value("0.5") == 0.5
        assert parse_mic_value(16) == 16.0

    def test_parse_with_operators(self):
        assert parse_mic_value(">=32") == 32.0
        assert parse_mic_value("<=0.25") == 0.25
        assert parse_mic_value(">256") == 256.0

    def test_parse_invalid(self):
        assert parse_mic_value(None) is None
        assert parse_mic_value("Resistant") is None
        assert parse_mic_value(float("nan")) is None

    def test_log2_conversion(self):
        assert mic_to_log2(1.0) == 0.0
        assert mic_to_log2(2.0) == 1.0
        assert mic_to_log2(0.5) == -1.0
        assert mic_to_log2(256.0) == 8.0


class TestAMRFinderModels:
    def _make_hit(self, symbol: str, drug_class: str = "BETA-LACTAM", method: str = "BLASTX") -> AMRFinderHit:
        return AMRFinderHit(
            contig_id="contig_1",
            start=100,
            stop=1000,
            strand="+",
            element_symbol=symbol,
            element_name=f"{symbol} gene",
            scope="core",
            type="AMR",
            subtype="AMR",
            drug_class=drug_class,
            drug_subclass=drug_class,
            method=method,
            target_length=300,
            ref_length=300,
            coverage=100.0,
            identity=99.5,
            closest_ref="ABC123",
            closest_ref_name=f"reference {symbol}",
        )

    def test_genome_profile(self):
        hits = [
            self._make_hit("blaTEM-1", "BETA-LACTAM"),
            self._make_hit("tet(A)", "TETRACYCLINE"),
            self._make_hit("gyrA_S83L", "QUINOLONE", method="POINTX"),
        ]
        profile = GenomeAMRProfile(sample_id="test", hits=hits)

        assert len(profile.amr_hits) == 3
        assert len(profile.point_mutations) == 1
        assert "BETA-LACTAM" in profile.drug_classes
        assert "blaTEM-1" in profile.gene_symbols

    def test_empty_profile(self):
        profile = GenomeAMRProfile(sample_id="empty", hits=[])
        assert len(profile.amr_hits) == 0
        assert len(profile.point_mutations) == 0
        assert profile.drug_classes == []


class TestFeatureExtraction:
    def _make_profile(self, sample_id: str, symbols: list[str]) -> GenomeAMRProfile:
        hits = []
        for sym in symbols:
            hits.append(AMRFinderHit(
                contig_id="c1", start=0, stop=100, strand="+",
                element_symbol=sym, element_name=sym, scope="core",
                type="AMR", subtype="AMR", drug_class="BETA-LACTAM",
                drug_subclass="BETA-LACTAM", method="BLASTX",
                target_length=300, ref_length=300,
                coverage=100.0, identity=99.0,
                closest_ref="X", closest_ref_name="X",
            ))
        return GenomeAMRProfile(sample_id=sample_id, hits=hits)

    def test_feature_matrix_shape(self):
        p1 = self._make_profile("s1", ["blaTEM-1", "tet(A)", "sul1"])
        p2 = self._make_profile("s2", ["blaTEM-1", "mcr-1"])

        df = build_feature_matrix([p1, p2])
        # 4 gene symbols × 2 cols + 1 drug class × 1 col + 3 summary = expected cols
        assert df.shape[0] == 2
        assert "blaTEM-1_present" in df.columns
        assert "blaTEM-1_identity" in df.columns
        assert "n_amr_genes" in df.columns
        assert "n_point_mutations" in df.columns

    def test_presence_values(self):
        p1 = self._make_profile("s1", ["blaTEM-1"])
        p2 = self._make_profile("s2", [])

        df = build_feature_matrix([p1, p2], gene_symbols=["blaTEM-1"])
        assert df.loc["s1", "blaTEM-1_present"] == 1.0
        assert df.loc["s2", "blaTEM-1_present"] == 0.0

    def test_fixed_symbols(self):
        p = self._make_profile("s1", ["blaTEM-1"])
        df = build_feature_matrix([p], gene_symbols=["blaTEM-1", "sul1", "tet(A)"])
        # 3 genes × 2 cols + 1 drug class + 3 summary
        assert "sul1_present" in df.columns
        assert df.loc["s1", "sul1_present"] == 0.0


class TestMICPredictor:
    def test_train_and_predict(self):
        rng = np.random.RandomState(42)
        n_samples = 100
        n_features = 20
        X = rng.rand(n_samples, n_features)
        y = rng.uniform(-2, 8, n_samples)

        predictor = MICPredictor(antibiotic="test_drug")
        metrics = predictor.train(X, y)

        assert "mae" in metrics
        assert "essential_agreement" in metrics
        assert metrics["n_train"] > 0

        preds = predictor.predict(X[:5])
        assert len(preds) == 5
        for p in preds:
            assert p == int(p)

    def test_predict_mic(self):
        rng = np.random.RandomState(42)
        X = rng.rand(50, 10)
        y = rng.uniform(0, 6, 50)

        predictor = MICPredictor(antibiotic="test_drug")
        predictor.train(X, y)

        mic_values = predictor.predict_mic(X[:3])
        assert len(mic_values) == 3
        assert all(m > 0 for m in mic_values)

    def test_round_to_dilution(self):
        predictor = MICPredictor(antibiotic="test")
        rounded = predictor._round_to_dilution(np.array([0.7, 2.3, -1.4, 15.0]))
        assert rounded[0] == 1.0
        assert rounded[1] == 2.0
        assert rounded[2] == -1.0
        assert rounded[3] == 10.0

    def test_cross_validate(self):
        rng = np.random.RandomState(42)
        n_samples = 60
        n_features = 10
        X = rng.rand(n_samples, n_features)
        y = rng.uniform(-2, 8, n_samples)

        predictor = MICPredictor(antibiotic="test_cv")
        cv_metrics = predictor.cross_validate(X, y, n_folds=3)

        assert "mae_mean" in cv_metrics
        assert "mae_std" in cv_metrics
        assert "essential_agreement_mean" in cv_metrics
        assert "exact_match_mean" in cv_metrics
        assert cv_metrics["n_folds"] == 3
        assert cv_metrics["n_samples"] == n_samples
        assert len(cv_metrics["fold_metrics"]) == 3
        # Final model should be trained
        assert predictor.model is not None
        preds = predictor.predict(X[:3])
        assert len(preds) == 3
