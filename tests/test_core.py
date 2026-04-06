"""Basic tests for core AMRCast components."""

import numpy as np
import pytest

from amrcast.config.settings import Settings, get_settings
from amrcast.data.harmonize import harmonize_mic_data, mic_to_log2, parse_mic_value
from amrcast.genome.models import AMRHit, CalledGene, GenomeAnnotation
from amrcast.features.gene_features import build_gene_feature_matrix
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


class TestGenomeModels:
    def test_called_gene(self):
        gene = CalledGene(
            gene_id="gene_00001",
            contig_id="contig_1",
            start=100,
            end=400,
            strand=1,
            protein_sequence="MKTLLVIVFVG",
        )
        assert gene.gene_id == "gene_00001"
        assert len(gene.protein_sequence) > 0

    def test_amr_hit(self):
        hit = AMRHit(
            gene_family="blaOXA",
            gene_id="gene_00001",
            query_name="blaOXA-1",
            evalue=1e-50,
            score=200.0,
            identity=0.95,
            coverage=0.98,
            protein_sequence="MKTLLVIVFVG",
        )
        assert hit.gene_family == "blaOXA"
        assert hit.evalue < 1e-10


class TestFeatureExtraction:
    def _make_annotation(self, sample_id: str, families: list[str]) -> GenomeAnnotation:
        genes = []
        amr_hits = []
        for i, fam in enumerate(families):
            gid = f"gene_{i:05d}"
            genes.append(
                CalledGene(
                    gene_id=gid,
                    contig_id="contig_1",
                    start=i * 1000,
                    end=i * 1000 + 300,
                    strand=1,
                    protein_sequence="MFAKE",
                )
            )
            amr_hits.append(
                AMRHit(
                    gene_family=fam,
                    gene_id=gid,
                    query_name=fam,
                    evalue=1e-50,
                    score=100.0 + i * 10,
                    identity=0.9,
                    coverage=0.95,
                    protein_sequence="MFAKE",
                )
            )
        return GenomeAnnotation(
            sample_id=sample_id,
            num_contigs=1,
            total_length=5000000,
            genes=genes,
            amr_hits=amr_hits,
        )

    def test_feature_matrix_shape(self):
        ann1 = self._make_annotation("sample1", ["blaOXA", "gyrA", "tetA"])
        ann2 = self._make_annotation("sample2", ["blaOXA", "mcr1"])

        df = build_gene_feature_matrix([ann1, ann2])
        # 4 unique families × 2 columns each = 8 columns
        assert df.shape == (2, 8)
        assert "blaOXA_present" in df.columns
        assert "blaOXA_score" in df.columns

    def test_presence_values(self):
        ann1 = self._make_annotation("sample1", ["blaOXA"])
        ann2 = self._make_annotation("sample2", [])

        df = build_gene_feature_matrix([ann1, ann2], gene_families=["blaOXA"])
        assert df.loc["sample1", "blaOXA_present"] == 1.0
        assert df.loc["sample2", "blaOXA_present"] == 0.0

    def test_fixed_families(self):
        ann = self._make_annotation("s1", ["blaOXA"])
        df = build_gene_feature_matrix([ann], gene_families=["blaOXA", "gyrA", "tetA"])
        assert df.shape[1] == 6  # 3 families × 2 columns


class TestMICPredictor:
    def test_train_and_predict(self):
        rng = np.random.RandomState(42)
        n_samples = 100
        n_features = 20
        X = rng.rand(n_samples, n_features)
        y = rng.uniform(-2, 8, n_samples)  # log2(MIC) values

        predictor = MICPredictor(antibiotic="test_drug")
        metrics = predictor.train(X, y)

        assert "mae" in metrics
        assert "essential_agreement" in metrics
        assert metrics["n_train"] > 0

        # Predict
        preds = predictor.predict(X[:5])
        assert len(preds) == 5
        # Predictions should be rounded to integers (doubling dilutions)
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
        assert all(m > 0 for m in mic_values)  # MIC must be positive

    def test_round_to_dilution(self):
        predictor = MICPredictor(antibiotic="test")
        rounded = predictor._round_to_dilution(np.array([0.7, 2.3, -1.4, 15.0]))
        assert rounded[0] == 1.0  # rounds to nearest
        assert rounded[1] == 2.0
        assert rounded[2] == -1.0
        assert rounded[3] == 10.0  # clipped to max
