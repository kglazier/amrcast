"""Tests for ESM-2 integration and protein extraction."""

import numpy as np
import pytest
from pathlib import Path

from amrcast.genome.models import AMRFinderHit, GenomeAMRProfile
from amrcast.genome.protein_extractor import _translate, _reverse_complement


class TestTranslation:
    def test_simple_translate(self):
        # ATG=M, AAA=K, TTT=F, TAA=stop
        assert _translate("ATGAAATTT") == "MKF"

    def test_stop_codon(self):
        assert _translate("ATGTAA") == "M"
        assert _translate("ATGAAATAAGGG") == "MK"

    def test_reverse_complement(self):
        assert _reverse_complement("ATCG") == "CGAT"
        assert _reverse_complement("AAAA") == "TTTT"


class TestESMEmbedder:
    """Tests that don't require the actual ESM-2 model."""

    def test_import(self):
        from amrcast.features.esm_embeddings import ESMEmbedder, ESM_MODELS
        assert "esm2_t33_650M_UR50D" in ESM_MODELS
        assert ESM_MODELS["esm2_t33_650M_UR50D"] == 1280

    def test_embedder_init(self):
        from amrcast.features.esm_embeddings import ESMEmbedder
        embedder = ESMEmbedder(model_name="esm2_t33_650M_UR50D")
        assert embedder.embedding_dim == 1280

    def test_embed_genome_proteins_empty(self):
        from amrcast.features.esm_embeddings import ESMEmbedder
        embedder = ESMEmbedder()
        result = embedder.embed_genome_proteins([], {})
        assert result.shape == (1280,)
        assert np.all(result == 0)

    def test_cache_key_deterministic(self):
        from amrcast.features.esm_embeddings import ESMEmbedder
        embedder = ESMEmbedder()
        key1 = embedder._cache_key("MKTLLVIVFVG")
        key2 = embedder._cache_key("MKTLLVIVFVG")
        key3 = embedder._cache_key("MKTLLVIVFVX")
        assert key1 == key2
        assert key1 != key3


class TestFeatureAggregator:
    def test_gene_only_mode(self):
        from amrcast.features.aggregator import build_combined_features

        hit = AMRFinderHit(
            contig_id="c1", start=0, stop=100, strand="+",
            element_symbol="blaTEM-1", element_name="blaTEM-1", scope="core",
            type="AMR", subtype="AMR", drug_class="BETA-LACTAM",
            drug_subclass="BETA-LACTAM", method="BLASTX",
            target_length=300, ref_length=300,
            coverage=100.0, identity=99.0,
            closest_ref="X", closest_ref_name="X",
        )
        profile = GenomeAMRProfile(sample_id="test", hits=[hit])

        df = build_combined_features([profile], use_esm=False)
        assert "blaTEM-1_present" in df.columns
        # Should NOT have ESM columns when use_esm=False
        assert not any(c.startswith("esm_") for c in df.columns)
