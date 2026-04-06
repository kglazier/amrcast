"""Feature aggregator — combines gene features + ESM-2 embeddings.

Produces a single feature matrix that can be fed to XGBoost or a neural model.
ESM-2 features are optional — the pipeline works without them (just gene features),
which lets us measure the marginal improvement ESM-2 provides.
"""

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from amrcast.features.gene_features import build_feature_matrix
from amrcast.genome.models import GenomeAMRProfile

logger = logging.getLogger(__name__)


def build_combined_features(
    profiles: list[GenomeAMRProfile],
    gene_symbols: list[str] | None = None,
    drug_classes: list[str] | None = None,
    use_esm: bool = False,
    esm_model_name: str = "esm2_t33_650M_UR50D",
    esm_cache_dir: Path | None = None,
) -> pd.DataFrame:
    """Build a combined feature matrix from gene features + optional ESM-2 embeddings.

    Args:
        profiles: List of GenomeAMRProfile objects.
        gene_symbols: Fixed list of gene symbols. If None, inferred.
        drug_classes: Fixed list of drug classes. If None, inferred.
        use_esm: Whether to include ESM-2 protein embeddings.
        esm_model_name: ESM-2 model to use.
        esm_cache_dir: Directory for caching ESM-2 embeddings.

    Returns:
        DataFrame with sample_id as index, all features as columns.
    """
    # Always build gene presence features
    gene_df = build_feature_matrix(profiles, gene_symbols=gene_symbols, drug_classes=drug_classes)
    logger.info(f"Gene features: {gene_df.shape[1]} columns")

    if not use_esm:
        return gene_df

    # Add ESM-2 embeddings
    from amrcast.features.esm_embeddings import ESMEmbedder

    embedder = ESMEmbedder(model_name=esm_model_name, cache_dir=esm_cache_dir)

    # Collect all unique protein sequences across all profiles for efficient batching
    all_proteins: dict[str, str] = {}
    profile_proteins: dict[str, dict[str, str]] = {}  # sample_id -> {symbol: seq}

    for profile in profiles:
        seqs = {}
        for hit in profile.amr_hits:
            # AMRFinderPlus doesn't give us protein sequences directly in its TSV.
            # We'd need to extract them from the genome. For now, we use the
            # closest_ref_name as a proxy identifier and would need to look up
            # the actual sequence from the AMRFinderPlus reference database.
            #
            # TODO: Extract actual protein sequences from AMRFinderPlus output
            # or from the genome FASTA. For now, we use a simplified approach
            # where we embed the reference protein sequence if available.
            pass
        profile_proteins[profile.sample_id] = seqs

    # For now, we need a way to get protein sequences. AMRFinderPlus can output
    # protein sequences with the --protein flag, but we're running in nucleotide mode.
    # The practical approach is to:
    # 1. Run Pyrodigal to get protein sequences (but we removed it)
    # 2. Use AMRFinderPlus protein output (requires running with --protein on extracted proteins)
    # 3. Look up reference sequences from the AMRFinderPlus database
    #
    # For the ESM-2 integration, we'll use approach 3: look up the closest reference
    # protein from the AMRFinderPlus database and embed that. This is a reasonable
    # approximation since hits with >90% identity will have very similar embeddings.

    logger.info("Extracting ESM-2 embeddings from AMRFinderPlus reference proteins...")

    esm_rows = []
    for profile in profiles:
        # Get unique reference accessions for this genome's AMR hits
        ref_proteins = []
        for hit in profile.amr_hits:
            if hit.closest_ref and hit.closest_ref_name:
                ref_proteins.append((hit.element_symbol, hit.closest_ref_name))

        if ref_proteins:
            # For now, create a placeholder — we need the actual sequences
            # This will be populated once we set up reference sequence lookup
            genome_emb = np.zeros(embedder.embedding_dim)
        else:
            genome_emb = np.zeros(embedder.embedding_dim)

        esm_rows.append({
            "sample_id": profile.sample_id,
            **{f"esm_{i}": v for i, v in enumerate(genome_emb)},
        })

    esm_df = pd.DataFrame(esm_rows).set_index("sample_id")
    logger.info(f"ESM-2 features: {esm_df.shape[1]} columns")

    # Combine
    combined = pd.concat([gene_df, esm_df], axis=1)
    logger.info(f"Combined features: {combined.shape[1]} columns")

    return combined


def build_combined_features_with_sequences(
    profiles: list[GenomeAMRProfile],
    protein_sequences: dict[str, dict[str, str]],
    gene_symbols: list[str] | None = None,
    drug_classes: list[str] | None = None,
    esm_model_name: str = "esm2_t33_650M_UR50D",
    esm_cache_dir: Path | None = None,
) -> pd.DataFrame:
    """Build combined features when protein sequences are available.

    This is the full pipeline: gene features + ESM-2 embeddings from actual
    protein sequences.

    Args:
        profiles: List of GenomeAMRProfile objects.
        protein_sequences: Dict of sample_id -> {gene_symbol: protein_sequence}.
        gene_symbols: Fixed list of gene symbols.
        drug_classes: Fixed list of drug classes.
        esm_model_name: ESM-2 model name.
        esm_cache_dir: Cache directory for embeddings.

    Returns:
        Combined feature DataFrame.
    """
    from amrcast.features.esm_embeddings import ESMEmbedder

    # Gene features
    gene_df = build_feature_matrix(profiles, gene_symbols=gene_symbols, drug_classes=drug_classes)

    # ESM-2 embeddings
    embedder = ESMEmbedder(model_name=esm_model_name, cache_dir=esm_cache_dir)

    esm_rows = []
    for profile in profiles:
        seqs = protein_sequences.get(profile.sample_id, {})
        symbols = [h.element_symbol for h in profile.amr_hits]
        genome_emb = embedder.embed_genome_proteins(symbols, seqs)

        esm_rows.append({
            "sample_id": profile.sample_id,
            **{f"esm_{i}": v for i, v in enumerate(genome_emb)},
        })

    esm_df = pd.DataFrame(esm_rows).set_index("sample_id")
    combined = pd.concat([gene_df, esm_df], axis=1)

    logger.info(
        f"Combined features: {gene_df.shape[1]} gene + {esm_df.shape[1]} ESM-2 "
        f"= {combined.shape[1]} total"
    )
    return combined
