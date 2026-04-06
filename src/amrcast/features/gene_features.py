"""Gene presence/absence and identity score features from AMR detection."""

import numpy as np
import pandas as pd

from amrcast.genome.models import AMRHit, GenomeAnnotation


def build_gene_feature_matrix(
    annotations: list[GenomeAnnotation],
    gene_families: list[str] | None = None,
) -> pd.DataFrame:
    """Build a gene presence/absence + identity score matrix.

    Each row is a genome. Each gene family gets two columns:
    - {family}_present: 1/0 binary presence
    - {family}_score: best hit score (0 if absent)

    Args:
        annotations: List of genome annotations.
        gene_families: Fixed list of gene families to use as columns.
            If None, inferred from all annotations.

    Returns:
        DataFrame with sample_id as index, gene feature columns.
    """
    # Collect all gene families if not provided
    if gene_families is None:
        all_families: set[str] = set()
        for ann in annotations:
            for hit in ann.amr_hits:
                all_families.add(hit.gene_family)
        gene_families = sorted(all_families)

    rows = []
    for ann in annotations:
        # Index hits by gene family (keep best score per family)
        best_by_family: dict[str, AMRHit] = {}
        for hit in ann.amr_hits:
            if hit.gene_family not in best_by_family or hit.score > best_by_family[hit.gene_family].score:
                best_by_family[hit.gene_family] = hit

        row: dict[str, float] = {"sample_id": ann.sample_id}
        for family in gene_families:
            if family in best_by_family:
                row[f"{family}_present"] = 1.0
                row[f"{family}_score"] = best_by_family[family].score
            else:
                row[f"{family}_present"] = 0.0
                row[f"{family}_score"] = 0.0

        rows.append(row)

    df = pd.DataFrame(rows).set_index("sample_id")
    return df


def extract_features_single(
    annotation: GenomeAnnotation,
    gene_families: list[str],
) -> np.ndarray:
    """Extract feature vector for a single genome.

    Returns a flat numpy array: [family1_present, family1_score, family2_present, ...].
    """
    df = build_gene_feature_matrix([annotation], gene_families=gene_families)
    return df.values[0]
