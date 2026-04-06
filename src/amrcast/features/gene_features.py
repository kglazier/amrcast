"""Feature extraction from AMRFinderPlus output.

Builds feature matrices from gene presence/absence, identity scores,
point mutations, and drug class indicators.
"""

import numpy as np
import pandas as pd

from amrcast.genome.models import GenomeAMRProfile


def build_feature_matrix(
    profiles: list[GenomeAMRProfile],
    gene_symbols: list[str] | None = None,
    drug_classes: list[str] | None = None,
) -> pd.DataFrame:
    """Build a feature matrix from AMRFinderPlus profiles.

    Features per genome:
    - {gene}_present: 1/0 for each known AMR gene symbol
    - {gene}_identity: best identity score (0-100) for each gene
    - {drug_class}_class: 1/0 for each drug class with detected resistance
    - n_amr_genes: total count of AMR genes detected
    - n_point_mutations: count of point mutations detected
    - n_drug_classes: count of distinct drug classes

    Args:
        profiles: List of GenomeAMRProfile objects.
        gene_symbols: Fixed list of gene symbols for columns. If None, inferred.
        drug_classes: Fixed list of drug classes. If None, inferred.

    Returns:
        DataFrame with sample_id as index.
    """
    # Collect all gene symbols and drug classes if not provided
    if gene_symbols is None:
        all_symbols: set[str] = set()
        for p in profiles:
            for h in p.amr_hits:
                all_symbols.add(h.element_symbol)
        gene_symbols = sorted(all_symbols)

    if drug_classes is None:
        all_classes: set[str] = set()
        for p in profiles:
            for h in p.amr_hits:
                if h.drug_class != "NA":
                    all_classes.add(h.drug_class)
        drug_classes = sorted(all_classes)

    rows = []
    for profile in profiles:
        row: dict[str, float] = {"sample_id": profile.sample_id}

        # Index AMR hits by gene symbol (keep best identity per symbol)
        best_by_symbol: dict[str, float] = {}
        for h in profile.amr_hits:
            sym = h.element_symbol
            if sym not in best_by_symbol or h.identity > best_by_symbol[sym]:
                best_by_symbol[sym] = h.identity

        # Gene presence/absence + identity
        for sym in gene_symbols:
            if sym in best_by_symbol:
                row[f"{sym}_present"] = 1.0
                row[f"{sym}_identity"] = best_by_symbol[sym]
            else:
                row[f"{sym}_present"] = 0.0
                row[f"{sym}_identity"] = 0.0

        # Drug class indicators
        detected_classes = {h.drug_class for h in profile.amr_hits if h.drug_class != "NA"}
        for dc in drug_classes:
            row[f"{dc}_class"] = 1.0 if dc in detected_classes else 0.0

        # Summary features
        row["n_amr_genes"] = float(len(profile.amr_hits))
        row["n_point_mutations"] = float(len(profile.point_mutations))
        row["n_drug_classes"] = float(len(detected_classes))

        rows.append(row)

    df = pd.DataFrame(rows).set_index("sample_id")
    return df


def extract_features_single(
    profile: GenomeAMRProfile,
    gene_symbols: list[str],
    drug_classes: list[str],
) -> np.ndarray:
    """Extract feature vector for a single genome."""
    df = build_feature_matrix([profile], gene_symbols=gene_symbols, drug_classes=drug_classes)
    return df.values[0]
