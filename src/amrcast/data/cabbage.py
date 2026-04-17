"""CABBAGE dataset integration — load and convert to AMRCast training format.

CABBAGE (Comprehensive Assessment of Bacterial-Based AMR prediction from GEnotypes)
is a unified AMR genotype-phenotype database from EBI, incorporating NARMS and other
sources. Data format: one row per (isolate × drug × gene) with MIC measurements.

We convert this to the same format as our NARMS pipeline:
  - antibiogram_mic.csv: one row per (isolate × drug) with MIC value
  - genotype lookup: isolate -> comma-separated gene list (like NCBI AMR_genotypes)
"""

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from amrcast.data.harmonize import parse_mic_value, mic_to_log2

logger = logging.getLogger(__name__)


def load_cabbage(
    cabbage_path: Path,
    genus: str | None = None,
) -> pd.DataFrame:
    """Load CABBAGE merged CSV and optionally filter by genus."""
    df = pd.read_csv(cabbage_path, low_memory=False)
    if genus:
        df = df[df["genus"].str.lower() == genus.lower()]
    logger.info(f"CABBAGE: {len(df):,} rows, {df['BioSample_ID'].nunique():,} isolates")
    return df


def cabbage_to_training_data(
    cabbage_path: Path,
    genus: str,
    existing_mic_path: Path | None = None,
    min_isolates_per_drug: int = 100,
    narms_threshold: int = 500,
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Convert CABBAGE data to AMRCast training format.

    Selectively merges with existing NARMS data: CABBAGE is only added for
    drugs where NARMS has fewer than narms_threshold isolates. This avoids
    degrading well-performing drugs with mixed-source data while filling gaps.

    Returns:
        (feature_df, target_df, isolate_groups) — same format as build_narms_training_data
    """
    import re
    from amrcast.data.narms_features import parse_genotypes

    df = load_cabbage(cabbage_path, genus=genus)

    # Only keep rows with MIC measurement + genotype
    df = df[df["measurement"].notna() & df["gene_symbol"].notna()]

    # Build per-isolate genotype strings (aggregate genes per isolate)
    genotypes_by_isolate = (
        df.groupby("BioSample_ID")["amr_element_symbol"]
        .apply(lambda x: ",".join(sorted(set(x.dropna()))))
        .to_dict()
    )

    # Build MIC table: one row per (isolate × drug)
    mic_rows = []
    seen = set()
    for _, row in df.iterrows():
        key = (row["BioSample_ID"], row["antibiotic_name"])
        if key in seen:
            continue
        seen.add(key)
        mic_rows.append({
            "biosample_acc": row["BioSample_ID"],
            "antibiotic": str(row["antibiotic_name"]).lower().strip(),
            "mic_value": row["measurement"],
            "measurement_sign": row.get("measurement_sign", "=="),
            "resistance_phenotype": row.get("resistance_phenotype", ""),
            "method": row.get("laboratory_typing_method", ""),
            "platform": row.get("platform", "broth dilution"),
            "testing_standard": row.get("ast_standard", ""),
        })

    cabbage_mic = pd.DataFrame(mic_rows)
    logger.info(f"CABBAGE MIC table: {len(cabbage_mic):,} records, {cabbage_mic['biosample_acc'].nunique():,} isolates")

    # Selective merge: only add CABBAGE for drugs where NARMS is thin
    if existing_mic_path and existing_mic_path.exists():
        existing = pd.read_csv(existing_mic_path)
        existing_isolates = set(existing["biosample_acc"])

        # Find which drugs are data-limited in NARMS
        narms_per_drug = existing.groupby("antibiotic")["biosample_acc"].nunique()
        thin_drugs = set(narms_per_drug[narms_per_drug < narms_threshold].index)

        # Also include drugs that only exist in CABBAGE
        cabbage_drugs = set(cabbage_mic["antibiotic"].unique())
        narms_drugs = set(existing["antibiotic"].unique())
        new_drugs = cabbage_drugs - narms_drugs

        use_cabbage_for = thin_drugs | new_drugs

        if use_cabbage_for:
            logger.info(
                f"Adding CABBAGE data for {len(use_cabbage_for)} data-limited drugs: "
                f"{sorted(use_cabbage_for)[:10]}{'...' if len(use_cabbage_for) > 10 else ''}"
            )
            # Only take CABBAGE rows for thin drugs, excluding already-known isolates
            new_mic = cabbage_mic[
                cabbage_mic["antibiotic"].isin(use_cabbage_for)
                & ~cabbage_mic["biosample_acc"].isin(existing_isolates)
            ]
            combined = pd.concat([existing, new_mic], ignore_index=True)
            logger.info(
                f"Selective merge: {len(existing):,} NARMS + {len(new_mic):,} CABBAGE "
                f"= {len(combined):,} total"
            )
        else:
            logger.info("All drugs have sufficient NARMS data, skipping CABBAGE")
            combined = existing
    else:
        combined = cabbage_mic

    # Parse MIC values
    combined["mic_numeric"] = combined["mic_value"].apply(lambda x: parse_mic_value(x))
    combined = combined.dropna(subset=["mic_numeric"])
    combined = combined[combined["mic_numeric"] > 0]
    combined["log2_mic"] = combined["mic_numeric"].apply(mic_to_log2)

    # Filter antibiotics
    ab_counts = combined.groupby("antibiotic")["biosample_acc"].nunique()
    valid_abs = ab_counts[ab_counts >= min_isolates_per_drug].index.tolist()
    combined = combined[combined["antibiotic"].isin(valid_abs)]

    # Build target DataFrame
    target_df = (
        combined.groupby(["biosample_acc", "antibiotic"])
        .agg(
            log2_mic=("log2_mic", "median"),
            mic_numeric=("mic_numeric", "median"),
            measurement_sign=("measurement_sign", "first"),
        )
        .reset_index()
    )

    # Interval labels for censored MIC
    def _mic_interval(row):
        sign = str(row["measurement_sign"]).strip()
        log2 = row["log2_mic"]
        if sign in ("<=", "<"):
            return -np.inf, log2
        elif sign in (">=", ">"):
            return log2, np.inf
        else:
            return log2, log2

    intervals = target_df.apply(_mic_interval, axis=1, result_type="expand")
    target_df["log2_mic_lower"] = intervals[0]
    target_df["log2_mic_upper"] = intervals[1]

    # Build genotype lookup for all isolates (CABBAGE + existing NARMS)
    # For existing NARMS isolates, we need their genotypes from the metadata TSV
    all_genotypes = dict(genotypes_by_isolate)

    # Build feature matrix
    point_mutation_pattern = re.compile(r"_[A-Z]\d+[A-Z]")
    all_genes: set[str] = set()
    parsed_genotypes: dict[str, list[str]] = {}

    for acc in target_df["biosample_acc"].unique():
        gstr = all_genotypes.get(acc, "")
        genes = [g.strip() for g in gstr.split(",") if g.strip()] if gstr else []
        parsed_genotypes[acc] = genes
        all_genes.update(genes)

    gene_symbols = sorted(g for g in all_genes if not point_mutation_pattern.search(g))
    point_mutations = sorted(g for g in all_genes if point_mutation_pattern.search(g))

    logger.info(f"Features: {len(gene_symbols)} genes + {len(point_mutations)} point mutations")

    rows = []
    for acc in target_df["biosample_acc"].unique():
        genes = set(parsed_genotypes.get(acc, []))
        row = {"biosample_acc": acc}
        for g in gene_symbols:
            row[f"{g}_present"] = 1.0 if g in genes else 0.0
        for pm in point_mutations:
            row[f"{pm}_present"] = 1.0 if pm in genes else 0.0
        row["n_amr_genes"] = float(len(genes))
        row["n_point_mutations"] = float(sum(1 for g in genes if point_mutation_pattern.search(g)))
        rows.append(row)

    feature_df = pd.DataFrame(rows).set_index("biosample_acc")

    # Build genotype groups
    genotype_to_group: dict[str, int] = {}
    isolate_groups: dict[str, int] = {}
    for acc in feature_df.index:
        gstr = all_genotypes.get(acc, "")
        if gstr not in genotype_to_group:
            genotype_to_group[gstr] = len(genotype_to_group)
        isolate_groups[acc] = genotype_to_group[gstr]

    logger.info(
        f"Feature matrix: {feature_df.shape}, "
        f"{len(genotype_to_group)} unique genotype groups"
    )

    return feature_df, target_df, isolate_groups
