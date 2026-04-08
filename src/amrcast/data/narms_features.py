"""Build feature matrices from NCBI pre-computed AMRFinderPlus genotypes.

Instead of running AMRFinderPlus ourselves, we use NCBI's pre-computed
genotype calls from the Pathogen Detection pipeline. The AMR_genotypes
column contains comma-separated gene symbols like "acrF,blaTEM-1,sul1".

This gives us gene presence/absence features for thousands of isolates
without downloading a single genome FASTA.
"""

import logging
import re

import numpy as np
import pandas as pd

from amrcast.data.harmonize import parse_mic_value, mic_to_log2

logger = logging.getLogger(__name__)


def parse_genotypes(genotype_str: str) -> list[str]:
    """Parse AMR_genotypes string into list of gene symbols.

    Input: '"acrF,blaTEM-1,sul1,gyrA_S83L"' or 'NULL'
    Output: ['acrF', 'blaTEM-1', 'sul1', 'gyrA_S83L']
    """
    if not genotype_str or genotype_str == "NULL" or pd.isna(genotype_str):
        return []
    # Strip quotes
    s = genotype_str.strip('"')
    if not s:
        return []
    return [g.strip() for g in s.split(",") if g.strip()]


def build_narms_training_data(
    mic_path: str | None = None,
    metadata_path: str | None = None,
    joined_df: pd.DataFrame | None = None,
    antibiotics: list[str] | None = None,
    platform_filter: str | None = "ensitit",
    min_isolates_per_drug: int = 100,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build feature matrix + target values from NARMS/NCBI data.

    Args:
        mic_path: Path to antibiogram_mic.csv.
        metadata_path: Path to ecoli_amr_metadata.tsv.
        joined_df: Pre-joined DataFrame (if already loaded).
        antibiotics: Filter to these antibiotics. None = all with enough data.
        platform_filter: Substring filter for testing platform (e.g., "ensitit" for Sensititre).
        min_isolates_per_drug: Skip antibiotics with fewer isolates.

    Returns:
        (feature_df, target_df) where:
          feature_df: DataFrame with gene presence features, indexed by biosample_acc
          target_df: DataFrame with columns [biosample_acc, antibiotic, mic_value, log2_mic, ...]
    """
    # Load and join if not pre-joined
    if joined_df is None:
        mic_df = pd.read_csv(mic_path)
        meta_df = pd.read_csv(
            metadata_path, sep="\t",
            usecols=["biosample_acc", "asm_acc", "bioproject_acc",
                     "AMR_genotypes", "AMR_genotypes_core", "number_amr_genes"],
            low_memory=False,
            on_bad_lines="skip",
            quoting=3,  # QUOTE_NONE — NCBI TSVs have unescaped quotes
        )
        joined_df = mic_df.merge(meta_df, on="biosample_acc", how="inner")

    df = joined_df.copy()

    # Filter by platform
    if platform_filter:
        df = df[df["platform"].str.contains(platform_filter, case=False, na=False)]
        logger.info(f"After platform filter: {len(df)} records")

    # Parse MIC values
    df["mic_numeric"] = df["mic_value"].apply(lambda x: parse_mic_value(x))
    df = df.dropna(subset=["mic_numeric"])
    df = df[df["mic_numeric"] > 0]
    df["log2_mic"] = df["mic_numeric"].apply(mic_to_log2)

    # Filter antibiotics
    if antibiotics:
        df = df[df["antibiotic"].isin([a.lower() for a in antibiotics])]

    # Only keep antibiotics with enough data
    ab_counts = df.groupby("antibiotic")["biosample_acc"].nunique()
    valid_abs = ab_counts[ab_counts >= min_isolates_per_drug].index.tolist()
    df = df[df["antibiotic"].isin(valid_abs)]

    logger.info(
        f"Training data: {len(df)} records, {df['biosample_acc'].nunique()} isolates, "
        f"{len(valid_abs)} antibiotics"
    )

    # Build target DataFrame (one row per isolate × antibiotic)
    # Take median MIC if duplicates exist
    target_df = (
        df.groupby(["biosample_acc", "antibiotic"])
        .agg(
            log2_mic=("log2_mic", "median"),
            mic_numeric=("mic_numeric", "median"),
            measurement_sign=("measurement_sign", "first"),
        )
        .reset_index()
    )

    # Build feature matrix from AMR_genotypes
    # Get unique genotypes per isolate
    isolate_genotypes = (
        df.drop_duplicates(subset=["biosample_acc"])
        .set_index("biosample_acc")["AMR_genotypes"]
    )

    # Collect all gene symbols across all isolates
    all_genes: set[str] = set()
    parsed_genotypes: dict[str, list[str]] = {}
    for acc, gstr in isolate_genotypes.items():
        genes = parse_genotypes(gstr)
        parsed_genotypes[acc] = genes
        all_genes.update(genes)

    # Separate point mutations from acquired genes
    point_mutation_pattern = re.compile(r"_[A-Z]\d+[A-Z]")
    gene_symbols = sorted(g for g in all_genes if not point_mutation_pattern.search(g))
    point_mutations = sorted(g for g in all_genes if point_mutation_pattern.search(g))

    logger.info(
        f"Features: {len(gene_symbols)} genes + {len(point_mutations)} point mutations"
    )

    # Build feature matrix
    rows = []
    for acc in target_df["biosample_acc"].unique():
        genes = set(parsed_genotypes.get(acc, []))
        row = {"biosample_acc": acc}

        # Gene presence (binary)
        for g in gene_symbols:
            row[f"{g}_present"] = 1.0 if g in genes else 0.0

        # Point mutation presence (binary)
        for pm in point_mutations:
            row[f"{pm}_present"] = 1.0 if pm in genes else 0.0

        # Summary features
        row["n_amr_genes"] = float(len(genes))
        row["n_point_mutations"] = float(sum(1 for g in genes if point_mutation_pattern.search(g)))

        # Drug-class-aware mutation counts would require drug class info
        # which isn't in the genotype string. We can add that later.

        rows.append(row)

    feature_df = pd.DataFrame(rows).set_index("biosample_acc")

    logger.info(f"Feature matrix: {feature_df.shape}")
    return feature_df, target_df


def build_features_from_amrfinder(
    gene_symbols_detected: list[str],
    feature_columns: list[str],
    hit_methods: dict[str, str] | None = None,
) -> np.ndarray:
    """Build a NARMS-compatible feature vector from AMRFinderPlus output.

    This is used at prediction time: we run AMRFinderPlus on a new genome
    and need to produce features matching what the NARMS models expect.

    NCBI's genotype strings use "=POINT" suffix for point mutations
    (e.g., "gyrA_S83L=POINT") while AMRFinderPlus outputs just "gyrA_S83L"
    with method "POINTX"/"POINTN". We normalize by adding the suffix.

    Args:
        gene_symbols_detected: Gene symbols from AMRFinderPlus.
        feature_columns: The saved feature column names from training.
        hit_methods: Optional dict of symbol -> AMRFinderPlus method
            (e.g., {"gyrA_S83L": "POINTX"}). Used to add "=POINT" suffix.

    Returns:
        1D numpy array matching feature_columns order.
    """
    # Normalize gene symbols to match NCBI format
    normalized = set()
    for sym in gene_symbols_detected:
        method = (hit_methods or {}).get(sym, "")
        if method in ("POINTX", "POINTN"):
            normalized.add(f"{sym}=POINT")
        else:
            normalized.add(sym)
        # Also add the raw symbol in case the model uses it
        normalized.add(sym)

    point_mutation_pattern = re.compile(r"_[A-Z]\d+[A-Z]")

    features = np.zeros(len(feature_columns))
    for i, col in enumerate(feature_columns):
        if col == "n_amr_genes":
            features[i] = float(len(gene_symbols_detected))
        elif col == "n_point_mutations":
            n_pts = sum(
                1 for g in gene_symbols_detected
                if (hit_methods or {}).get(g, "") in ("POINTX", "POINTN")
                or point_mutation_pattern.search(g)
            )
            features[i] = float(n_pts)
        elif col.endswith("_present"):
            gene = col[: -len("_present")]
            features[i] = 1.0 if gene in normalized else 0.0

    return features
