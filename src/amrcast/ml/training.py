"""Training pipeline — orchestrates AMRFinderPlus processing, feature extraction, and model training."""

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from amrcast.data.harmonize import harmonize_mic_data
from amrcast.features.gene_features import build_feature_matrix
from amrcast.genome.amrfinder import run_amrfinder, parse_amrfinder_file
from amrcast.genome.models import GenomeAMRProfile
from amrcast.ml.xgboost_model import MICPredictor

logger = logging.getLogger(__name__)


def run_training_pipeline(
    data_dir: Path,
    model_dir: Path,
    antibiotics: list[str] | None = None,
    organism: str = "Escherichia",
    use_cached_amrfinder: bool = True,
) -> dict:
    """Run the full training pipeline.

    1. Load and harmonize MIC metadata
    2. Run AMRFinderPlus on each genome (or load cached results)
    3. Build feature matrix
    4. Train per-antibiotic XGBoost models

    Args:
        data_dir: Root data directory (contains raw/genomes/ and raw/amr_metadata.csv).
        model_dir: Where to save trained models.
        antibiotics: Which antibiotics to train on. If None, train on all available.
        organism: Organism for AMRFinderPlus (enables point mutation detection).
        use_cached_amrfinder: If True, reuse cached AMRFinderPlus TSV output.

    Returns:
        Dict of antibiotic -> training metrics.
    """
    # Step 1: Load metadata
    metadata_path = data_dir / "raw" / "amr_metadata.csv"
    if not metadata_path.exists():
        raise FileNotFoundError(
            f"Metadata not found: {metadata_path}. Run 'amrcast data download' first."
        )

    logger.info("Loading and harmonizing MIC metadata...")
    raw_meta = pd.read_csv(metadata_path)
    meta = harmonize_mic_data(raw_meta)

    if antibiotics:
        meta = meta[meta["antibiotic"].isin([a.lower() for a in antibiotics])]

    available_antibiotics = sorted(meta["antibiotic"].unique())
    logger.info(f"Antibiotics with data: {available_antibiotics}")

    # Step 2: Run AMRFinderPlus on each genome
    genomes_dir = data_dir / "raw" / "genomes"
    amrfinder_dir = data_dir / "amrfinder_cache"
    amrfinder_dir.mkdir(parents=True, exist_ok=True)

    genome_ids = meta["genome_id"].unique()
    logger.info(f"Processing {len(genome_ids)} genomes with AMRFinderPlus...")

    profiles: dict[str, GenomeAMRProfile] = {}

    for i, gid in enumerate(genome_ids):
        gid_str = str(gid)
        fasta_path = genomes_dir / f"{gid_str}.fasta"
        cache_path = amrfinder_dir / f"{gid_str}.tsv"

        if not fasta_path.exists():
            continue

        try:
            # Use cached results if available
            if use_cached_amrfinder and cache_path.exists():
                profile = _load_cached_profile(cache_path)
            else:
                profile = run_amrfinder(fasta_path, organism=organism)
                _cache_profile(profile, cache_path)

            profiles[gid_str] = profile

            if (i + 1) % 10 == 0:
                logger.info(f"  Processed {i + 1}/{len(genome_ids)} genomes")
        except Exception as e:
            logger.warning(f"  Failed to process {gid_str}: {e}")

    logger.info(f"Successfully processed {len(profiles)}/{len(genome_ids)} genomes")

    if not profiles:
        raise ValueError("No genomes were successfully processed")

    # Step 3: Build feature matrix
    logger.info("Building feature matrix...")
    profile_list = list(profiles.values())
    feature_df = build_feature_matrix(profile_list)

    if feature_df.shape[1] == 0:
        raise ValueError("No features extracted. Check AMRFinderPlus output.")

    # Infer gene symbols and drug classes for prediction time
    gene_symbols = sorted({
        col.replace("_present", "")
        for col in feature_df.columns
        if col.endswith("_present")
    })
    drug_classes = sorted({
        col.replace("_class", "")
        for col in feature_df.columns
        if col.endswith("_class")
    })

    # Save metadata for prediction
    model_dir.mkdir(parents=True, exist_ok=True)
    with open(model_dir / "gene_symbols.json", "w") as f:
        json.dump(gene_symbols, f)
    with open(model_dir / "drug_classes.json", "w") as f:
        json.dump(drug_classes, f)
    with open(model_dir / "feature_columns.json", "w") as f:
        json.dump(list(feature_df.columns), f)

    # Step 4: Train per-antibiotic models
    all_metrics = {}

    for antibiotic in available_antibiotics:
        ab_meta = meta[meta["antibiotic"] == antibiotic].copy()
        ab_meta = ab_meta[ab_meta["genome_id"].astype(str).isin(profiles.keys())]

        # Average MIC if multiple measurements per genome
        mic_by_genome = (
            ab_meta.groupby("genome_id")["log2_mic"]
            .median()
            .reset_index()
        )

        valid_ids = [gid for gid in mic_by_genome["genome_id"] if gid in feature_df.index]
        if len(valid_ids) < 10:
            logger.warning(
                f"[{antibiotic}] Only {len(valid_ids)} samples — skipping (need >= 10)"
            )
            continue

        X = feature_df.loc[valid_ids].values
        y = mic_by_genome.set_index("genome_id").loc[valid_ids, "log2_mic"].values

        logger.info(f"[{antibiotic}] Training with {len(valid_ids)} samples, {X.shape[1]} features...")
        predictor = MICPredictor(antibiotic=antibiotic)
        metrics = predictor.train(X, y, feature_names=list(feature_df.columns))
        predictor.save(model_dir)
        all_metrics[antibiotic] = metrics

    logger.info("Training complete!")
    return all_metrics


def _cache_profile(profile: GenomeAMRProfile, cache_path: Path) -> None:
    """Cache an AMRFinderPlus profile as a simple JSON for fast reload."""
    cache_path.write_text(profile.model_dump_json(indent=2))


def _load_cached_profile(cache_path: Path) -> GenomeAMRProfile:
    """Load a cached AMRFinderPlus profile."""
    return GenomeAMRProfile.model_validate_json(cache_path.read_text())
