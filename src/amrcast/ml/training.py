"""Training pipeline — orchestrates data loading, feature extraction, and model training."""

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from amrcast.data.harmonize import harmonize_mic_data
from amrcast.features.gene_features import build_gene_feature_matrix
from amrcast.genome.models import GenomeAnnotation
from amrcast.genome.pipeline import process_genome
from amrcast.ml.xgboost_model import MICPredictor

logger = logging.getLogger(__name__)


def run_training_pipeline(
    data_dir: Path,
    card_dir: Path,
    model_dir: Path,
    antibiotics: list[str] | None = None,
) -> dict:
    """Run the full training pipeline.

    1. Load and harmonize MIC metadata
    2. Process genomes (gene calling + AMR detection)
    3. Build feature matrix
    4. Train per-antibiotic XGBoost models

    Args:
        data_dir: Root data directory (contains raw/genomes/ and raw/amr_metadata.csv).
        card_dir: CARD database directory.
        model_dir: Where to save trained models.
        antibiotics: Which antibiotics to train on. If None, train on all available.

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

    # Step 2: Process genomes
    genomes_dir = data_dir / "raw" / "genomes"
    genome_ids = meta["genome_id"].unique()

    logger.info(f"Processing {len(genome_ids)} genomes...")
    annotations: dict[str, GenomeAnnotation] = {}

    for i, gid in enumerate(genome_ids):
        fasta_path = genomes_dir / f"{gid}.fasta"
        if not fasta_path.exists():
            logger.warning(f"  Genome FASTA not found: {fasta_path}")
            continue

        try:
            ann = process_genome(fasta_path, card_dir=card_dir)
            annotations[str(gid)] = ann
            if (i + 1) % 10 == 0:
                logger.info(f"  Processed {i + 1}/{len(genome_ids)} genomes")
        except Exception as e:
            logger.warning(f"  Failed to process {gid}: {e}")

    logger.info(f"Successfully processed {len(annotations)}/{len(genome_ids)} genomes")

    if not annotations:
        raise ValueError("No genomes were successfully processed")

    # Step 3: Build feature matrix
    logger.info("Building feature matrix...")
    ann_list = list(annotations.values())
    feature_df = build_gene_feature_matrix(ann_list)

    if feature_df.shape[1] == 0:
        raise ValueError(
            "No AMR gene features extracted. This likely means no AMR genes were "
            "detected across all genomes. Check that CARD database is properly set up."
        )

    gene_families = [
        c.replace("_present", "")
        for c in feature_df.columns
        if c.endswith("_present")
    ]

    # Save gene families list for prediction time
    model_dir.mkdir(parents=True, exist_ok=True)
    import json

    with open(model_dir / "gene_families.json", "w") as f:
        json.dump(gene_families, f)

    # Save feature column names
    with open(model_dir / "feature_columns.json", "w") as f:
        json.dump(list(feature_df.columns), f)

    # Step 4: Train per-antibiotic models
    all_metrics = {}

    for antibiotic in available_antibiotics:
        ab_meta = meta[meta["antibiotic"] == antibiotic].copy()

        # Match to processed genomes
        ab_meta = ab_meta[ab_meta["genome_id"].astype(str).isin(annotations.keys())]

        # Average MIC if multiple measurements per genome
        mic_by_genome = (
            ab_meta.groupby("genome_id")["log2_mic"]
            .median()
            .reset_index()
        )

        # Align with feature matrix
        valid_ids = [gid for gid in mic_by_genome["genome_id"] if gid in feature_df.index]
        if len(valid_ids) < 10:
            logger.warning(
                f"[{antibiotic}] Only {len(valid_ids)} samples — skipping (need >= 10)"
            )
            continue

        X = feature_df.loc[valid_ids].values
        y = mic_by_genome.set_index("genome_id").loc[valid_ids, "log2_mic"].values

        logger.info(f"[{antibiotic}] Training with {len(valid_ids)} samples...")
        predictor = MICPredictor(antibiotic=antibiotic)
        metrics = predictor.train(
            X, y, feature_names=list(feature_df.columns)
        )
        predictor.save(model_dir)
        all_metrics[antibiotic] = metrics

    logger.info("Training complete!")
    return all_metrics
