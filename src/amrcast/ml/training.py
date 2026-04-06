"""Training pipeline — orchestrates AMRFinderPlus processing, feature extraction, and model training."""

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from amrcast.data.harmonize import harmonize_mic_data
from amrcast.features.gene_features import build_feature_matrix
from amrcast.genome.amrfinder import run_amrfinder, run_amrfinder_batch, parse_amrfinder_file
from amrcast.genome.models import GenomeAMRProfile
from amrcast.ml.xgboost_model import MICPredictor

logger = logging.getLogger(__name__)


def run_training_pipeline(
    data_dir: Path,
    model_dir: Path,
    antibiotics: list[str] | None = None,
    organism: str = "Escherichia",
    use_cached_amrfinder: bool = True,
    use_esm: bool = False,
    esm_model_name: str = "esm2_t33_650M_UR50D",
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

    # Load cached profiles and identify which genomes need processing
    profiles: dict[str, GenomeAMRProfile] = {}
    to_process: list[Path] = []

    for gid in genome_ids:
        gid_str = str(gid)
        fasta_path = genomes_dir / f"{gid_str}.fasta"
        cache_path = amrfinder_dir / f"{gid_str}.json"

        if not fasta_path.exists():
            continue

        if use_cached_amrfinder and cache_path.exists():
            try:
                profiles[gid_str] = _load_cached_profile(cache_path)
            except Exception:
                to_process.append(fasta_path)
        else:
            to_process.append(fasta_path)

    logger.info(
        f"Genomes: {len(profiles)} cached, {len(to_process)} to process with AMRFinderPlus"
    )

    # Batch process uncached genomes in a single WSL session
    if to_process:
        tsv_dir = amrfinder_dir / "tsv"
        batch_profiles = run_amrfinder_batch(
            to_process, output_dir=tsv_dir, organism=organism
        )
        for sample_id, profile in batch_profiles.items():
            profiles[sample_id] = profile
            # Cache for next time
            cache_path = amrfinder_dir / f"{sample_id}.json"
            _cache_profile(profile, cache_path)

    logger.info(f"Successfully processed {len(profiles)} genomes total")

    if not profiles:
        raise ValueError("No genomes were successfully processed")

    # Step 3: Build feature matrix
    logger.info("Building feature matrix...")
    profile_list = list(profiles.values())

    if use_esm:
        # Extract protein sequences from genomes for ESM-2 embedding
        from amrcast.genome.protein_extractor import extract_proteins_from_genome
        from amrcast.features.aggregator import build_combined_features_with_sequences

        logger.info("Extracting protein sequences for ESM-2 embeddings...")
        protein_sequences = {}
        for gid_str, profile in profiles.items():
            fasta_path = genomes_dir / f"{gid_str}.fasta"
            if fasta_path.exists():
                protein_sequences[gid_str] = extract_proteins_from_genome(profile, fasta_path)

        esm_cache = data_dir / "esm_cache"
        feature_df = build_combined_features_with_sequences(
            profile_list,
            protein_sequences=protein_sequences,
            esm_model_name=esm_model_name,
            esm_cache_dir=esm_cache,
        )
    else:
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
