"""Train models for any species from NCBI antibiogram + pathogen detection data."""

import json
import logging
from pathlib import Path

import typer

logger = logging.getLogger(__name__)


def train_species(
    data_dir: Path,
    platform_filter: str = "ensitit",
    min_isolates: int = 500,
    n_folds: int = 5,
) -> dict:
    """Train all antibiotic models for a species from NCBI data.

    Expects data_dir to contain:
      - antibiogram_mic.csv (from download_antibiogram_data)
      - amr_metadata.tsv (from NCBI FTP)

    Returns dict of antibiotic -> CV metrics.
    """
    from amrcast.data.narms_features import build_narms_training_data
    from amrcast.ml.xgboost_model import MICPredictor

    mic_path = data_dir / "antibiogram_mic.csv"
    metadata_path = data_dir / "amr_metadata.tsv"

    if not mic_path.exists():
        raise FileNotFoundError(f"MIC data not found: {mic_path}")
    if not metadata_path.exists():
        raise FileNotFoundError(f"Metadata not found: {metadata_path}")

    features, targets = build_narms_training_data(
        mic_path=str(mic_path),
        metadata_path=str(metadata_path),
        platform_filter=platform_filter,
        min_isolates_per_drug=min_isolates,
    )

    model_dir = data_dir / "models"
    model_dir.mkdir(parents=True, exist_ok=True)

    all_antibiotics = sorted(targets["antibiotic"].unique())
    results = {}

    for ab in all_antibiotics:
        ab_targets = targets[targets["antibiotic"] == ab]
        valid_ids = [acc for acc in ab_targets["biosample_acc"] if acc in features.index]
        if len(valid_ids) < 50:
            continue

        X = features.loc[valid_ids].values
        y = ab_targets.set_index("biosample_acc").loc[valid_ids, "log2_mic"].values

        predictor = MICPredictor(antibiotic=ab)
        cv = predictor.cross_validate(
            X, y, feature_names=list(features.columns), n_folds=n_folds
        )
        predictor.save(model_dir)

        results[ab] = {
            "n_samples": cv["n_samples"],
            "mae_mean": cv["mae_mean"],
            "ea_mean": cv["essential_agreement_mean"],
            "exact_mean": cv["exact_match_mean"],
        }

    # Save feature columns
    with open(model_dir / "feature_columns.json", "w") as f:
        json.dump(list(features.columns), f)
    with open(model_dir / "training_results.json", "w") as f:
        json.dump(results, f, indent=2)

    return results
