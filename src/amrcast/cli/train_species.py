"""Train models for any species from NCBI antibiogram + pathogen detection data."""

import json
import logging
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)


def train_species(
    data_dir: Path,
    platform_filter: str = "ensitit",
    min_isolates: int = 500,
    n_folds: int = 5,
    use_groups: bool = True,
) -> dict:
    """Train all antibiotic models for a species from NCBI data.

    Args:
        data_dir: Directory containing antibiogram_mic.csv and amr_metadata.tsv.
        platform_filter: Filter MIC data by testing platform (e.g., "ensitit").
        min_isolates: Skip antibiotics with fewer isolates.
        n_folds: Number of CV folds.
        use_groups: If True, use genotype-grouped CV to prevent clonal leakage.

    Returns dict of antibiotic -> CV metrics.
    """
    from amrcast.data.narms_features import build_narms_training_data
    from amrcast.ml.xgboost_model import MICPredictor

    mic_path = data_dir / "antibiogram_mic.csv"

    # Support both naming conventions
    metadata_path = data_dir / "amr_metadata.tsv"
    if not metadata_path.exists():
        # Try species-specific naming (e.g., ecoli_amr_metadata.tsv)
        tsv_files = list(data_dir.glob("*amr_metadata.tsv"))
        if tsv_files:
            metadata_path = tsv_files[0]

    if not mic_path.exists():
        raise FileNotFoundError(f"MIC data not found: {mic_path}")
    if not metadata_path.exists():
        raise FileNotFoundError(f"Metadata not found in: {data_dir}")

    features, targets, isolate_groups = build_narms_training_data(
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

        groups = None
        if use_groups:
            groups = np.array([isolate_groups[acc] for acc in valid_ids])

        predictor = MICPredictor(antibiotic=ab)
        cv = predictor.cross_validate(
            X, y, feature_names=list(features.columns),
            n_folds=n_folds, groups=groups,
        )
        predictor.save(model_dir)

        results[ab] = {
            "n_samples": cv["n_samples"],
            "mae_mean": cv["mae_mean"],
            "ea_mean": cv["essential_agreement_mean"],
            "exact_mean": cv["exact_match_mean"],
            "grouped": cv.get("grouped", False),
        }

    # Save feature columns
    with open(model_dir / "feature_columns.json", "w") as f:
        json.dump(list(features.columns), f)
    with open(model_dir / "training_results.json", "w") as f:
        json.dump(results, f, indent=2)

    return results
