"""Experiment: Baseline vs ESM-2 (mean-pool) vs ESM-2 (per-class).

Three-way comparison:
  1. Baseline: gene presence/absence + mutation counts (no embeddings)
  2. ESM-2 mean-pool: gene features + single mean-pooled embedding (old approach)
  3. ESM-2 per-class: gene features + per-drug-class embeddings (new approach)

Usage:
    python -m experiments.compare_baseline_esm
    python -m experiments.compare_baseline_esm --antibiotics ampicillin
"""

import argparse
import json
import logging
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from amrcast.data.harmonize import harmonize_mic_data
from amrcast.features.gene_features import build_feature_matrix
from amrcast.genome.models import GenomeAMRProfile
from amrcast.genome.protein_extractor import extract_proteins_from_genome
from amrcast.features.aggregator import build_combined_features_with_sequences
from amrcast.ml.xgboost_model import MICPredictor

logger = logging.getLogger(__name__)


def load_profiles(data_dir: Path) -> dict[str, GenomeAMRProfile]:
    cache_dir = data_dir / "amrfinder_cache"
    profiles = {}
    for path in cache_dir.glob("*.json"):
        try:
            profile = GenomeAMRProfile.model_validate_json(path.read_text())
            profiles[profile.sample_id] = profile
        except Exception:
            pass
    return profiles


def run_experiment(
    data_dir: Path,
    antibiotics: list[str] | None = None,
    esm_model_name: str = "esm2_t33_650M_UR50D",
    n_folds: int = 5,
    esm_components: int = 32,
) -> dict:
    """Run three-way comparison: baseline vs ESM-2 mean-pool vs ESM-2 per-class."""
    metadata_path = data_dir / "raw" / "amr_metadata.csv"
    raw_meta = pd.read_csv(metadata_path)
    meta = harmonize_mic_data(raw_meta)
    if antibiotics:
        meta = meta[meta["antibiotic"].isin([a.lower() for a in antibiotics])]

    profiles = load_profiles(data_dir)
    genomes_dir = data_dir / "raw" / "genomes"
    available_antibiotics = sorted(meta["antibiotic"].unique())
    profile_list = list(profiles.values())

    logger.info(f"Profiles: {len(profiles)}, Antibiotics: {available_antibiotics}")

    # 1. Baseline features (gene + mutation counts, no ESM)
    baseline_features = build_feature_matrix(profile_list)
    logger.info(f"Baseline: {baseline_features.shape}")

    # 2-3. Extract protein sequences (shared between both ESM approaches)
    logger.info("Extracting protein sequences...")
    protein_sequences = {}
    for gid_str, profile in profiles.items():
        fasta_path = genomes_dir / f"{gid_str}.fasta"
        if fasta_path.exists():
            protein_sequences[gid_str] = extract_proteins_from_genome(profile, fasta_path)
    logger.info(f"Extracted proteins for {len(protein_sequences)} genomes")

    esm_cache = data_dir / "esm_cache"

    # 2. ESM-2 mean-pool (legacy)
    logger.info("Building ESM-2 mean-pool features...")
    esm_meanpool_features = build_combined_features_with_sequences(
        profile_list,
        protein_sequences=protein_sequences,
        esm_model_name=esm_model_name,
        esm_cache_dir=esm_cache,
        esm_per_class=False,
    )
    logger.info(f"ESM-2 mean-pool: {esm_meanpool_features.shape}")

    # 3. ESM-2 per-class (new)
    logger.info("Building ESM-2 per-class features...")
    esm_perclass_features = build_combined_features_with_sequences(
        profile_list,
        protein_sequences=protein_sequences,
        esm_model_name=esm_model_name,
        esm_cache_dir=esm_cache,
        esm_per_class=True,
        esm_components=esm_components,
    )
    logger.info(f"ESM-2 per-class: {esm_perclass_features.shape}")

    # Run CV for each method × antibiotic
    methods = {
        "baseline": baseline_features,
        "esm_meanpool": esm_meanpool_features,
        "esm_perclass": esm_perclass_features,
    }

    results = {}
    for antibiotic in available_antibiotics:
        ab_meta = meta[meta["antibiotic"] == antibiotic].copy()
        ab_meta = ab_meta[ab_meta["genome_id"].astype(str).isin(profiles.keys())]
        mic_by_genome = ab_meta.groupby("genome_id")["log2_mic"].median().reset_index()

        ab_results = {"antibiotic": antibiotic}

        for method_name, feature_df in methods.items():
            valid_ids = [
                gid for gid in mic_by_genome["genome_id"]
                if gid in feature_df.index
            ]

            if len(valid_ids) < 20:
                logger.warning(f"[{antibiotic}/{method_name}] Only {len(valid_ids)} samples, skipping")
                ab_results[method_name] = None
                continue

            X = feature_df.loc[valid_ids].values
            y = mic_by_genome.set_index("genome_id").loc[valid_ids, "log2_mic"].values

            logger.info(
                f"\n[{antibiotic}] {method_name}: "
                f"{len(valid_ids)} samples, {X.shape[1]} features"
            )
            predictor = MICPredictor(antibiotic=f"{antibiotic}_{method_name}")
            cv = predictor.cross_validate(
                X, y,
                feature_names=list(feature_df.columns),
                n_folds=n_folds,
            )
            ab_results[method_name] = {
                k: v for k, v in cv.items() if k != "fold_metrics"
            }

        results[antibiotic] = ab_results

    return results


def print_results(results: dict) -> None:
    print("\n" + "=" * 80)
    print("  EXPERIMENT: Baseline vs ESM-2 Mean-Pool vs ESM-2 Per-Class")
    print("=" * 80)

    method_labels = {
        "baseline": "Baseline",
        "esm_meanpool": "ESM Mean-Pool",
        "esm_perclass": "ESM Per-Class",
    }

    for ab, r in results.items():
        print(f"\n  {ab.upper()}")
        print(f"  {'-' * 70}")

        header = f"  {'Metric':<22}"
        for method in ["baseline", "esm_meanpool", "esm_perclass"]:
            header += f" {method_labels[method]:>18}"
        print(header)

        for metric_name, label in [
            ("mae", "MAE (log2)"),
            ("essential_agreement", "Essential Agreement"),
            ("exact_match", "Exact Match"),
        ]:
            line = f"  {label:<22}"
            for method in ["baseline", "esm_meanpool", "esm_perclass"]:
                m = r.get(method)
                if m:
                    line += f" {m[f'{metric_name}_mean']:.2f} ± {m[f'{metric_name}_std']:.2f}"
                    line = f"{line:>18}" if len(line) < 40 else line
                else:
                    line += f" {'N/A':>18}"
            print(line)

        line = f"  {'Samples':<22}"
        for method in ["baseline", "esm_meanpool", "esm_perclass"]:
            m = r.get(method)
            line += f" {m['n_samples']:>18}" if m else f" {'N/A':>18}"
        print(line)

    print("\n" + "=" * 80)
    print("  EA = within ±1 doubling dilution (clinical standard)")
    print("  Lower MAE better. Higher EA/Exact Match better.")
    print("=" * 80)


def main():
    parser = argparse.ArgumentParser(description="Three-way ESM-2 comparison")
    parser.add_argument("--data-dir", type=Path, default=Path("data/real"))
    parser.add_argument("--antibiotics", type=str, default=None)
    parser.add_argument("--esm-model", type=str, default="esm2_t33_650M_UR50D")
    parser.add_argument("--n-folds", type=int, default=5)
    parser.add_argument("--esm-components", type=int, default=32)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    ab_list = [a.strip() for a in args.antibiotics.split(",")] if args.antibiotics else None

    results = run_experiment(
        data_dir=args.data_dir,
        antibiotics=ab_list,
        esm_model_name=args.esm_model,
        n_folds=args.n_folds,
        esm_components=args.esm_components,
    )

    print_results(results)

    output_path = args.output or Path(f"experiments/results_{datetime.now():%Y%m%d_%H%M%S}.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
