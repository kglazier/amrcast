"""CLI command for MIC prediction."""

import json
import logging
from pathlib import Path

import numpy as np
import typer

from amrcast.config.settings import get_settings

predict_app = typer.Typer(name="predict", help="Predict antibiotic resistance from genome.")


@predict_app.command("run")
def predict(
    input_file: Path = typer.Argument(..., help="Input genome FASTA file."),
    antibiotics: str = typer.Option(
        None,
        help="Comma-separated antibiotics. Default: all available models.",
    ),
    model_dir: Path = typer.Option(None, help="Model directory. Overrides --organism."),
    output: Path = typer.Option(None, "-o", help="Output JSON file."),
    fmt: str = typer.Option("table", "--format", "-f", help="Output format: table, json."),
    explain: bool = typer.Option(False, "--explain", help="Include SHAP explanations."),
    organism: str = typer.Option(
        "ecoli",
        "--organism", "-O",
        help="Species: ecoli, salmonella, klebsiella. Sets model dir and AMRFinderPlus organism.",
    ),
) -> None:
    """Predict MIC values for a genome assembly."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    # Map organism shorthand to model dir and AMRFinderPlus organism name
    ORGANISM_MAP = {
        "ecoli": ("data/narms/models", "Escherichia"),
        "escherichia": ("data/narms/models", "Escherichia"),
        "salmonella": ("data/salmonella/models", "Salmonella"),
        "klebsiella": ("data/klebsiella/models", "Klebsiella"),
    }

    org_key = organism.lower().strip()
    if org_key in ORGANISM_MAP:
        default_model_dir, amrfinder_organism = ORGANISM_MAP[org_key]
    else:
        default_model_dir = "data/narms/models"
        amrfinder_organism = organism

    settings = get_settings()
    model_dir = model_dir or Path(default_model_dir)

    # Auto-download models if not present
    if not (model_dir / "feature_columns.json").exists() and org_key in ORGANISM_MAP:
        from amrcast.models.download import ensure_models
        typer.echo(f"Downloading {org_key} models...", err=True)
        model_dir = ensure_models(org_key)

    if not input_file.exists():
        typer.echo(f"Error: Input file not found: {input_file}", err=True)
        raise typer.Exit(1)

    # Load feature columns
    feature_columns_path = model_dir / "feature_columns.json"
    if not feature_columns_path.exists():
        typer.echo(
            f"Error: No trained models found in {model_dir}. "
            f"Run training first or set --model-dir.",
            err=True,
        )
        raise typer.Exit(1)

    with open(feature_columns_path) as f:
        feature_columns = json.load(f)

    # Find available models
    model_files = list(model_dir.glob("xgb_*_meta.json"))
    available_abs = []
    for mf in model_files:
        with open(mf) as f:
            meta = json.load(f)
        available_abs.append(meta["antibiotic"])

    if antibiotics:
        target_abs = [a.strip().lower() for a in antibiotics.split(",")]
        missing = set(target_abs) - set(available_abs)
        if missing:
            typer.echo(f"Warning: No models for: {missing}", err=True)
        target_abs = [a for a in target_abs if a in available_abs]
    else:
        target_abs = sorted(available_abs)

    if not target_abs:
        typer.echo("Error: No models available for requested antibiotics.", err=True)
        raise typer.Exit(1)

    typer.echo(f"Processing {input_file.name}...", err=True)
    typer.echo(f"  Models: {len(target_abs)} antibiotics from {model_dir}", err=True)

    # Step 1: Run AMRFinderPlus
    from amrcast.genome.amrfinder import run_amrfinder

    profile = run_amrfinder(input_file, organism=amrfinder_organism)
    typer.echo(
        f"  AMRFinderPlus: {len(profile.amr_hits)} AMR genes, "
        f"{len(profile.point_mutations)} point mutations, "
        f"{len(profile.drug_classes)} drug classes",
        err=True,
    )

    # Step 2: Build features matching the trained model's expected columns
    from amrcast.data.narms_features import build_features_from_amrfinder

    all_gene_symbols = [h.element_symbol for h in profile.hits]
    hit_methods = {h.element_symbol: h.method for h in profile.hits}
    feature_vec = build_features_from_amrfinder(
        all_gene_symbols, feature_columns, hit_methods=hit_methods
    )
    X = feature_vec.reshape(1, -1)

    # Step 3: Predict
    from amrcast.ml.xgboost_model import MICPredictor
    from amrcast.explain.clinical import classify_mic

    predictions = []
    explanations = []

    for ab in target_abs:
        predictor = MICPredictor(antibiotic=ab)
        predictor.load(model_dir)

        log2_mic = float(predictor.predict(X)[0])
        mic = float(2 ** log2_mic)
        clinical_cat = classify_mic(ab, mic)

        pred = {
            "antibiotic": ab,
            "predicted_mic_ug_ml": round(mic, 4),
            "predicted_log2_mic": round(log2_mic, 2),
            "clinical_category": clinical_cat,
        }

        if explain and predictor.model is not None:
            try:
                from amrcast.explain.shap_explainer import explain_prediction

                explanation = explain_prediction(
                    model=predictor.model,
                    X=X,
                    feature_names=feature_columns,
                    antibiotic=ab,
                    predicted_log2_mic=log2_mic,
                )
                pred["explanation"] = explanation.to_dict()
                explanations.append(explanation)
            except Exception as e:
                pred["explain_error"] = str(e)

        predictions.append(pred)

    result = {
        "sample": input_file.name,
        "amrfinder_summary": {
            "amr_genes": len(profile.amr_hits),
            "point_mutations": len(profile.point_mutations),
            "drug_classes": profile.drug_classes,
            "gene_symbols": profile.gene_symbols,
        },
        "predictions": predictions,
    }

    # Output
    if output:
        with open(output, "w") as f:
            json.dump(result, f, indent=2)
        typer.echo(f"Results written to {output}", err=True)

    if fmt == "json":
        typer.echo(json.dumps(result, indent=2))
    else:
        # Table output
        _print_table(input_file.name, profile, predictions)

    if explain and explanations:
        typer.echo("")
        for exp in explanations:
            typer.echo(exp.detailed_report())


def _print_table(sample_name: str, profile, predictions: list[dict]) -> None:
    """Print predictions as a clean table."""
    typer.echo(f"\n  {sample_name}")
    typer.echo(f"  {len(profile.amr_hits)} AMR genes, {len(profile.point_mutations)} point mutations\n")

    # Header
    typer.echo(f"  {'Antibiotic':<28} {'MIC (ug/mL)':>12} {'Category':>14}")
    typer.echo(f"  {'-' * 28} {'-' * 12} {'-' * 14}")

    for p in predictions:
        mic = p["predicted_mic_ug_ml"]
        mic_str = f"{mic:.3f}" if mic < 1 else f"{mic:.1f}"
        cat = p["clinical_category"]

        # Color-code category
        if cat == "Resistant":
            cat_display = f"** {cat} **"
        elif cat == "Intermediate":
            cat_display = f"   {cat}   "
        else:
            cat_display = f"   {cat}   "

        typer.echo(f"  {p['antibiotic']:<28} {mic_str:>12} {cat_display:>14}")

    typer.echo("")
