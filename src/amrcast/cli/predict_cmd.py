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
    model_dir: Path = typer.Option(None, help="Model directory."),
    output: Path = typer.Option(None, "-o", help="Output JSON file. Default: stdout."),
    explain: bool = typer.Option(False, "--explain", help="Include SHAP explanations."),
    organism: str = typer.Option("Escherichia", help="Organism for AMRFinderPlus."),
) -> None:
    """Predict MIC values for a genome assembly."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    settings = get_settings()
    model_dir = model_dir or settings.model_dir

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

    profile = run_amrfinder(input_file, organism=organism)
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

    output_json = json.dumps(result, indent=2)

    if output:
        with open(output, "w") as f:
            f.write(output_json)
        typer.echo(f"\nResults written to {output}", err=True)
    else:
        typer.echo(output_json)

    # Print human-readable reports to stderr when explaining
    if explain and explanations:
        typer.echo("\n" + "=" * 60, err=True)
        typer.echo("  PREDICTION REPORT", err=True)
        typer.echo("=" * 60, err=True)
        for exp in explanations:
            typer.echo(exp.detailed_report(), err=True)
