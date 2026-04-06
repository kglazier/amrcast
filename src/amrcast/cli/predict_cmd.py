"""CLI command for MIC prediction."""

import json
import logging
from pathlib import Path

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

    # Load model metadata
    gene_symbols_path = model_dir / "gene_symbols.json"
    drug_classes_path = model_dir / "drug_classes.json"
    feature_columns_path = model_dir / "feature_columns.json"

    if not gene_symbols_path.exists():
        typer.echo(
            f"Error: No trained models found in {model_dir}. Run 'amrcast train run' first.",
            err=True,
        )
        raise typer.Exit(1)

    with open(gene_symbols_path) as f:
        gene_symbols = json.load(f)
    with open(drug_classes_path) as f:
        drug_classes = json.load(f)
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
        target_abs = available_abs

    if not target_abs:
        typer.echo("Error: No models available for requested antibiotics.", err=True)
        raise typer.Exit(1)

    typer.echo(f"Processing {input_file.name}...", err=True)

    # Step 1: Run AMRFinderPlus
    from amrcast.genome.amrfinder import run_amrfinder

    profile = run_amrfinder(input_file, organism=organism)
    typer.echo(
        f"  {len(profile.amr_hits)} AMR genes, "
        f"{len(profile.point_mutations)} point mutations, "
        f"{len(profile.drug_classes)} drug classes",
        err=True,
    )

    # Step 2: Extract features
    from amrcast.features.gene_features import build_feature_matrix

    feature_df = build_feature_matrix(
        [profile], gene_symbols=gene_symbols, drug_classes=drug_classes
    )
    X = feature_df.values

    # Step 3: Predict
    from amrcast.ml.xgboost_model import MICPredictor

    predictions = []
    for ab in target_abs:
        predictor = MICPredictor(antibiotic=ab)
        predictor.load(model_dir)

        log2_mic = float(predictor.predict(X)[0])
        mic = float(2 ** log2_mic)

        pred = {
            "antibiotic": ab,
            "predicted_mic_ug_ml": round(mic, 4),
            "predicted_log2_mic": round(log2_mic, 2),
        }

        # Optional SHAP explanations
        if explain and predictor.model is not None:
            try:
                import shap

                explainer = shap.TreeExplainer(predictor.model)
                shap_values = explainer.shap_values(X)
                top_indices = abs(shap_values[0]).argsort()[-5:][::-1]
                pred["top_features"] = [
                    {
                        "feature": feature_columns[i] if i < len(feature_columns) else f"f{i}",
                        "shap_value": round(float(shap_values[0][i]), 3),
                    }
                    for i in top_indices
                ]
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
