"""CLI command for model training."""

import logging
from pathlib import Path

import typer

from amrcast.config.settings import get_settings

train_app = typer.Typer(name="train", help="Train MIC prediction models.")


@train_app.command("run")
def train(
    antibiotics: str = typer.Option(
        None,
        help="Comma-separated antibiotics to train on. Default: all available.",
    ),
    data_dir: Path = typer.Option(None, help="Data directory."),
    model_dir: Path = typer.Option(None, help="Model output directory."),
    organism: str = typer.Option("Escherichia", help="Organism for AMRFinderPlus."),
    esm: bool = typer.Option(False, "--esm", help="Include ESM-2 protein embeddings (requires GPU)."),
    esm_model: str = typer.Option("esm2_t33_650M_UR50D", help="ESM-2 model name."),
    cv: bool = typer.Option(True, help="Use k-fold cross-validation (recommended)."),
    n_folds: int = typer.Option(5, help="Number of CV folds."),
) -> None:
    """Train XGBoost MIC prediction models on downloaded data."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    settings = get_settings()
    data_dir = data_dir or settings.data_dir
    model_dir = model_dir or settings.model_dir

    ab_list = [a.strip() for a in antibiotics.split(",")] if antibiotics else None

    typer.echo("Training MIC prediction models...")
    typer.echo(f"  Data: {data_dir}")
    typer.echo(f"  Models: {model_dir}")
    typer.echo(f"  Organism: {organism}")
    typer.echo(f"  ESM-2: {'enabled (' + esm_model + ')' if esm else 'disabled'}")
    typer.echo(f"  Cross-validation: {n_folds}-fold" if cv else "  Cross-validation: off")
    if ab_list:
        typer.echo(f"  Antibiotics: {ab_list}")

    from amrcast.ml.training import run_training_pipeline

    metrics = run_training_pipeline(
        data_dir=data_dir,
        model_dir=model_dir,
        antibiotics=ab_list,
        organism=organism,
        use_esm=esm,
        esm_model_name=esm_model,
        use_cv=cv,
        n_folds=n_folds,
    )

    typer.echo("\n=== Training Results ===")
    for ab, m in metrics.items():
        if "mae_mean" in m:
            # CV results
            typer.echo(
                f"  {ab} ({m['n_samples']} samples, {m['n_folds']}-fold CV):\n"
                f"    MAE  = {m['mae_mean']:.2f} ± {m['mae_std']:.2f}\n"
                f"    EA   = {m['essential_agreement_mean']:.1%} ± {m['essential_agreement_std']:.1%}\n"
                f"    Exact= {m['exact_match_mean']:.1%} ± {m['exact_match_std']:.1%}"
            )
        else:
            # Single-split results
            typer.echo(
                f"  {ab}: MAE={m['mae']:.2f}, "
                f"EA={m['essential_agreement']:.1%}, "
                f"n={m['n_train']+m['n_val']}"
            )

    typer.echo(f"\nModels saved to {model_dir}/")
