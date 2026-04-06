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
    if ab_list:
        typer.echo(f"  Antibiotics: {ab_list}")

    from amrcast.ml.training import run_training_pipeline

    card_dir = settings.card_dir / "hmms"

    metrics = run_training_pipeline(
        data_dir=data_dir,
        card_dir=card_dir,
        model_dir=model_dir,
        antibiotics=ab_list,
    )

    typer.echo("\n=== Training Results ===")
    for ab, m in metrics.items():
        typer.echo(
            f"  {ab}: MAE={m['mae']:.2f}, "
            f"EA={m['essential_agreement']:.1%}, "
            f"n={m['n_train']+m['n_val']}"
        )

    typer.echo(f"\nModels saved to {model_dir}/")
