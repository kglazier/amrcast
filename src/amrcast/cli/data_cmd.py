"""CLI commands for data acquisition."""

import logging
from pathlib import Path

import typer

from amrcast.config.settings import get_settings

data_app = typer.Typer(name="data", help="Download and manage training data.")

logger = logging.getLogger(__name__)


@data_app.command("download")
def download(
    n_genomes: int = typer.Option(50, help="Number of genomes to download."),
    antibiotics: str = typer.Option(
        "ciprofloxacin,ampicillin",
        help="Comma-separated antibiotics to get MIC data for.",
    ),
    data_dir: Path = typer.Option(None, help="Data directory. Defaults to config."),
) -> None:
    """Download E. coli genomes + MIC data from BV-BRC."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    settings = get_settings()
    data_dir = data_dir or settings.data_dir

    ab_list = [a.strip() for a in antibiotics.split(",")]

    typer.echo(f"Downloading data to {data_dir}/")
    typer.echo(f"  Antibiotics: {ab_list}")
    typer.echo(f"  Target genomes: {n_genomes}")

    from amrcast.data.download import download_sample_dataset

    metadata_path = download_sample_dataset(
        output_dir=data_dir,
        antibiotics=ab_list,
        n_genomes=n_genomes,
    )
    typer.echo(f"  Metadata: {metadata_path}")
    typer.echo("\nDone! Run 'amrcast train run' to train models.")
