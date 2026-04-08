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


@data_app.command("expand")
def expand(
    target_genomes: int = typer.Option(400, help="Target number of downloaded genomes."),
    antibiotics: str = typer.Option(
        "ciprofloxacin,ampicillin",
        help="Comma-separated antibiotics.",
    ),
    data_dir: Path = typer.Option(None, help="Data directory. Defaults to config."),
    no_refresh: bool = typer.Option(False, help="Skip re-fetching metadata from BV-BRC."),
) -> None:
    """Expand dataset: fetch more metadata from BV-BRC and download missing genomes."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    settings = get_settings()
    data_dir = data_dir or settings.data_dir
    ab_list = [a.strip() for a in antibiotics.split(",")]

    typer.echo(f"Expanding dataset in {data_dir}/")
    typer.echo(f"  Target genomes: {target_genomes}")
    typer.echo(f"  Refresh metadata: {not no_refresh}")

    from amrcast.data.download import expand_dataset

    stats = expand_dataset(
        output_dir=data_dir,
        antibiotics=ab_list,
        target_genomes=target_genomes,
        refresh_metadata=not no_refresh,
    )

    typer.echo(f"\n=== Dataset Stats ===")
    typer.echo(f"  Metadata rows: {stats['total_metadata_rows']}")
    typer.echo(f"  Unique genomes in metadata: {stats['unique_genomes_in_metadata']}")
    typer.echo(f"  Downloaded genomes: {stats['total_downloaded']} ({stats['newly_downloaded']} new)")
    if stats['failed']:
        typer.echo(f"  Failed: {stats['failed']}")
    for ab, ab_stats in stats['per_antibiotic'].items():
        typer.echo(f"  {ab}: {ab_stats['downloaded']}/{ab_stats['metadata_genomes']} genomes")


@data_app.command("status")
def status(
    data_dir: Path = typer.Option(None, help="Data directory. Defaults to config."),
) -> None:
    """Show current dataset stats."""
    import pandas as pd

    settings = get_settings()
    data_dir = data_dir or settings.data_dir
    raw_dir = data_dir / "raw"
    metadata_path = raw_dir / "amr_metadata.csv"
    genomes_dir = raw_dir / "genomes"

    if not metadata_path.exists():
        typer.echo("No metadata found. Run 'amrcast data download' first.")
        raise typer.Exit(1)

    meta = pd.read_csv(metadata_path)
    downloaded = {f.stem for f in genomes_dir.glob("*.fasta")} if genomes_dir.exists() else set()

    typer.echo(f"=== Dataset Status ===")
    typer.echo(f"  Metadata: {len(meta)} rows, {meta['genome_id'].nunique()} unique genomes")
    typer.echo(f"  Downloaded: {len(downloaded)} genomes")

    for ab in sorted(meta["antibiotic"].unique()):
        ab_genomes = set(meta[meta["antibiotic"] == ab]["genome_id"].astype(str))
        overlap = ab_genomes & downloaded
        typer.echo(f"  {ab}: {len(overlap)}/{len(ab_genomes)} genomes available")
