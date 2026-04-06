"""Data acquisition from BV-BRC (PATRIC)."""

import logging
from pathlib import Path

import requests

logger = logging.getLogger(__name__)

BVBRC_API = "https://www.bv-brc.org/api"


def query_bvbrc_amr_metadata(
    species_taxon_id: int = 562,
    antibiotics: list[str] | None = None,
    limit: int = 100,
) -> list[dict]:
    """Query BV-BRC for genomes with AMR phenotype data (MIC values).

    Args:
        species_taxon_id: NCBI taxon ID (562 = E. coli).
        antibiotics: Filter to specific antibiotics. If None, get all.
        limit: Maximum records per antibiotic.

    Returns:
        List of dicts with genome_id, antibiotic, mic, etc.
    """
    if antibiotics is None:
        antibiotics = ["ciprofloxacin", "ampicillin"]

    all_records = []
    for antibiotic in antibiotics:
        url = (
            f"{BVBRC_API}/genome_amr/"
            f"?eq(antibiotic,{antibiotic})"
            f"&eq(taxon_id,{species_taxon_id})"
            f"&eq(laboratory_typing_method,MIC)"
            f"&select(genome_id,genome_name,antibiotic,measurement_value,"
            f"measurement_sign,resistant_phenotype)"
            f"&limit({limit})"
        )

        logger.info(f"Querying BV-BRC for {antibiotic} MIC data...")
        resp = requests.get(url, headers={"Accept": "application/json"}, timeout=180)
        resp.raise_for_status()
        records = resp.json()
        logger.info(f"  Got {len(records)} records for {antibiotic}")
        all_records.extend(records)

    return all_records


def download_genome_fasta(genome_id: str, output_dir: Path) -> Path:
    """Download a genome assembly FASTA from BV-BRC."""
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{genome_id}.fasta"

    if output_path.exists():
        return output_path

    url = (
        f"{BVBRC_API}/genome_sequence/"
        f"?eq(genome_id,{genome_id})"
        f"&http_accept=application/sralign+dna+fasta"
    )
    resp = requests.get(url, timeout=120)
    resp.raise_for_status()

    if not resp.text.strip():
        raise ValueError(f"Empty response for genome {genome_id}")

    with open(output_path, "w") as f:
        f.write(resp.text)

    return output_path


def download_sample_dataset(
    output_dir: Path,
    antibiotics: list[str] | None = None,
    n_genomes: int = 50,
) -> Path:
    """Download a sample dataset for training.

    Downloads MIC metadata + genome assemblies from BV-BRC.
    """
    import pandas as pd

    if antibiotics is None:
        antibiotics = ["ciprofloxacin", "ampicillin"]

    raw_dir = output_dir / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    genomes_dir = raw_dir / "genomes"
    genomes_dir.mkdir(exist_ok=True)

    # Step 1: Get AMR metadata
    metadata_path = raw_dir / "amr_metadata.csv"
    if metadata_path.exists():
        logger.info(f"Metadata already exists: {metadata_path}")
        metadata = pd.read_csv(metadata_path)
    else:
        records = query_bvbrc_amr_metadata(
            antibiotics=antibiotics,
            limit=n_genomes * 2,
        )

        if not records:
            raise ValueError("No AMR records found")

        metadata = pd.DataFrame(records)
        metadata.to_csv(metadata_path, index=False)
        logger.info(f"Saved {len(metadata)} AMR records to {metadata_path}")

    # Step 2: Download genome FASTAs
    genome_ids = metadata["genome_id"].unique()[:n_genomes]
    logger.info(f"Downloading {len(genome_ids)} genome assemblies...")

    downloaded = []
    for i, gid in enumerate(genome_ids):
        try:
            download_genome_fasta(str(gid), genomes_dir)
            downloaded.append(str(gid))
            if (i + 1) % 10 == 0:
                logger.info(f"  Downloaded {i + 1}/{len(genome_ids)}")
        except Exception as e:
            logger.warning(f"  Failed to download {gid}: {e}")

    logger.info(f"Successfully downloaded {len(downloaded)}/{len(genome_ids)} genomes")

    # Filter metadata to downloaded genomes
    metadata = metadata[metadata["genome_id"].astype(str).isin(downloaded)]
    metadata.to_csv(metadata_path, index=False)

    return metadata_path
