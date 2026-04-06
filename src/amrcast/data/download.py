"""Data acquisition from BV-BRC (PATRIC) and CARD."""

import gzip
import io
import logging
import shutil
from pathlib import Path

import requests

logger = logging.getLogger(__name__)

# BV-BRC SOLR API endpoint
BVBRC_API = "https://www.bv-brc.org/api"

# CARD database download URL
CARD_DATA_URL = "https://card.mcmaster.ca/latest/data"


def download_card_hmms(output_dir: Path) -> Path:
    """Download CARD database and extract HMM profiles.

    The CARD 'data' download is a tar.bz2 containing multiple files including
    protein_fasta_protein_homolog_model.fasta and AMR detection models.

    For the vertical slice, we'll download the CARD protein homolog models
    and build HMMs from them, OR download pre-built HMMs if available.

    Returns:
        Path to directory containing .hmm files.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    hmm_dir = output_dir / "hmms"
    hmm_dir.mkdir(exist_ok=True)

    # Check if already downloaded
    existing_hmms = list(hmm_dir.glob("*.hmm"))
    if existing_hmms:
        logger.info(f"CARD HMMs already present: {len(existing_hmms)} files")
        return hmm_dir

    # Download CARD data archive
    logger.info("Downloading CARD database...")
    archive_path = output_dir / "card-data.tar.bz2"

    resp = requests.get(CARD_DATA_URL, stream=True, timeout=300)
    resp.raise_for_status()
    with open(archive_path, "wb") as f:
        for chunk in resp.iter_content(chunk_size=8192):
            f.write(chunk)
    logger.info(f"Downloaded CARD data to {archive_path}")

    # Extract
    logger.info("Extracting CARD archive...")
    shutil.unpack_archive(str(archive_path), str(output_dir))

    # Look for pre-built HMM files or protein homolog FASTA
    # CARD distributes protein_fasta_protein_homolog_model.fasta
    # We need to convert these to HMMs or use a different approach
    homolog_fasta = output_dir / "protein_fasta_protein_homolog_model.fasta"
    if not homolog_fasta.exists():
        # Try finding it in subdirectories
        for p in output_dir.rglob("protein_fasta_protein_homolog_model.fasta"):
            homolog_fasta = p
            break

    if homolog_fasta.exists():
        logger.info(f"Found CARD protein homolog models: {homolog_fasta}")
        # For the vertical slice, we'll use the FASTA directly with phmmer
        # (sequence-vs-sequence search) instead of HMMs
        # Copy to hmm_dir as our reference
        shutil.copy2(str(homolog_fasta), str(hmm_dir / "card_proteins.fasta"))
    else:
        logger.warning("Could not find CARD protein homolog FASTA")

    return hmm_dir


def query_bvbrc_amr_metadata(
    species: str = "Escherichia coli",
    antibiotics: list[str] | None = None,
    limit: int = 100,
) -> list[dict]:
    """Query BV-BRC for genomes with AMR phenotype data (MIC values).

    Args:
        species: Species name to query.
        antibiotics: Filter to specific antibiotics. If None, get all.
        limit: Maximum number of records to return.

    Returns:
        List of dicts with genome_id, antibiotic, mic, measurement_sign, etc.
    """
    if antibiotics is None:
        antibiotics = ["ciprofloxacin", "ampicillin"]

    # Query the AMR phenotype table
    all_records = []
    for antibiotic in antibiotics:
        query = (
            f"eq(organism,{species})"
            f"&eq(antibiotic,{antibiotic})"
            f"&eq(measurement,MIC)"
            f"&ne(measurement_value,null)"
            f"&select(genome_id,antibiotic,measurement_value,measurement_sign,"
            f"measurement_unit,resistant_phenotype,laboratory_typing_method)"
            f"&limit({limit})"
            f"&sort(+genome_id)"
        )

        url = f"{BVBRC_API}/genome_amr/?{query}"
        headers = {"Accept": "application/json"}

        logger.info(f"Querying BV-BRC for {antibiotic} MIC data...")
        resp = requests.get(url, headers=headers, timeout=120)
        resp.raise_for_status()
        records = resp.json()
        logger.info(f"  Got {len(records)} records for {antibiotic}")
        all_records.extend(records)

    return all_records


def download_genome_fasta(genome_id: str, output_dir: Path) -> Path:
    """Download a genome assembly FASTA from BV-BRC.

    Args:
        genome_id: BV-BRC genome ID (e.g., "511145.12").
        output_dir: Directory to save the FASTA file.

    Returns:
        Path to downloaded FASTA file.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{genome_id}.fasta"

    if output_path.exists():
        return output_path

    url = f"{BVBRC_API}/genome_sequence/?eq(genome_id,{genome_id})&http_accept=application/sralign+dna+fasta"
    resp = requests.get(url, timeout=120)
    resp.raise_for_status()

    if not resp.text.strip():
        raise ValueError(f"Empty response for genome {genome_id}")

    with open(output_path, "w") as f:
        f.write(resp.text)

    return output_path


def download_sample_dataset(
    output_dir: Path,
    species: str = "Escherichia coli",
    antibiotics: list[str] | None = None,
    n_genomes: int = 50,
) -> Path:
    """Download a small sample dataset for the vertical slice.

    Downloads MIC metadata + genome assemblies for a small set of E. coli.

    Args:
        output_dir: Root data directory.
        species: Species to download.
        antibiotics: Antibiotics to get MIC data for.
        n_genomes: Number of genomes to download.

    Returns:
        Path to the metadata CSV file.
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
            species=species,
            antibiotics=antibiotics,
            limit=n_genomes * 2,  # request extra since some may fail download
        )

        if not records:
            raise ValueError(f"No AMR records found for {species}")

        metadata = pd.DataFrame(records)
        metadata.to_csv(metadata_path, index=False)
        logger.info(f"Saved {len(metadata)} AMR records to {metadata_path}")

    # Step 2: Download genome FASTAs for unique genome IDs
    genome_ids = metadata["genome_id"].unique()[:n_genomes]
    logger.info(f"Downloading {len(genome_ids)} genome assemblies...")

    downloaded = []
    for i, gid in enumerate(genome_ids):
        try:
            fasta_path = download_genome_fasta(str(gid), genomes_dir)
            downloaded.append(str(gid))
            if (i + 1) % 10 == 0:
                logger.info(f"  Downloaded {i + 1}/{len(genome_ids)}")
        except Exception as e:
            logger.warning(f"  Failed to download {gid}: {e}")

    logger.info(f"Successfully downloaded {len(downloaded)}/{len(genome_ids)} genomes")

    # Filter metadata to only downloaded genomes
    metadata = metadata[metadata["genome_id"].astype(str).isin(downloaded)]
    metadata.to_csv(metadata_path, index=False)

    return metadata_path
