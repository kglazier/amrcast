"""Data acquisition from BV-BRC (PATRIC)."""

import logging
import time
from pathlib import Path

import requests

logger = logging.getLogger(__name__)

BVBRC_API = "https://www.bv-brc.org/api"


def query_bvbrc_amr_metadata(
    species_taxon_id: int = 562,
    antibiotics: list[str] | None = None,
    limit: int = 2500,
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

    # Step 2: Download genome FASTAs (skip already downloaded)
    genome_ids = metadata["genome_id"].unique()[:n_genomes]
    logger.info(f"Downloading up to {len(genome_ids)} genome assemblies...")

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
    return metadata_path


def expand_dataset(
    output_dir: Path,
    antibiotics: list[str] | None = None,
    target_genomes: int = 400,
    refresh_metadata: bool = True,
) -> dict:
    """Expand the dataset by fetching more metadata and downloading missing genomes.

    Unlike download_sample_dataset, this:
    - Never overwrites metadata — merges new records with existing
    - Downloads ALL genomes referenced in metadata (up to target_genomes)
    - Reports progress stats

    Returns:
        Dict with stats about what was downloaded.
    """
    import pandas as pd

    if antibiotics is None:
        antibiotics = ["ciprofloxacin", "ampicillin"]

    raw_dir = output_dir / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    genomes_dir = raw_dir / "genomes"
    genomes_dir.mkdir(exist_ok=True)
    metadata_path = raw_dir / "amr_metadata.csv"

    # Step 1: Refresh metadata from BV-BRC (merge with existing)
    existing_meta = pd.read_csv(metadata_path) if metadata_path.exists() else pd.DataFrame()
    logger.info(f"Existing metadata: {len(existing_meta)} rows")

    if refresh_metadata:
        logger.info("Fetching fresh metadata from BV-BRC...")
        records = query_bvbrc_amr_metadata(
            antibiotics=antibiotics,
            limit=2500,
        )
        new_meta = pd.DataFrame(records)

        if not existing_meta.empty:
            # Merge: keep all rows, deduplicate
            combined = pd.concat([existing_meta, new_meta], ignore_index=True)
            combined = combined.drop_duplicates(
                subset=["genome_id", "antibiotic", "measurement_value"],
                keep="first",
            )
        else:
            combined = new_meta

        combined.to_csv(metadata_path, index=False)
        logger.info(f"Metadata updated: {len(existing_meta)} -> {len(combined)} rows")
        metadata = combined
    else:
        metadata = existing_meta

    # Step 2: Figure out which genomes we need
    all_genome_ids = [str(gid) for gid in metadata["genome_id"].unique()]
    already_downloaded = {
        f.stem for f in genomes_dir.glob("*.fasta")
    }
    need_download = [gid for gid in all_genome_ids if gid not in already_downloaded]

    # Cap at target
    need_download = need_download[:max(0, target_genomes - len(already_downloaded))]

    logger.info(
        f"Genomes: {len(already_downloaded)} already downloaded, "
        f"{len(need_download)} to download (target: {target_genomes})"
    )

    # Step 3: Download missing genomes
    newly_downloaded = []
    failed = []
    for i, gid in enumerate(need_download):
        try:
            download_genome_fasta(gid, genomes_dir)
            newly_downloaded.append(gid)
        except Exception as e:
            failed.append(gid)
            logger.warning(f"  Failed {gid}: {e}")

        if (i + 1) % 25 == 0:
            logger.info(
                f"  Progress: {i + 1}/{len(need_download)} "
                f"({len(newly_downloaded)} ok, {len(failed)} failed)"
            )
        # Be polite to BV-BRC API
        if (i + 1) % 50 == 0:
            time.sleep(2)

    total_downloaded = len(already_downloaded) + len(newly_downloaded)
    logger.info(
        f"Done: {total_downloaded} total genomes "
        f"({len(newly_downloaded)} new, {len(failed)} failed)"
    )

    # Report per-antibiotic coverage
    downloaded_set = already_downloaded | set(newly_downloaded)
    stats = {
        "total_metadata_rows": len(metadata),
        "unique_genomes_in_metadata": len(all_genome_ids),
        "total_downloaded": total_downloaded,
        "newly_downloaded": len(newly_downloaded),
        "failed": len(failed),
        "per_antibiotic": {},
    }
    for ab in antibiotics:
        ab_genomes = set(metadata[metadata["antibiotic"] == ab]["genome_id"].astype(str))
        overlap = ab_genomes & downloaded_set
        stats["per_antibiotic"][ab] = {
            "metadata_genomes": len(ab_genomes),
            "downloaded": len(overlap),
        }

    return stats
