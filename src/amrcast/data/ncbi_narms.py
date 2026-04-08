"""NARMS/NCBI data acquisition — E. coli with standardized MIC data.

Pulls quantitative MIC data from NCBI BioSample antibiogram tables and
links to pre-computed AMRFinderPlus genotypes from the Pathogen Detection
metadata TSV.

This gives us clean, standardized data (Sensititre broth microdilution)
without needing to download genome FASTAs.
"""

import csv
import logging
import re
import time
import xml.etree.ElementTree as ET
from io import StringIO
from pathlib import Path

import pandas as pd
import requests

logger = logging.getLogger(__name__)

EUTILS_BASE = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"


def fetch_antibiogram_ids(organism: str = "Escherichia coli", batch_size: int = 10000) -> list[str]:
    """Get all BioSample IDs for a species with antibiogram data."""
    org_encoded = organism.replace(" ", "+")
    url = (
        f"{EUTILS_BASE}/esearch.fcgi"
        f"?db=biosample"
        f"&term={org_encoded}[Organism]+AND+antibiogram[filter]"
        f"&retmax={batch_size}"
        f"&retmode=json"
    )
    resp = requests.get(url, timeout=60)
    resp.raise_for_status()
    data = resp.json()

    total = int(data["esearchresult"]["count"])
    ids = data["esearchresult"]["idlist"]
    logger.info(f"Found {total} {organism} BioSamples with antibiogram")

    # Fetch remaining if more than batch_size
    while len(ids) < total:
        url_next = (
            f"{EUTILS_BASE}/esearch.fcgi"
            f"?db=biosample"
            f"&term=Escherichia+coli[Organism]+AND+antibiogram[filter]"
            f"&retmax={batch_size}"
            f"&retstart={len(ids)}"
            f"&retmode=json"
        )
        resp = requests.get(url_next, timeout=60)
        resp.raise_for_status()
        new_ids = resp.json()["esearchresult"]["idlist"]
        if not new_ids:
            break
        ids.extend(new_ids)
        logger.info(f"  Fetched {len(ids)}/{total} IDs")
        time.sleep(0.4)

    return ids


def fetch_antibiogram_batch(biosample_ids: list[str]) -> list[dict]:
    """Fetch antibiogram data for a batch of BioSample IDs.

    Returns list of dicts with: biosample_acc, antibiotic, mic_value,
    measurement_sign, resistance_phenotype, method, platform.
    """
    id_str = ",".join(biosample_ids)
    url = (
        f"{EUTILS_BASE}/efetch.fcgi"
        f"?db=biosample"
        f"&id={id_str}"
        f"&retmode=xml"
    )
    resp = requests.get(url, timeout=120)
    resp.raise_for_status()

    records = []
    root = ET.fromstring(resp.text)

    for biosample in root.findall(".//BioSample"):
        acc = biosample.get("accession", "")

        # Find antibiogram table
        for table in biosample.findall(".//Table[@class='Antibiogram.1.0']"):
            # Get header
            header = []
            for cell in table.findall("Header/Cell"):
                header.append(cell.text or "")

            # Parse rows
            for row in table.findall("Body/Row"):
                cells = [c.text or "" for c in row.findall("Cell")]
                row_dict = dict(zip(header, cells))

                # Only keep MIC measurements
                if row_dict.get("Laboratory typing method", "").upper() != "MIC":
                    continue

                mic_str = row_dict.get("Measurement", "")
                # Skip combination drug values like "8/4"
                if "/" in mic_str and row_dict.get("Antibiotic", "") not in (
                    "trimethoprim-sulfamethoxazole",
                ):
                    continue

                records.append({
                    "biosample_acc": acc,
                    "antibiotic": row_dict.get("Antibiotic", "").lower().strip(),
                    "mic_value": mic_str,
                    "measurement_sign": _clean_sign(row_dict.get("Measurement sign", "")),
                    "resistance_phenotype": row_dict.get("Resistance phenotype", "").lower(),
                    "method": row_dict.get("Laboratory typing method", ""),
                    "platform": row_dict.get("Laboratory typing platform", ""),
                    "testing_standard": row_dict.get("Testing standard", ""),
                })

    return records


def _clean_sign(sign: str) -> str:
    """Normalize measurement sign from XML."""
    sign = sign.strip()
    # XML entities get decoded, but just in case
    sign = sign.replace("&gt;", ">").replace("&lt;", "<")
    return sign


def download_antibiogram_data(
    output_path: Path,
    organism: str = "Escherichia coli",
    batch_size: int = 200,
) -> pd.DataFrame:
    """Download antibiogram MIC data from NCBI BioSample for a species.

    Args:
        output_path: Where to save the CSV.
        organism: Species name (e.g., "Escherichia coli", "Salmonella enterica", "Klebsiella pneumoniae").
        batch_size: BioSample IDs per E-utilities request (max ~200 for XML).

    Returns:
        DataFrame with all MIC records.
    """
    if output_path.exists():
        logger.info(f"Antibiogram data already exists: {output_path}")
        return pd.read_csv(output_path)

    logger.info(f"Fetching {organism} BioSample IDs with antibiogram data...")
    all_ids = fetch_antibiogram_ids(organism=organism)
    logger.info(f"Total BioSample IDs: {len(all_ids)}")

    all_records = []
    for i in range(0, len(all_ids), batch_size):
        batch = all_ids[i:i + batch_size]
        try:
            records = fetch_antibiogram_batch(batch)
            all_records.extend(records)
        except Exception as e:
            logger.warning(f"  Batch {i}-{i+batch_size} failed: {e}")

        if (i // batch_size + 1) % 10 == 0:
            logger.info(
                f"  Progress: {i + batch_size}/{len(all_ids)} BioSamples, "
                f"{len(all_records)} MIC records"
            )
        # Rate limit: NCBI asks for ≤3 requests/second without API key
        time.sleep(0.35)

    df = pd.DataFrame(all_records)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    logger.info(f"Saved {len(df)} MIC records to {output_path}")

    return df


def load_pathogen_metadata(
    metadata_path: Path,
    narms_only: bool = False,
) -> pd.DataFrame:
    """Load and filter the NCBI Pathogen Detection metadata TSV.

    Args:
        metadata_path: Path to the downloaded PDG*.metadata.tsv.
        narms_only: If True, filter to NARMS BioProjects only.

    Returns:
        DataFrame with columns: biosample_acc, asm_acc, amr_genotypes, etc.
    """
    logger.info(f"Loading pathogen metadata from {metadata_path}...")
    df = pd.read_csv(metadata_path, sep="\t", low_memory=False)

    # First row is the actual header (column 0 is "#label")
    if "#label" in df.columns:
        df = df.rename(columns={"#label": "label"})

    if narms_only:
        narms_projects = {"PRJNA292663", "PRJNA292667"}
        df = df[df["bioproject_acc"].isin(narms_projects)]
        logger.info(f"Filtered to NARMS: {len(df)} isolates")

    # Keep useful columns
    keep_cols = [
        "biosample_acc", "asm_acc", "bioproject_acc",
        "AMR_genotypes", "AMR_genotypes_core",
        "number_amr_genes", "number_drugs_resistant",
        "scientific_name", "collection_date", "geo_loc_name",
        "host", "isolation_source",
    ]
    available = [c for c in keep_cols if c in df.columns]
    df = df[available].copy()

    return df


def build_narms_dataset(
    data_dir: Path,
    narms_only: bool = False,
) -> pd.DataFrame:
    """Build the complete NARMS dataset: MIC + AMRFinderPlus genotypes.

    Joins antibiogram MIC data to pathogen detection metadata by biosample_acc.

    Returns:
        DataFrame with one row per (isolate, antibiotic) pair, containing
        both MIC value and AMRFinderPlus genotype calls.
    """
    mic_path = data_dir / "antibiogram_mic.csv"
    metadata_path = data_dir / "ecoli_amr_metadata.tsv"

    if not metadata_path.exists():
        raise FileNotFoundError(
            f"Metadata not found: {metadata_path}\n"
            f"Download from: https://ftp.ncbi.nlm.nih.gov/pathogen/Results/"
            f"Escherichia_coli_Shigella/latest_snps/AMR/"
        )

    # Load MIC data
    if not mic_path.exists():
        raise FileNotFoundError(
            f"Antibiogram data not found: {mic_path}\n"
            f"Run download_antibiogram_data() first."
        )
    mic_df = pd.read_csv(mic_path)

    # Load pathogen metadata
    meta_df = load_pathogen_metadata(metadata_path, narms_only=narms_only)

    # Join on biosample_acc
    joined = mic_df.merge(meta_df, on="biosample_acc", how="inner")
    logger.info(
        f"Joined dataset: {len(joined)} records "
        f"({joined['biosample_acc'].nunique()} unique isolates)"
    )

    return joined
