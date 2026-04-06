"""AMRFinderPlus integration — run and parse output.

Requires AMRFinderPlus installed in WSL (Linux) or natively.
Install: conda install -c bioconda -c conda-forge ncbi-amrfinderplus
Database: amrfinder -u
"""

import csv
import logging
import platform
import subprocess
from io import StringIO
from pathlib import Path

from amrcast.genome.models import AMRFinderHit, GenomeAMRProfile

logger = logging.getLogger(__name__)


def _build_amrfinder_command(
    fasta_path: Path,
    organism: str = "Escherichia",
    threads: int = 4,
    amrfinder_path: str = "amrfinder",
) -> list[str]:
    """Build the amrfinder command, wrapping in WSL if on Windows."""
    # Convert Windows path to WSL path if needed
    fasta_str = str(fasta_path)
    is_windows = platform.system() == "Windows"

    if is_windows:
        # Convert C:\Users\... to /mnt/c/Users/...
        wsl_path = fasta_str.replace("\\", "/")
        if len(wsl_path) >= 2 and wsl_path[1] == ":":
            drive = wsl_path[0].lower()
            wsl_path = f"/mnt/{drive}{wsl_path[2:]}"

        return [
            "wsl", "--exec", "bash", "--noprofile", "--norc", "-c",
            f"export PATH=$HOME/miniconda3/bin:$PATH; "
            f"{amrfinder_path} "
            f"--nucleotide {wsl_path} "
            f"--organism {organism} "
            f"--plus "
            f"--threads {threads}"
        ]
    else:
        return [
            amrfinder_path,
            "--nucleotide", fasta_str,
            "--organism", organism,
            "--plus",
            "--threads", str(threads),
        ]


def run_amrfinder(
    fasta_path: Path,
    organism: str = "Escherichia",
    threads: int = 4,
    amrfinder_path: str = "amrfinder",
) -> GenomeAMRProfile:
    """Run AMRFinderPlus on a genome FASTA and return parsed results.

    Args:
        fasta_path: Path to input genome FASTA.
        organism: Organism name for organism-specific detection (enables point mutations).
        threads: Number of threads.
        amrfinder_path: Path to amrfinder binary.

    Returns:
        GenomeAMRProfile with all detected hits.
    """
    if not fasta_path.exists():
        raise FileNotFoundError(f"FASTA not found: {fasta_path}")

    cmd = _build_amrfinder_command(fasta_path, organism, threads, amrfinder_path)
    logger.info(f"Running AMRFinderPlus on {fasta_path.name}")

    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        timeout=300,
    )

    if result.returncode != 0:
        stderr = result.stderr.strip()
        raise RuntimeError(f"AMRFinderPlus failed (exit {result.returncode}): {stderr}")

    return parse_amrfinder_output(result.stdout, sample_id=fasta_path.stem)


def parse_amrfinder_output(tsv_text: str, sample_id: str = "unknown") -> GenomeAMRProfile:
    """Parse AMRFinderPlus TSV output into structured data.

    Can also be used to parse a saved TSV file.
    """
    hits = []
    reader = csv.DictReader(StringIO(tsv_text), delimiter="\t")

    for row in reader:
        try:
            hit = AMRFinderHit(
                contig_id=row.get("Contig id", ""),
                start=int(row.get("Start", 0)),
                stop=int(row.get("Stop", 0)),
                strand=row.get("Strand", ""),
                element_symbol=row.get("Element symbol", ""),
                element_name=row.get("Element name", ""),
                scope=row.get("Scope", ""),
                type=row.get("Type", ""),
                subtype=row.get("Subtype", ""),
                drug_class=row.get("Class", "NA"),
                drug_subclass=row.get("Subclass", "NA"),
                method=row.get("Method", ""),
                target_length=int(row.get("Target length", 0)),
                ref_length=int(row.get("Reference sequence length", 0)),
                coverage=float(row.get("% Coverage of reference", 0)),
                identity=float(row.get("% Identity to reference", 0)),
                closest_ref=row.get("Closest reference accession", ""),
                closest_ref_name=row.get("Closest reference name", ""),
            )
            hits.append(hit)
        except (ValueError, KeyError) as e:
            logger.warning(f"Skipping malformed AMRFinderPlus row: {e}")

    logger.info(
        f"Parsed {len(hits)} total hits "
        f"({sum(1 for h in hits if h.type == 'AMR')} AMR, "
        f"{sum(1 for h in hits if h.method in ('POINTX', 'POINTN'))} point mutations)"
    )
    return GenomeAMRProfile(sample_id=sample_id, hits=hits)


def parse_amrfinder_file(tsv_path: Path, sample_id: str | None = None) -> GenomeAMRProfile:
    """Parse a saved AMRFinderPlus TSV output file."""
    if not tsv_path.exists():
        raise FileNotFoundError(f"TSV not found: {tsv_path}")

    text = tsv_path.read_text()
    sid = sample_id or tsv_path.stem
    return parse_amrfinder_output(text, sample_id=sid)
