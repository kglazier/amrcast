"""Genome processing pipeline — gene calling + AMR detection in one step."""

from pathlib import Path

from amrcast.genome.amr_detection import detect_amr_genes
from amrcast.genome.annotation import call_genes
from amrcast.genome.models import GenomeAnnotation


def process_genome(
    fasta_path: Path,
    card_dir: Path,
    min_contig_length: int = 500,
    evalue_threshold: float = 1e-10,
) -> GenomeAnnotation:
    """Run the full genome processing pipeline.

    1. Call genes with Pyrodigal
    2. Detect AMR genes with PyHMMER against CARD database

    Args:
        fasta_path: Path to input FASTA assembly.
        card_dir: Directory containing CARD .hmm or .fasta files.
        min_contig_length: Minimum contig length for gene calling.
        evalue_threshold: E-value cutoff for AMR hits.

    Returns:
        GenomeAnnotation with genes and AMR hits.
    """
    genes, stats = call_genes(fasta_path, min_length=min_contig_length)

    amr_hits = detect_amr_genes(
        genes=genes,
        card_dir=card_dir,
        evalue_threshold=evalue_threshold,
    )

    return GenomeAnnotation(
        sample_id=fasta_path.stem,
        num_contigs=stats["num_contigs"],
        total_length=stats["total_length"],
        genes=genes,
        amr_hits=amr_hits,
    )
