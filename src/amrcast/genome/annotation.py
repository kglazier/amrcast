"""Gene calling using Pyrodigal."""

from pathlib import Path

import pyrodigal
from Bio import SeqIO

from amrcast.genome.models import CalledGene


def call_genes(fasta_path: Path, min_length: int = 500) -> tuple[list[CalledGene], dict]:
    """Call genes from a FASTA assembly using Pyrodigal (metagenomic mode).

    Returns:
        Tuple of (list of called genes, genome stats dict).
    """
    # Read contigs
    contigs = []
    for record in SeqIO.parse(str(fasta_path), "fasta"):
        if len(record.seq) >= min_length:
            contigs.append(record)

    if not contigs:
        raise ValueError(f"No contigs >= {min_length}bp found in {fasta_path}")

    genome_stats = {
        "num_contigs": len(contigs),
        "total_length": sum(len(c.seq) for c in contigs),
    }

    # Use metagenomic mode — works without training on the specific genome
    gene_finder = pyrodigal.GeneFinder(meta=True)

    genes: list[CalledGene] = []
    gene_counter = 0

    for contig in contigs:
        sequence = str(contig.seq)
        predicted = gene_finder.find_genes(sequence.encode())

        for pred in predicted:
            gene_counter += 1
            protein = pred.translate()
            genes.append(
                CalledGene(
                    gene_id=f"gene_{gene_counter:05d}",
                    contig_id=contig.id,
                    start=pred.begin,
                    end=pred.end,
                    strand=pred.strand,
                    protein_sequence=protein if isinstance(protein, str) else str(protein),
                )
            )

    return genes, genome_stats
