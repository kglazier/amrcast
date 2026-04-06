"""Extract protein sequences for AMR genes detected by AMRFinderPlus.

AMRFinderPlus in nucleotide mode reports the closest reference accession
for each hit. We can extract the actual translated protein from the genome
using the coordinates AMRFinderPlus provides, or look up the reference
sequence from the AMRFinderPlus database.

For ESM-2 embedding, we use a two-step approach:
1. Use AMRFinderPlus coordinates to extract the nucleotide region
2. Translate to protein using standard codon table

This gives us the ACTUAL protein from this specific genome, not the reference —
which is exactly what ESM-2 needs to detect novel variants.
"""

import logging
from pathlib import Path

from amrcast.genome.models import AMRFinderHit, GenomeAMRProfile

logger = logging.getLogger(__name__)

# Standard codon table
CODON_TABLE = {
    "TTT": "F", "TTC": "F", "TTA": "L", "TTG": "L",
    "CTT": "L", "CTC": "L", "CTA": "L", "CTG": "L",
    "ATT": "I", "ATC": "I", "ATA": "I", "ATG": "M",
    "GTT": "V", "GTC": "V", "GTA": "V", "GTG": "V",
    "TCT": "S", "TCC": "S", "TCA": "S", "TCG": "S",
    "CCT": "P", "CCC": "P", "CCA": "P", "CCG": "P",
    "ACT": "T", "ACC": "T", "ACA": "T", "ACG": "T",
    "GCT": "A", "GCC": "A", "GCA": "A", "GCG": "A",
    "TAT": "Y", "TAC": "Y", "TAA": "*", "TAG": "*",
    "CAT": "H", "CAC": "H", "CAA": "Q", "CAG": "Q",
    "AAT": "N", "AAC": "N", "AAA": "K", "AAG": "K",
    "GAT": "D", "GAC": "D", "GAA": "E", "GAG": "E",
    "TGT": "C", "TGC": "C", "TGA": "*", "TGG": "W",
    "CGT": "R", "CGC": "R", "CGA": "R", "CGG": "R",
    "AGT": "S", "AGC": "S", "AGA": "R", "AGG": "R",
    "GGT": "G", "GGC": "G", "GGA": "G", "GGG": "G",
}


def _reverse_complement(seq: str) -> str:
    comp = {"A": "T", "T": "A", "C": "G", "G": "C", "N": "N"}
    return "".join(comp.get(b, "N") for b in reversed(seq.upper()))


def _translate(dna: str) -> str:
    protein = []
    for i in range(0, len(dna) - 2, 3):
        codon = dna[i:i+3].upper()
        aa = CODON_TABLE.get(codon, "X")
        if aa == "*":
            break
        protein.append(aa)
    return "".join(protein)


def _read_fasta_contigs(fasta_path: Path) -> dict[str, str]:
    """Read a FASTA file into a dict of contig_id -> sequence."""
    contigs = {}
    current_id = None
    current_seq = []

    with open(fasta_path) as f:
        for line in f:
            line = line.strip()
            if line.startswith(">"):
                if current_id:
                    contigs[current_id] = "".join(current_seq)
                current_id = line[1:].split()[0]
                current_seq = []
            else:
                current_seq.append(line)
    if current_id:
        contigs[current_id] = "".join(current_seq)

    return contigs


def extract_proteins_from_genome(
    profile: GenomeAMRProfile,
    fasta_path: Path,
) -> dict[str, str]:
    """Extract actual protein sequences for AMR hits from the genome FASTA.

    Uses the coordinates from AMRFinderPlus to extract the nucleotide region,
    then translates to protein.

    Args:
        profile: AMRFinderPlus results with hit coordinates.
        fasta_path: Path to the genome FASTA file.

    Returns:
        Dict of element_symbol -> protein_sequence.
    """
    contigs = _read_fasta_contigs(fasta_path)
    proteins = {}

    for hit in profile.amr_hits:
        if hit.contig_id not in contigs:
            continue

        contig_seq = contigs[hit.contig_id]
        start = hit.start - 1  # Convert 1-based to 0-based
        stop = hit.stop

        if start < 0 or stop > len(contig_seq):
            continue

        nuc_seq = contig_seq[start:stop]

        if hit.strand == "-":
            nuc_seq = _reverse_complement(nuc_seq)

        protein = _translate(nuc_seq)

        if len(protein) >= 10:  # Skip very short fragments
            # Use element_symbol as key; if duplicate symbols, keep longer
            if hit.element_symbol not in proteins or len(protein) > len(proteins[hit.element_symbol]):
                proteins[hit.element_symbol] = protein

    logger.info(
        f"Extracted {len(proteins)} protein sequences from {fasta_path.name}"
    )
    return proteins
