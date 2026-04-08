"""Look up AMR reference protein sequences from the AMRFinderPlus database.

This avoids downloading genome assemblies for ESM-2 — we embed the reference
protein for each detected gene variant instead. For hits with >90% identity
(which is the AMRFinderPlus threshold), the reference is a close approximation.
"""

import logging
from pathlib import Path

logger = logging.getLogger(__name__)


def load_amrfinder_reference_proteins(fasta_path: Path) -> dict[str, str]:
    """Load AMRFinderPlus reference protein FASTA into a gene_name -> sequence dict.

    Header format: >WP_000239590.1|1|1|blaCTX-M-15|blaCTX-M|hydrolase|...
    The gene name is field 4 (0-indexed field 3).

    Returns:
        Dict of gene_symbol -> amino acid sequence.
    """
    proteins = {}
    current_name = None
    current_seq = []

    with open(fasta_path) as f:
        for line in f:
            line = line.strip()
            if line.startswith(">"):
                if current_name and current_seq:
                    seq = "".join(current_seq)
                    # Keep the longer sequence if duplicate gene names
                    if current_name not in proteins or len(seq) > len(proteins[current_name]):
                        proteins[current_name] = seq

                # Parse header: >accession|...|...|gene_name|...
                parts = line[1:].split("|")
                current_name = parts[3] if len(parts) > 3 else parts[0]
                current_seq = []
            else:
                current_seq.append(line)

    if current_name and current_seq:
        seq = "".join(current_seq)
        if current_name not in proteins or len(seq) > len(proteins[current_name]):
            proteins[current_name] = seq

    logger.info(f"Loaded {len(proteins)} reference proteins from {fasta_path.name}")
    return proteins


def get_reference_sequences_for_isolate(
    genotype_symbols: list[str],
    reference_db: dict[str, str],
) -> dict[str, str]:
    """Look up reference protein sequences for an isolate's detected genes.

    Args:
        genotype_symbols: Gene symbols detected (e.g., ["blaCTX-M-15", "gyrA_S83L=POINT"]).
        reference_db: Dict from load_amrfinder_reference_proteins().

    Returns:
        Dict of gene_symbol -> protein_sequence (only genes found in reference).
    """
    result = {}
    for sym in genotype_symbols:
        # Strip =POINT suffix for lookup
        clean = sym.replace("=POINT", "").replace("=PARTIAL_END_OF_CONTIG", "")
        if clean in reference_db:
            # Strip stop codons and non-standard characters for ESM-2 compatibility
            seq = reference_db[clean].replace("*", "").replace("X", "")
            if seq:
                result[sym] = seq

    return result
