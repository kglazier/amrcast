"""AMR gene detection using PyHMMER against CARD profiles.

Supports two modes:
- phmmer: sequence-vs-sequence search against CARD protein FASTA (default for vertical slice)
- hmmsearch: sequence-vs-HMM search against pre-built HMM profiles
"""

import logging
from pathlib import Path

import pyhmmer

from amrcast.genome.models import AMRHit, CalledGene

logger = logging.getLogger(__name__)


def _build_query_sequences(genes: list[CalledGene]) -> list[pyhmmer.easel.DigitalSequence]:
    """Convert genes to PyHMMER digital sequences."""
    alphabet = pyhmmer.easel.Alphabet.amino()
    sequences = []
    for gene in genes:
        seq_str = gene.protein_sequence.rstrip("*")
        if not seq_str:
            continue
        seq = pyhmmer.easel.TextSequence(
            name=gene.gene_id.encode(),
            sequence=seq_str,
        )
        sequences.append(seq.digitize(alphabet))
    return sequences


def _load_card_targets(card_dir: Path) -> list[pyhmmer.easel.DigitalSequence]:
    """Load CARD protein sequences as digital sequences for phmmer."""
    alphabet = pyhmmer.easel.Alphabet.amino()
    fasta_files = list(card_dir.glob("*.fasta"))
    if not fasta_files:
        raise FileNotFoundError(f"No .fasta files found in {card_dir}")

    targets = []
    with pyhmmer.easel.SequenceFile(
        str(fasta_files[0]), digital=True, alphabet=alphabet
    ) as seq_file:
        for seq in seq_file:
            targets.append(seq)

    logger.info(f"Loaded {len(targets)} CARD reference proteins")
    return targets


def _parse_card_header(name: str) -> tuple[str, str]:
    """Parse CARD FASTA header to extract gene family and description.

    CARD headers look like: gb|ABC123|ARO:3000001|family_name [organism]
    or: ARO:3000001|gene_name
    """
    parts = name.split("|")

    # Try to extract a meaningful gene family name
    gene_family = parts[-1].split("[")[0].strip() if parts else name
    description = name

    return gene_family, description


def detect_amr_genes_phmmer(
    genes: list[CalledGene],
    card_dir: Path,
    evalue_threshold: float = 1e-20,
    min_coverage: float = 0.8,
) -> list[AMRHit]:
    """Detect AMR genes using phmmer (sequence-vs-sequence) against CARD proteins.

    This is the default mode for the vertical slice since CARD distributes
    protein FASTA files rather than pre-built HMMs.

    Uses strict thresholds to reduce cross-family noise. For example, an E. coli
    acrB efflux pump would otherwise match MexB, smeE, adeJ etc. from other species
    because RND efflux pumps share structural homology. We only want the best
    CARD match per called gene.
    """
    if not genes:
        return []

    queries = _build_query_sequences(genes)
    if not queries:
        return []

    targets = _load_card_targets(card_dir)
    if not targets:
        return []

    gene_lookup = {g.gene_id: g for g in genes}
    hits: list[AMRHit] = []

    # phmmer: search each query against all targets
    for top_hits in pyhmmer.phmmer(queries, targets, cpus=0):
        query_name = top_hits.query.name if top_hits.query else "unknown"

        for hit in top_hits:
            if hit.evalue > evalue_threshold:
                continue
            if not hit.included:
                continue

            gene = gene_lookup.get(query_name)
            if gene is None:
                continue

            # Parse the target (CARD protein) name
            target_name = hit.name or "unknown"
            gene_family, description = _parse_card_header(hit.name or "unknown")

            # Coverage and identity estimates from best domain alignment
            best_domain = hit.best_domain
            coverage = 0.0
            identity = 0.0

            if best_domain and best_domain.alignment:
                ali = best_domain.alignment
                query_len = ali.hmm_length
                if query_len > 0:
                    ali_span = ali.hmm_to - ali.hmm_from + 1
                    coverage = min(ali_span / query_len, 1.0)
                    identity = min(hit.score / (query_len * 3.0), 1.0)

            # Skip low-coverage hits (partial matches / domain-only matches)
            if coverage < min_coverage:
                continue

            hits.append(
                AMRHit(
                    gene_family=gene_family,
                    gene_id=query_name,
                    query_name=target_name,
                    evalue=hit.evalue,
                    score=hit.score,
                    identity=identity,
                    coverage=coverage,
                    protein_sequence=gene.protein_sequence,
                    description=description,
                )
            )

    # Deduplicate: keep SINGLE best CARD hit per called gene.
    # This prevents one E. coli protein from being counted as 10 different
    # AMR gene families just because efflux pumps share homology.
    best_per_gene: dict[str, AMRHit] = {}
    for hit in hits:
        if hit.gene_id not in best_per_gene or hit.score > best_per_gene[hit.gene_id].score:
            best_per_gene[hit.gene_id] = hit

    logger.info(
        f"Found {len(best_per_gene)} AMR hits across {len(genes)} genes "
        f"(filtered from {len(hits)} raw hits)"
    )
    return list(best_per_gene.values())


def detect_amr_genes_hmmsearch(
    genes: list[CalledGene],
    hmm_dir: Path,
    evalue_threshold: float = 1e-10,
) -> list[AMRHit]:
    """Detect AMR genes using hmmsearch against pre-built HMM profiles."""
    if not genes:
        return []

    hmm_files = list(hmm_dir.glob("*.hmm"))
    if not hmm_files:
        raise FileNotFoundError(f"No .hmm files found in {hmm_dir}")

    sequences = _build_query_sequences(genes)
    gene_lookup = {g.gene_id: g for g in genes}

    # Load HMMs
    hmms = []
    for hmm_file in hmm_files:
        with pyhmmer.plan7.HMMFile(str(hmm_file)) as reader:
            for hmm in reader:
                hmms.append(hmm)

    if not hmms:
        raise ValueError(f"No HMM profiles loaded from {hmm_dir}")

    hits: list[AMRHit] = []

    for top_hits in pyhmmer.hmmsearch(hmms, sequences, cpus=0):
        for hit in top_hits:
            if hit.evalue > evalue_threshold:
                continue
            if not hit.included:
                continue

            gene_id = hit.name
            gene = gene_lookup.get(gene_id)
            if gene is None:
                continue

            query_obj = top_hits.query
            hmm_name = query_obj.name if query_obj and query_obj.name else "unknown"
            hmm_acc = query_obj.accession if query_obj and query_obj.accession else ""

            best_domain = hit.best_domain
            coverage = 0.0
            identity = 0.0
            if best_domain and best_domain.alignment:
                ali = best_domain.alignment
                ali_span = ali.hmm_to - ali.hmm_from + 1
                hmm_len = ali.hmm_length
                coverage = ali_span / hmm_len if hmm_len > 0 else 0.0
                identity = min(hit.score / (hmm_len * 3.0), 1.0) if hmm_len > 0 else 0.0

            hits.append(
                AMRHit(
                    gene_family=hmm_name,
                    gene_id=gene_id,
                    query_name=hmm_name,
                    evalue=hit.evalue,
                    score=hit.score,
                    identity=identity,
                    coverage=coverage,
                    protein_sequence=gene.protein_sequence,
                    description=hmm_acc,
                )
            )

    best_hits: dict[tuple[str, str], AMRHit] = {}
    for hit in hits:
        key = (hit.gene_id, hit.gene_family)
        if key not in best_hits or hit.score > best_hits[key].score:
            best_hits[key] = hit

    logger.info(f"Found {len(best_hits)} AMR hits across {len(genes)} genes")
    return list(best_hits.values())


def detect_amr_genes(
    genes: list[CalledGene],
    card_dir: Path,
    evalue_threshold: float = 1e-20,
    min_coverage: float = 0.8,
) -> list[AMRHit]:
    """Detect AMR genes — auto-selects phmmer or hmmsearch based on available files."""
    hmm_files = list(card_dir.glob("*.hmm"))
    fasta_files = list(card_dir.glob("*.fasta"))

    if hmm_files:
        logger.info("Using hmmsearch mode (HMM profiles found)")
        return detect_amr_genes_hmmsearch(genes, card_dir, evalue_threshold)
    elif fasta_files:
        logger.info("Using phmmer mode (CARD protein FASTA found)")
        return detect_amr_genes_phmmer(genes, card_dir, evalue_threshold, min_coverage)
    else:
        raise FileNotFoundError(
            f"No .hmm or .fasta files found in {card_dir}. "
            "Run 'amrcast data download' to fetch CARD database."
        )


def build_gene_family_list(hits: list[AMRHit]) -> list[str]:
    """Extract unique sorted gene family names from AMR hits."""
    return sorted({h.gene_family for h in hits})
