"""Data models for genome processing outputs."""

from pydantic import BaseModel


class CalledGene(BaseModel):
    """A gene predicted by Pyrodigal."""

    gene_id: str
    contig_id: str
    start: int
    end: int
    strand: int  # 1 or -1
    protein_sequence: str


class AMRHit(BaseModel):
    """An AMR gene detected by HMM search against CARD."""

    gene_family: str
    gene_id: str
    query_name: str
    evalue: float
    score: float
    identity: float  # 0-1 scale
    coverage: float  # 0-1 scale
    protein_sequence: str
    description: str = ""


class GenomeAnnotation(BaseModel):
    """Complete annotation output for a genome."""

    sample_id: str
    num_contigs: int
    total_length: int
    genes: list[CalledGene]
    amr_hits: list[AMRHit]
