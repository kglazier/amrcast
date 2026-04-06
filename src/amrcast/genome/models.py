"""Data models for AMRFinderPlus output."""

from pydantic import BaseModel


class AMRFinderHit(BaseModel):
    """A single row from AMRFinderPlus output."""

    contig_id: str
    start: int
    stop: int
    strand: str
    element_symbol: str
    element_name: str
    scope: str  # "core" or "plus"
    type: str  # "AMR", "STRESS", "VIRULENCE"
    subtype: str
    drug_class: str
    drug_subclass: str
    method: str  # "EXACTX", "BLASTX", "PARTIALX", "HMM", "POINTX" etc.
    target_length: int
    ref_length: int
    coverage: float  # percent
    identity: float  # percent
    closest_ref: str
    closest_ref_name: str


class GenomeAMRProfile(BaseModel):
    """Complete AMRFinderPlus results for a genome."""

    sample_id: str
    hits: list[AMRFinderHit]

    @property
    def amr_hits(self) -> list[AMRFinderHit]:
        """Only AMR-type hits (not virulence/stress)."""
        return [h for h in self.hits if h.type == "AMR"]

    @property
    def point_mutations(self) -> list[AMRFinderHit]:
        """Hits detected via point mutation method."""
        return [h for h in self.hits if h.method in ("POINTX", "POINTN")]

    @property
    def gene_symbols(self) -> list[str]:
        """Unique AMR gene symbols detected."""
        return sorted({h.element_symbol for h in self.amr_hits})

    @property
    def drug_classes(self) -> list[str]:
        """Unique drug classes with detected resistance."""
        return sorted({h.drug_class for h in self.amr_hits if h.drug_class != "NA"})
