"""CLSI breakpoint data and clinical interpretation.

Maps predicted MIC values to Susceptible / Intermediate / Resistant categories
using CLSI M100 breakpoints for Enterobacterales (E. coli).
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class Breakpoint:
    """CLSI breakpoint for one antibiotic."""
    antibiotic: str
    susceptible_lte: float  # MIC ≤ this = Susceptible (µg/mL)
    resistant_gte: float    # MIC ≥ this = Resistant (µg/mL)
    unit: str = "µg/mL"
    source: str = "CLSI M100 (2024), Enterobacterales"


# CLSI M100 breakpoints for Enterobacterales
# Source: CLSI M100, 34th Edition (2024)
BREAKPOINTS: dict[str, Breakpoint] = {
    "ampicillin": Breakpoint(
        antibiotic="ampicillin",
        susceptible_lte=8,
        resistant_gte=32,
    ),
    "ciprofloxacin": Breakpoint(
        antibiotic="ciprofloxacin",
        susceptible_lte=0.25,
        resistant_gte=1,
    ),
    "gentamicin": Breakpoint(
        antibiotic="gentamicin",
        susceptible_lte=4,
        resistant_gte=16,
    ),
    "tetracycline": Breakpoint(
        antibiotic="tetracycline",
        susceptible_lte=4,
        resistant_gte=16,
    ),
    "trimethoprim-sulfamethoxazole": Breakpoint(
        antibiotic="trimethoprim-sulfamethoxazole",
        susceptible_lte=2,
        resistant_gte=4,
    ),
    "ceftriaxone": Breakpoint(
        antibiotic="ceftriaxone",
        susceptible_lte=1,
        resistant_gte=4,
    ),
    "meropenem": Breakpoint(
        antibiotic="meropenem",
        susceptible_lte=1,
        resistant_gte=4,
    ),
    "amoxicillin-clavulanate": Breakpoint(
        antibiotic="amoxicillin-clavulanate",
        susceptible_lte=8,
        resistant_gte=32,
    ),
    "cefazolin": Breakpoint(
        antibiotic="cefazolin",
        susceptible_lte=2,
        resistant_gte=8,
    ),
    "azithromycin": Breakpoint(
        antibiotic="azithromycin",
        susceptible_lte=16,
        resistant_gte=32,
    ),
}


def classify_mic(antibiotic: str, mic_ug_ml: float) -> str:
    """Classify MIC as S/I/R using CLSI breakpoints.

    Returns "Susceptible", "Intermediate", or "Resistant".
    Returns "Unknown" if no breakpoint data for this antibiotic.
    """
    bp = BREAKPOINTS.get(antibiotic.lower())
    if bp is None:
        return "Unknown"

    if mic_ug_ml <= bp.susceptible_lte:
        return "Susceptible"
    elif mic_ug_ml >= bp.resistant_gte:
        return "Resistant"
    else:
        return "Intermediate"


def get_breakpoint(antibiotic: str) -> Breakpoint | None:
    return BREAKPOINTS.get(antibiotic.lower())


# Common AMR gene annotations — maps gene symbol patterns to human-readable descriptions
GENE_ANNOTATIONS: dict[str, str] = {
    # Beta-lactamases
    "blaTEM": "TEM-type beta-lactamase; hydrolyzes penicillins and early cephalosporins",
    "blaSHV": "SHV-type beta-lactamase; hydrolyzes penicillins, some cephalosporin activity",
    "blaCTX-M": "CTX-M extended-spectrum beta-lactamase; hydrolyzes 3rd-gen cephalosporins",
    "blaOXA": "OXA-type beta-lactamase; variable spectrum including carbapenems",
    "blaCMY": "AmpC-type cephalosporinase; hydrolyzes cephamycins and 3rd-gen cephalosporins",
    "blaKPC": "KPC carbapenemase; hydrolyzes carbapenems, confers extensive resistance",
    "blaNDM": "NDM metallo-beta-lactamase; hydrolyzes nearly all beta-lactams",
    # Aminoglycosides
    "aac": "Aminoglycoside acetyltransferase; modifies aminoglycoside antibiotics",
    "aph": "Aminoglycoside phosphotransferase; modifies aminoglycoside antibiotics",
    "ant": "Aminoglycoside nucleotidyltransferase; modifies aminoglycoside antibiotics",
    # Quinolone resistance
    "gyrA": "DNA gyrase subunit A; point mutations reduce quinolone binding",
    "gyrB": "DNA gyrase subunit B; point mutations reduce quinolone binding",
    "parC": "Topoisomerase IV subunit C; point mutations reduce quinolone binding",
    "parE": "Topoisomerase IV subunit E; point mutations reduce quinolone binding",
    "qnr": "Quinolone resistance protein; protects DNA gyrase from quinolones",
    "aac(6')-Ib-cr": "Bifunctional enzyme; acetylates both aminoglycosides and ciprofloxacin",
    # Tetracycline
    "tet": "Tetracycline resistance; efflux pump or ribosomal protection",
    # Sulfonamides / Trimethoprim
    "sul": "Sulfonamide resistance; drug-insensitive dihydropteroate synthase",
    "dfr": "Trimethoprim resistance; drug-insensitive dihydrofolate reductase",
    "dfrA": "Trimethoprim resistance; drug-insensitive dihydrofolate reductase",
    # Colistin
    "mcr": "Mobilized colistin resistance; lipid A modification",
    # Efflux
    "oqxA": "Multidrug efflux pump subunit; quinolone and other resistance",
    "oqxB": "Multidrug efflux pump subunit; quinolone and other resistance",
    # Fosfomycin
    "fosA": "Fosfomycin resistance; glutathione transferase",
}


def annotate_gene(symbol: str) -> str:
    """Get a human-readable annotation for an AMR gene symbol.

    Tries exact match first, then prefix match.
    """
    # Exact match
    if symbol in GENE_ANNOTATIONS:
        return GENE_ANNOTATIONS[symbol]

    # Prefix match (e.g., "blaTEM-1" matches "blaTEM")
    for prefix, annotation in sorted(GENE_ANNOTATIONS.items(), key=lambda x: -len(x[0])):
        if symbol.startswith(prefix):
            return annotation

    return "AMR-associated gene"
