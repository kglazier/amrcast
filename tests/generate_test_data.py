"""Generate synthetic test data to validate the full AMRCast pipeline.

Creates:
- Fake E. coli genome FASTA (realistic-ish contigs)
- Fake CARD protein FASTA (small set of AMR reference proteins)
- Fake AMR metadata CSV with MIC values
"""

import random
from pathlib import Path

import pandas as pd

# Realistic amino acid frequencies
AA = "ACDEFGHIKLMNPQRSTVWY"


def random_dna(length: int) -> str:
    return "".join(random.choices("ATCG", k=length))


def random_protein(length: int) -> str:
    return "M" + "".join(random.choices(AA, k=length - 1))


def generate_mock_genome(
    output_path: Path,
    n_contigs: int = 5,
    seed: int = 42,
    inject_proteins: list[str] | None = None,
) -> None:
    """Generate a fake multi-contig E. coli-like genome FASTA.

    If inject_proteins is provided, embeds the reverse-translated DNA of those
    proteins into the first contig so phmmer will find them.
    """
    random.seed(seed)

    # Simple codon table for embedding proteins in DNA
    aa_to_codon = {
        "M": "ATG", "F": "TTT", "L": "CTG", "I": "ATT", "V": "GTG",
        "S": "TCG", "P": "CCG", "T": "ACG", "A": "GCG", "Y": "TAT",
        "H": "CAT", "Q": "CAG", "N": "AAT", "K": "AAA", "D": "GAT",
        "E": "GAG", "C": "TGT", "W": "TGG", "R": "CGG", "G": "GGT",
        "*": "TAA",
    }

    with open(output_path, "w") as f:
        for i in range(n_contigs):
            contig_len = random.randint(50000, 200000)
            seq = random_dna(contig_len)

            # Inject known proteins into first contig
            if i == 0 and inject_proteins:
                offset = 1000
                for prot in inject_proteins:
                    # Reverse translate protein to DNA (in frame)
                    codons = [aa_to_codon.get(aa, "NNN") for aa in prot]
                    codons.append("TAA")  # stop codon
                    dna_insert = "".join(codons)
                    # Place in the contig
                    if offset + len(dna_insert) < contig_len:
                        seq = seq[:offset] + dna_insert + seq[offset + len(dna_insert):]
                        offset += len(dna_insert) + 500  # space between genes

            f.write(f">contig_{i+1} length={contig_len}\n")
            for j in range(0, len(seq), 80):
                f.write(seq[j : j + 80] + "\n")


def generate_mock_card_proteins(output_path: Path, n_proteins: int = 30, seed: int = 42) -> None:
    """Generate fake CARD-like protein reference FASTA.

    Uses realistic AMR gene family names so feature extraction produces
    meaningful column names.
    """
    random.seed(seed)

    # Real AMR gene family names for realism
    gene_families = [
        "blaOXA-1", "blaOXA-48", "blaTEM-1", "blaCTX-M-15", "blaNDM-1",
        "blaSHV-12", "blaKPC-2", "aac(6')-Ib", "aac(3)-IIa", "aph(3'')-Ib",
        "tetA", "tetB", "tetM", "sul1", "sul2",
        "dfrA1", "dfrA17", "mcr-1", "qnrS1", "qnrB1",
        "gyrA_S83L", "gyrA_D87N", "parC_S80I", "parE_S458A",
        "ampC_promoter", "acrB", "marR", "ompF_loss", "ermB", "catA1",
    ]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        for i, name in enumerate(gene_families[:n_proteins]):
            prot_len = random.randint(150, 600)
            seq = random_protein(prot_len)
            f.write(f">gb|FAKE{i:04d}|ARO:300{i:04d}|{name} [Escherichia coli]\n")
            for j in range(0, len(seq), 80):
                f.write(seq[j : j + 80] + "\n")


def generate_mock_metadata(
    output_path: Path,
    genome_ids: list[str],
    seed: int = 42,
) -> None:
    """Generate fake AMR metadata CSV with MIC values.

    Creates realistic MIC distributions for ciprofloxacin and ampicillin.
    """
    random.seed(seed)

    antibiotics = {
        "ciprofloxacin": {
            "susceptible_mics": [0.008, 0.015, 0.03, 0.06, 0.125, 0.25],
            "resistant_mics": [4, 8, 16, 32, 64],
        },
        "ampicillin": {
            "susceptible_mics": [1, 2, 4],
            "resistant_mics": [16, 32, 64, 128, 256],
        },
    }

    rows = []
    for gid in genome_ids:
        for ab, mics in antibiotics.items():
            # ~60% resistant for test diversity
            if random.random() < 0.6:
                mic = random.choice(mics["resistant_mics"])
                phenotype = "Resistant"
            else:
                mic = random.choice(mics["susceptible_mics"])
                phenotype = "Susceptible"

            rows.append({
                "genome_id": gid,
                "antibiotic": ab,
                "measurement_value": mic,
                "measurement_sign": "=",
                "measurement_unit": "ug/mL",
                "resistant_phenotype": phenotype,
                "laboratory_typing_method": "MIC",
            })

    df = pd.DataFrame(rows)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)


def generate_full_test_dataset(base_dir: Path, n_genomes: int = 10) -> dict:
    """Generate a complete mock dataset for pipeline testing.

    Returns dict with paths to all generated files.
    """
    base_dir.mkdir(parents=True, exist_ok=True)

    # Generate CARD proteins FIRST so we can inject some into genomes
    card_dir = base_dir / "card" / "hmms"
    card_fasta = card_dir / "card_proteins.fasta"
    generate_mock_card_proteins(card_fasta)

    # Read back the CARD proteins for injection
    card_proteins = []
    with open(card_fasta) as f:
        current_seq = ""
        for line in f:
            if line.startswith(">"):
                if current_seq:
                    card_proteins.append(current_seq)
                current_seq = ""
            else:
                current_seq += line.strip()
        if current_seq:
            card_proteins.append(current_seq)

    # Generate genomes, injecting random subsets of CARD proteins
    genomes_dir = base_dir / "raw" / "genomes"
    genomes_dir.mkdir(parents=True, exist_ok=True)

    genome_ids = []
    for i in range(n_genomes):
        gid = f"mock_{i+1:04d}.{i+1}"
        genome_ids.append(gid)

        # Each genome gets a random subset of 3-8 CARD proteins injected
        random.seed(42 + i)
        n_inject = random.randint(3, min(8, len(card_proteins)))
        inject = random.sample(card_proteins, n_inject)

        generate_mock_genome(
            genomes_dir / f"{gid}.fasta",
            seed=42 + i,
            inject_proteins=inject,
        )

    # Generate metadata
    metadata_path = base_dir / "raw" / "amr_metadata.csv"
    generate_mock_metadata(metadata_path, genome_ids)

    return {
        "data_dir": base_dir,
        "card_dir": card_dir,
        "genomes_dir": genomes_dir,
        "metadata_path": metadata_path,
        "genome_ids": genome_ids,
    }


if __name__ == "__main__":
    import sys

    output = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("data/mock")
    paths = generate_full_test_dataset(output, n_genomes=10)
    print(f"Generated mock dataset in {output}/")
    print(f"  Genomes: {len(paths['genome_ids'])}")
    print(f"  Metadata: {paths['metadata_path']}")
    print(f"  CARD: {paths['card_dir']}")
