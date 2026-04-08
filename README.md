# AMRCast

Quantitative antibiotic resistance prediction from whole-genome sequencing. Predicts **MIC values** (not just Resistant/Susceptible) for multiple antibiotics across three bacterial species, with SHAP-based clinical explanations.

## Results

Trained on NARMS (FDA/CDC/USDA) isolates with standardized Sensititre broth microdilution MIC testing. Evaluated with 5-fold cross-validation.

### *E. coli* — 28 antibiotics, 10,654 isolates

| Antibiotic | Samples | Essential Agreement | MAE (log2) |
|---|---|---|---|
| Kanamycin | 1,130 | **99.1%** | 0.11 |
| Tetracycline | 5,923 | **98.1%** | 0.17 |
| Nalidixic acid | 4,467 | **98.1%** | 0.40 |
| Ceftriaxone | 3,977 | **97.9%** | 0.18 |
| Cefoxitin | 4,437 | **97.8%** | 0.43 |
| Azithromycin | 4,467 | **97.6%** | 0.50 |
| Chloramphenicol | 6,017 | **97.1%** | 0.50 |
| Gentamicin | 6,020 | **95.9%** | 0.56 |
| Meropenem | 3,354 | **94.2%** | 0.17 |
| Ampicillin | 6,040 | **93.7%** | 0.61 |
| Ciprofloxacin | 4,526 | **92.9%** | 0.27 |

25 of 28 antibiotics achieve >90% Essential Agreement.

### *Salmonella enterica* — 12 antibiotics, 937 isolates

| Antibiotic | Samples | Essential Agreement | MAE (log2) |
|---|---|---|---|
| Meropenem | 930 | **100.0%** | 0.01 |
| Azithromycin | 932 | **99.9%** | 0.30 |
| Ceftriaxone | 932 | **99.4%** | 0.07 |
| Tetracycline | 935 | **99.0%** | 0.07 |
| Ampicillin | 936 | **98.9%** | 0.29 |
| Gentamicin | 935 | **97.2%** | 0.52 |
| Ciprofloxacin | 932 | **96.1%** | 0.39 |
| Streptomycin | 932 | **91.4%** | 0.52 |

12 of 12 antibiotics achieve >91% Essential Agreement.

### *Klebsiella pneumoniae* — 26 antibiotics, 926 isolates

| Antibiotic | Samples | Essential Agreement | MAE (log2) |
|---|---|---|---|
| Ampicillin | 551 | **98.9%** | 0.11 |
| Minocycline | 161 | **95.1%** | 0.62 |
| Tobramycin | 567 | **90.3%** | 0.57 |
| Tetracycline | 477 | **84.9%** | 0.82 |
| Amikacin | 626 | **84.0%** | 0.85 |
| Gentamicin | 645 | **82.5%** | 0.81 |
| Ciprofloxacin | 650 | **79.1%** | 0.99 |

Klebsiella is a harder target — complex carbapenem resistance mechanisms and smaller training set.

## How it works

```
Genome FASTA
  -> AMRFinderPlus (NCBI) detects resistance genes + point mutations
  -> Feature extraction (gene presence, mutation counts)
  -> XGBoost predicts MIC per antibiotic (log2 scale)
  -> SHAP explains which genes drive the prediction
  -> Clinical report (S/I/R via CLSI breakpoints)
```

AMRFinderPlus handles gene detection (a solved problem). AMRCast builds the **quantitative prediction and interpretation layer** on top.

## Quick start

### Prerequisites

AMRFinderPlus must be installed. On Windows (via WSL):

```bash
wsl --install
# In WSL:
conda install -y -c bioconda -c conda-forge ncbi-amrfinderplus
amrfinder -u
```

On Linux/Mac:
```bash
conda install -c bioconda -c conda-forge ncbi-amrfinderplus
amrfinder -u
```

### Install

```bash
pip install -e .
```

### Predict

```bash
# E. coli (default)
amrcast predict run genome.fasta

# Salmonella
amrcast predict run genome.fasta --organism salmonella

# Klebsiella
amrcast predict run genome.fasta --organism klebsiella

# With SHAP explanations
amrcast predict run genome.fasta --explain

# Specific antibiotics
amrcast predict run genome.fasta --antibiotics ampicillin,ciprofloxacin,gentamicin

# JSON output (for piping)
amrcast predict run genome.fasta --format json
```

### Example output

```
$ amrcast predict run genome.fasta --antibiotics ampicillin,ciprofloxacin,tetracycline

  genome.fasta
  24 AMR genes, 6 point mutations

  Antibiotic                    MIC (ug/mL)       Category
  ---------------------------- ------------ --------------
  ampicillin                            8.0    Susceptible
  ciprofloxacin                         4.0 ** Resistant **
  tetracycline                         16.0 ** Resistant **
```

With `--explain`:

```
ciprofloxacin: 4.0 ug/mL -> Resistant
  CLSI breakpoints: S <= 0.25 | R >= 1 ug/mL

  Top contributing features:
    gyrA_S83L=POINT_present        (+3.84 log2)  [= 1.0]
      -> DNA gyrase subunit A; point mutations reduce quinolone binding
    parC_S80I=POINT_present        (+1.63 log2)  [= 1.0]
      -> Topoisomerase IV subunit C; point mutations reduce quinolone binding
    gyrA_D87N=POINT_present        (+1.24 log2)  [= 1.0]
      -> DNA gyrase subunit A; point mutations reduce quinolone binding
```

### Reproduce the training data

No genome downloads needed. The training pipeline uses NCBI pre-computed AMRFinderPlus results + MIC data from BioSample antibiogram tables.

```bash
# E. coli
mkdir -p data/narms
curl -o data/narms/amr_metadata.tsv \
  "https://ftp.ncbi.nlm.nih.gov/pathogen/Results/Escherichia_coli_Shigella/latest_snps/AMR/<current_version>.amr.metadata.tsv"
python -c "
from amrcast.data.ncbi_narms import download_antibiogram_data
from pathlib import Path
download_antibiogram_data(Path('data/narms/antibiogram_mic.csv'))
"

# Salmonella
mkdir -p data/salmonella
curl -o data/salmonella/amr_metadata.tsv \
  "https://ftp.ncbi.nlm.nih.gov/pathogen/Results/Salmonella/latest_snps/AMR/<current_version>.amr.metadata.tsv"
python -c "
from amrcast.data.ncbi_narms import download_antibiogram_data
from pathlib import Path
download_antibiogram_data(Path('data/salmonella/antibiogram_mic.csv'), organism='Salmonella enterica')
"

# Klebsiella
mkdir -p data/klebsiella
curl -o data/klebsiella/amr_metadata.tsv \
  "https://ftp.ncbi.nlm.nih.gov/pathogen/Results/Klebsiella/latest_snps/AMR/<current_version>.amr.metadata.tsv"
python -c "
from amrcast.data.ncbi_narms import download_antibiogram_data
from pathlib import Path
download_antibiogram_data(Path('data/klebsiella/antibiogram_mic.csv'), organism='Klebsiella pneumoniae')
"
```

Then train with:
```python
from amrcast.cli.train_species import train_species
from pathlib import Path

results = train_species(Path("data/narms"))       # E. coli
results = train_species(Path("data/salmonella"))   # Salmonella
results = train_species(Path("data/klebsiella"), platform_filter=None, min_isolates=100)  # Klebsiella
```

Check the FTP directories for current filenames — the version number changes daily.

## Architecture

- **Gene detection:** [AMRFinderPlus](https://github.com/ncbi/amr/wiki) (NCBI) -- curated, organism-specific, includes point mutations
- **ML model:** XGBoost on gene/mutation presence features
- **Training data:** NARMS (FDA/CDC/USDA) via NCBI BioSample antibiogram + Pathogen Detection
- **Explainability:** SHAP TreeExplainer + CLSI M100 breakpoints + gene annotations
- **Species:** *E. coli*, *Salmonella enterica*, *Klebsiella pneumoniae*

## Development

```bash
pip install -e ".[dev]"
pytest                    # 41 tests
```

## What this is NOT

- Not a gene detection tool (use AMRFinderPlus)
- Not a binary R/S classifier (existing tools do this)
- Not for TB (CRyPTIC consortium has this covered)

## License

MIT
