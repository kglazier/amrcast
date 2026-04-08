# AMRCast

Quantitative antibiotic resistance prediction from whole-genome sequencing. Predicts **MIC values** (not just Resistant/Susceptible) for 28 antibiotics in *E. coli*, with SHAP-based clinical explanations.

## Results

Trained on 10,654 *E. coli* isolates from NARMS (FDA/CDC/USDA) with standardized Sensititre broth microdilution MIC testing. Evaluated with 5-fold cross-validation.

| Antibiotic | Samples | Essential Agreement | MAE (log2) |
|---|---|---|---|
| Kanamycin | 1,130 | **99.1%** | 0.11 |
| Tetracycline | 5,923 | **98.1%** | 0.17 |
| Orbifloxacin | 1,365 | **98.3%** | 0.12 |
| Nalidixic acid | 4,467 | **98.1%** | 0.40 |
| Sulfisoxazole | 3,974 | **98.1%** | 0.22 |
| Ceftriaxone | 3,977 | **97.9%** | 0.18 |
| Cefoxitin | 4,437 | **97.8%** | 0.43 |
| Azithromycin | 4,467 | **97.6%** | 0.50 |
| Chloramphenicol | 6,017 | **97.1%** | 0.50 |
| Gentamicin | 6,020 | **95.9%** | 0.56 |
| Meropenem | 3,354 | **94.2%** | 0.17 |
| Ampicillin | 6,040 | **93.7%** | 0.61 |
| Ciprofloxacin | 4,526 | **92.9%** | 0.27 |
| Streptomycin | 3,970 | **91.7%** | 0.58 |
| Imipenem | 2,044 | **91.1%** | 0.33 |

25 of 28 antibiotics achieve >90% Essential Agreement (within 1 doubling dilution).

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
# Predict MIC for a genome (uses pre-trained NARMS models)
amrcast predict run genome.fasta

# With SHAP explanations
amrcast predict run genome.fasta --explain

# Specific antibiotics
amrcast predict run genome.fasta --antibiotics ampicillin,ciprofloxacin,gentamicin
```

### Example output

```
amrcast predict run genome.fasta --explain --antibiotics ampicillin,tetracycline
```

```json
{
  "predictions": [
    {
      "antibiotic": "ampicillin",
      "predicted_mic_ug_ml": 16.0,
      "clinical_category": "Intermediate",
      "explanation": {
        "top_features": [
          {"feature": "blaCMY-2", "shap_value": 0.91, "annotation": "AmpC cephalosporinase"},
          {"feature": "blaCTX-M-14", "shap_value": 0.82, "annotation": "ESBL; hydrolyzes 3rd-gen cephalosporins"}
        ],
        "breakpoint": {"susceptible_lte": 8, "resistant_gte": 32}
      }
    },
    {
      "antibiotic": "tetracycline",
      "predicted_mic_ug_ml": 16.0,
      "clinical_category": "Resistant",
      "explanation": {
        "top_features": [
          {"feature": "tet(A)", "shap_value": 1.67, "annotation": "Tetracycline efflux pump"}
        ]
      }
    }
  ]
}
```

With `--explain`, a human-readable report is also printed:

```
tetracycline: 16.0 ug/mL -> Resistant
  CLSI breakpoints: S <= 4 | R >= 16 ug/mL

  Top contributing features:
    tet(A)_present          (+1.67 log2)  [= 1.0]
      -> Tetracycline resistance; efflux pump or ribosomal protection
```

### Train your own models

```bash
# Download NARMS data from NCBI (metadata only, ~500 MB)
# Then train on all antibiotics with 5-fold CV
amrcast train run --data-dir data/narms --cv --n-folds 5
```

## Architecture

- **Gene detection:** [AMRFinderPlus](https://github.com/ncbi/amr/wiki) (NCBI) — curated, organism-specific, includes point mutations
- **ML model:** XGBoost on gene/mutation presence features (649 features)
- **Training data:** NARMS (FDA/CDC/USDA) via NCBI BioSample antibiogram + Pathogen Detection
- **Explainability:** SHAP TreeExplainer + CLSI M100 breakpoints + gene annotations
- **Species:** *E. coli* (Enterobacterales)

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
