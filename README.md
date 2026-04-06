# AMRCast

ML-powered antibiotic resistance prediction with quantitative MIC forecasting from whole genome sequencing data.

## What it does

AMRCast takes a bacterial genome assembly and predicts **quantitative MIC values** (minimum inhibitory concentration) — not just binary Resistant/Susceptible. It uses [NCBI AMRFinderPlus](https://github.com/ncbi/amr/wiki) for gene and point mutation detection, then applies machine learning to predict how resistant the organism actually is.

## Architecture

```
Genome FASTA → AMRFinderPlus (gene detection + point mutations)
             → Feature extraction (gene presence, identity, mutations, drug classes)
             → XGBoost MIC prediction (log2 scale, per antibiotic)
             → SHAP explanations
             → JSON output with predicted MIC values
```

AMRFinderPlus is an established, NCBI-maintained tool for AMR gene detection. AMRCast builds on top of it — we don't reinvent gene detection, we predict MIC from its output.

## Prerequisites

### AMRFinderPlus (required)

AMRFinderPlus runs on Linux. On Windows, install via WSL:

```bash
# Install WSL (admin PowerShell)
wsl --install

# Inside WSL, install miniconda + AMRFinderPlus
curl -sSL https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -o miniconda.sh
bash miniconda.sh -b -p $HOME/miniconda3
export PATH=$HOME/miniconda3/bin:$PATH
conda install -y -c bioconda -c conda-forge ncbi-amrfinderplus
amrfinder -u  # download database
```

On Linux/Mac:
```bash
conda install -c bioconda -c conda-forge ncbi-amrfinderplus
amrfinder -u
```

## Install

```bash
pip install -e .          # core
pip install -e ".[dev]"   # with dev tools
pip install -e ".[gpu]"   # with ESM-2 support (future)
```

## Usage

```bash
# Download training data (E. coli genomes + MIC values from BV-BRC)
amrcast data download --n-genomes 100

# Train models
amrcast train run

# Predict MIC for a new genome
amrcast predict run genome.fasta
amrcast predict run genome.fasta --explain  # with SHAP explanations
```

## Current status

- **Species:** E. coli (target: multi-species)
- **Antibiotics:** Ciprofloxacin, Ampicillin (target: 15+ antibiotics)
- **Gene detection:** AMRFinderPlus (NCBI) — curated, organism-specific, includes point mutations
- **ML model:** XGBoost on gene presence/identity + point mutations + drug class features
- **Planned:** ESM-2 protein language model embeddings for novel variant detection

## License

MIT
