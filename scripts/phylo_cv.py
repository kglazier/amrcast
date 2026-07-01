"""Real phylogeny-aware CV using NCBI Pathogen Detection SNP clusters (PDS).

Groups isolates by NCBI's precomputed ~50-SNP clusters so related genomes never
span the train/test boundary. Compares EA under:
  random  : plain KFold (the flattering number)
  genotype: identical AMR-gene-profile groups (the conservative proxy we ran before)
  phylo   : NCBI PDS SNP-cluster groups (the honest test)
"""
import csv
import numpy as np

from amrcast.data.narms_features import build_narms_training_data
from amrcast.ml.xgboost_model import MICPredictor

CLUSTER_FILE = "data/narms/clusters/ecoli_cluster_list.tsv"
DRUGS = ["ampicillin", "ciprofloxacin", "gentamicin", "tetracycline",
         "ceftriaxone", "meropenem", "cefepime", "cefotaxime"]

# biosample_acc -> PDS cluster id
bios_to_pds = {}
with open(CLUSTER_FILE) as fh:
    for row in csv.DictReader(fh, delimiter="\t"):
        bios_to_pds[row["biosample_acc"]] = row["PDS_acc"]

features, targets, geno_groups = build_narms_training_data(
    mic_path="data/narms/antibiogram_mic.csv",
    metadata_path="data/narms/ecoli_amr_metadata.tsv",
    platform_filter="ensitit",
    min_isolates_per_drug=100,
)

# Build phylo group ids: real PDS cluster where known, else unique singleton.
pds_group = {}
next_singleton = -1
for acc in features.index:
    pds = bios_to_pds.get(acc)
    if pds is None:
        pds_group[acc] = next_singleton
        next_singleton -= 1
    else:
        pds_group[acc] = pds

# Coverage / structure diagnostics
n = len(features)
covered = sum(1 for a in features.index if a in bios_to_pds)
from collections import Counter
sizes = Counter(pds_group.values())
in_shared = sum(1 for a in features.index if sizes[pds_group[a]] > 1)
print("=" * 64)
print(f"study isolates (Sensititre)     : {n}")
print(f"mapped to an NCBI SNP cluster   : {covered} ({covered/n:.0%})")
print(f"distinct phylo groups           : {len(sizes)}")
print(f"isolates in a shared PDS cluster: {in_shared} ({in_shared/n:.0%})")
print(f"5 largest PDS cluster sizes     : {sorted(sizes.values(), reverse=True)[:5]}")
print("=" * 64)

print(f"\n{'drug':<15}{'n':>6}{'random':>9}{'genotype':>10}{'phylo':>9}{'r->phylo':>10}")
print("-" * 59)
for ab in DRUGS:
    ab_t = targets[targets["antibiotic"] == ab]
    valid = [a for a in ab_t["biosample_acc"] if a in features.index]
    if len(valid) < 50:
        continue
    X = features.loc[valid].values
    y = ab_t.set_index("biosample_acc").loc[valid]["log2_mic"].values
    g_geno = np.array([geno_groups[a] for a in valid])
    # remap phylo ids (mixed str/int) to contiguous ints for GroupKFold
    uniq = {v: i for i, v in enumerate({pds_group[a] for a in valid})}
    g_phylo = np.array([uniq[pds_group[a]] for a in valid])

    ea = {}
    for label, g in (("random", None), ("genotype", g_geno), ("phylo", g_phylo)):
        cv = MICPredictor(antibiotic=ab).cross_validate(
            X, y, feature_names=list(features.columns), n_folds=5, groups=g)
        ea[label] = cv["essential_agreement_mean"]
    print(f"{ab:<15}{len(valid):>6}{ea['random']:>8.1%}{ea['genotype']:>10.1%}"
          f"{ea['phylo']:>9.1%}{(ea['phylo']-ea['random']):>+10.1%}")
