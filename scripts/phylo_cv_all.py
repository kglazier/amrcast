"""Full phylo-aware CV over ALL antibiotics (random vs genotype vs PDS phylo)."""
import csv, json
import numpy as np
from amrcast.data.narms_features import build_narms_training_data
from amrcast.ml.xgboost_model import MICPredictor

bios_to_pds = {}
with open("data/narms/clusters/ecoli_cluster_list.tsv") as fh:
    for row in csv.DictReader(fh, delimiter="\t"):
        bios_to_pds[row["biosample_acc"]] = row["PDS_acc"]

features, targets, geno_groups = build_narms_training_data(
    mic_path="data/narms/antibiogram_mic.csv",
    metadata_path="data/narms/ecoli_amr_metadata.tsv",
    platform_filter="ensitit", min_isolates_per_drug=100)

pds_group, nxt = {}, -1
for acc in features.index:
    p = bios_to_pds.get(acc)
    if p is None:
        pds_group[acc] = nxt; nxt -= 1
    else:
        pds_group[acc] = p

drugs = sorted(targets["antibiotic"].unique())
rows = []
print(f"{'drug':<16}{'n':>6}{'random':>9}{'genotype':>10}{'phylo':>9}{'r->phylo':>10}")
print("-" * 60)
for ab in drugs:
    ab_t = targets[targets["antibiotic"] == ab]
    valid = [a for a in ab_t["biosample_acc"] if a in features.index]
    if len(valid) < 50:
        continue
    X = features.loc[valid].values
    y = ab_t.set_index("biosample_acc").loc[valid]["log2_mic"].values
    g_geno = np.array([geno_groups[a] for a in valid])
    uniq = {v: i for i, v in enumerate({pds_group[a] for a in valid})}
    g_phylo = np.array([uniq[pds_group[a]] for a in valid])
    ea = {}
    for label, g in (("random", None), ("genotype", g_geno), ("phylo", g_phylo)):
        cv = MICPredictor(antibiotic=ab).cross_validate(
            X, y, feature_names=list(features.columns), n_folds=5, groups=g)
        ea[label] = cv["essential_agreement_mean"]
    rows.append({"drug": ab, "n": len(valid), **ea})
    print(f"{ab:<16}{len(valid):>6}{ea['random']:>8.1%}{ea['genotype']:>10.1%}"
          f"{ea['phylo']:>9.1%}{(ea['phylo']-ea['random']):>+10.1%}")

with open("data/narms/phylo_cv_all_results.json", "w") as f:
    json.dump(rows, f, indent=2)
print(f"\nwrote data/narms/phylo_cv_all_results.json ({len(rows)} drugs)")
