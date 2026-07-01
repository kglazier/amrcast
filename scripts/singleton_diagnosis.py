"""Are the 66% unmatched isolates GENUINE singletons or a data/timing gap?

Cross-references study isolates against NCBI's all_isolates.tsv (every genome in
the SNP analysis, clustered or not). For each study isolate:
  clustered      : in tree, assigned to a PDS cluster (>=2 members)
  true singleton : in tree, no PDS  -> report nearest-neighbor SNP distance
  not in tree    : absent from all_isolates -> genuine data/timing gap
"""
import csv
import numpy as np
import pandas as pd

# study biosamples (Sensititre) + their PDT target_acc
meta = pd.read_csv("data/narms/ecoli_amr_metadata.tsv", sep="\t",
                   usecols=["biosample_acc", "target_acc"],
                   low_memory=False, on_bad_lines="skip", quoting=3)
from amrcast.data.narms_features import build_narms_training_data
feats, _, _ = build_narms_training_data(
    "data/narms/antibiogram_mic.csv", "data/narms/ecoli_amr_metadata.tsv",
    platform_filter="ensitit", min_isolates_per_drug=100)
study = set(feats.index)

def base(pdt):  # strip version suffix: PDT000002365.3 -> PDT000002365
    return str(pdt).split(".")[0]

bios_to_pdt = {r.biosample_acc: base(r.target_acc)
               for r in meta.itertuples() if r.biosample_acc in study
               and isinstance(r.target_acc, str) and r.target_acc.startswith("PDT")}

# all_isolates: PDT base -> (has_cluster, min_dist_same, min_dist_opp)
tree = {}
with open("data/narms/clusters/ecoli_all_isolates.tsv") as fh:
    for r in csv.DictReader(fh, delimiter="\t"):
        pds = r.get("PDS_acc", "") or ""
        def num(x):
            try: return float(x)
            except: return np.nan
        tree[base(r["target_acc"])] = (pds.startswith("PDS"),
                                       num(r["min_dist_same"]), num(r["min_dist_opp"]))

clustered = singleton = not_in_tree = no_pdt = 0
singleton_dists = []
for acc in study:
    pdt = bios_to_pdt.get(acc)
    if pdt is None:
        no_pdt += 1; continue
    rec = tree.get(pdt)
    if rec is None:
        not_in_tree += 1
    elif rec[0]:
        clustered += 1
    else:
        singleton += 1
        d = np.nanmin([rec[1], rec[2]])
        if not np.isnan(d): singleton_dists.append(d)

n = len(study)
print("=" * 60)
print(f"study isolates                 : {n}")
print(f"  no PDT in metadata           : {no_pdt} ({no_pdt/n:.0%})")
print(f"  clustered (>=2, PDS)         : {clustered} ({clustered/n:.0%})")
print(f"  TRUE singleton (in tree, no cluster): {singleton} ({singleton/n:.0%})")
print(f"  NOT in SNP tree (data gap)   : {not_in_tree} ({not_in_tree/n:.0%})")
print("=" * 60)
if singleton_dists:
    a = np.array(singleton_dists)
    print("Nearest-neighbor SNP distance for TRUE singletons:")
    print(f"  n={len(a)}  median={np.median(a):.0f}  "
          f"25th={np.percentile(a,25):.0f}  75th={np.percentile(a,75):.0f}  max={a.max():.0f}")
    for thr in (50, 100, 250, 500, 1000):
        print(f"  nearest neighbor >{thr:>4} SNPs: {(a>thr).mean():.0%}")
    print("(If most are hundreds+ of SNPs away, they are genuinely diverse, not a data gap.)")
