"""Diagnostic: does genotype-grouped CV collapse to ~random CV?

If #unique genotype groups approaches #isolates, then GroupKFold behaves like
KFold and the "phylo-CV fix" tested nothing. Prints the numbers that settle it.
"""
from collections import Counter

from amrcast.data.narms_features import build_narms_training_data

DATA = "data/narms"

features, targets, isolate_groups = build_narms_training_data(
    mic_path=f"{DATA}/antibiogram_mic.csv",
    metadata_path=f"{DATA}/ecoli_amr_metadata.tsv",
    platform_filter="ensitit",
    min_isolates_per_drug=100,
)

n_isolates = len(features)
groups = [isolate_groups[a] for a in features.index]
n_groups = len(set(groups))
sizes = Counter(groups)
singletons = sum(1 for g, c in sizes.items() if c == 1)
biggest = sorted(sizes.values(), reverse=True)[:5]

print("=" * 60)
print(f"isolates (Sensititre)      : {n_isolates}")
print(f"unique genotype groups     : {n_groups}")
print(f"groups / isolates ratio    : {n_groups / n_isolates:.3f}")
print(f"singleton groups (size 1)  : {singletons} ({singletons / n_groups:.0%} of groups)")
print(f"isolates in singletons     : {singletons} ({singletons / n_isolates:.0%} of isolates)")
print(f"5 largest group sizes      : {biggest}")
print("=" * 60)
print("If ratio ~1.0 and most isolates are singletons, GroupKFold == KFold")
print("=> the genotype grouping tested nothing; real phylo-CV still undone.")
