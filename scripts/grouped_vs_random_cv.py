"""Does genotype-grouped CV actually change EA vs random KFold?

Trains each drug twice (groups on / off) and prints the EA delta. This is the
cheap precursor to real phylo-CV: if even genotype-grouping drops EA, leakage is
already biting; if it barely moves, the honest test is still Mash-based clustering.
No genome downloads required.
"""
import numpy as np

from amrcast.data.narms_features import build_narms_training_data
from amrcast.ml.xgboost_model import MICPredictor

DRUGS = ["ampicillin", "ciprofloxacin", "gentamicin", "tetracycline",
         "ceftriaxone", "meropenem", "cefepime", "cefotaxime"]

features, targets, isolate_groups = build_narms_training_data(
    mic_path="data/narms/antibiogram_mic.csv",
    metadata_path="data/narms/ecoli_amr_metadata.tsv",
    platform_filter="ensitit",
    min_isolates_per_drug=100,
)

print(f"\n{'drug':<15}{'n':>7}{'EA random':>12}{'EA grouped':>12}{'delta':>9}")
print("-" * 55)
for ab in DRUGS:
    ab_t = targets[targets["antibiotic"] == ab]
    valid = [a for a in ab_t["biosample_acc"] if a in features.index]
    if len(valid) < 50:
        continue
    X = features.loc[valid].values
    y = ab_t.set_index("biosample_acc").loc[valid]["log2_mic"].values
    groups = np.array([isolate_groups[a] for a in valid])

    ea = {}
    for label, g in (("random", None), ("grouped", groups)):
        cv = MICPredictor(antibiotic=ab).cross_validate(
            X, y, feature_names=list(features.columns), n_folds=5, groups=g)
        ea[label] = cv["essential_agreement_mean"]
    print(f"{ab:<15}{len(valid):>7}{ea['random']:>11.1%}{ea['grouped']:>12.1%}"
          f"{(ea['grouped'] - ea['random']):>+9.1%}")
