import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scikit_posthocs as sp
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.stats.multitest import multipletests

RESULTS_PATH = "redo_results.csv"
NUM_LOCATIONS = [4, 7, 10]
DISTS = [50, 75, 100, 125, 150]

df = pd.read_csv(os.getcwd() + "/" + RESULTS_PATH)

control_df = df[df["num_locations"] == 1]
treatment_df = df[df["num_locations"] != 1]

# Aggregate seed (training-noise) replicates within each independent location draw, leaving exactly one
# F1 per (num_locations, dist, draw_id) -- the true independent unit of replication in this redesign.
draw_means = (
    treatment_df.groupby(["num_locations", "dist", "draw_id"])["F1"]
    .mean()
    .reset_index()
)

# Control has no location-draw axis at all (there's only one possible "1-location" dataset: the origin
# point itself), so its 3 replicates are its 3 seeds directly, left unaggregated. This is an intentional
# asymmetry: control's spread reflects training-noise only, while every other cell's spread reflects
# genuine location-sampling variance (seed noise has already been averaged out of it).
group_labels = ["control"] + [f"{n},{d}" for n in NUM_LOCATIONS for d in DISTS]
groups = {"control": control_df["F1"].to_numpy()}
for n in NUM_LOCATIONS:
    for d in DISTS:
        subset = draw_means[(draw_means["num_locations"] == n) & (draw_means["dist"] == d)]
        groups[f"{n},{d}"] = subset["F1"].to_numpy()

group_arrays = [groups[label] for label in group_labels]

print("Per-cell summary (n, mean, sd):")
for label, arr in zip(group_labels, group_arrays):
    sd = np.std(arr, ddof=1) if len(arr) > 1 else float("nan")
    print(f"  {label:>10}: n={len(arr)}, mean={np.mean(arr):.4f}, sd={sd:.4f}")


# --- Factorial omnibus test -------------------------------------------------------------------
# Permutation-based two-way ANOVA: refits F1 ~ C(num_locations)*C(dist) on label-shuffled draw means
# to build an empirical null for each term's F-statistic. This preserves the same main-effect/
# interaction decomposition as the original parametric two-way ANOVA (so "significant interaction
# between num_locations and dist" can still be reported), but the p-values come from a permutation
# null rather than the F-distribution, so they don't need normality or homoscedasticity -- the two
# assumptions the original paper's own Levene/Shapiro-Wilk checks found reason to doubt. Control is
# excluded here: it has no dist level (a single fixed origin point), so it's a degenerate cell for a
# crossed 2-way design and is handled separately below instead.
def permutation_two_way_anova(data, n_perm=9999, seed=0):
    rng = np.random.default_rng(seed)

    def f_stats(f1_values):
        model = smf.ols("F1 ~ C(num_locations)*C(dist)", data=data.assign(F1=f1_values)).fit()
        return sm.stats.anova_lm(model, typ=2)["F"].drop("Residual")

    observed_f = f_stats(data["F1"].to_numpy())
    perm_f = pd.DataFrame([f_stats(rng.permutation(data["F1"].to_numpy())) for _ in range(n_perm)])
    perm_p = ((perm_f >= observed_f).sum() + 1) / (n_perm + 1)
    return pd.DataFrame({"F": observed_f, "perm_p": perm_p})


print("\nPermutation two-way ANOVA (factorial omnibus, treatment cells only, draw-averaged F1):")
print(permutation_two_way_anova(draw_means).round(4))


# --- Control comparison ------------------------------------------------------------------------
# Dunn's test (rank-based, computed across all 16 groups at once) gives more stable p-values than
# running 15 separate two-sample tests would, since its variance term pools ranks across every group.
# But the Dunnett-style question here -- "does control differ from each treatment cell" -- is only 15
# comparisons of actual interest. Correcting across the full 16-group pairwise matrix (120 comparisons,
# as the previous version of this script did) needlessly penalizes those 15 for the other 105
# treatment-vs-treatment comparisons nobody is asking about here, so Holm's correction is instead
# recomputed on just the relevant subset for each question.
long_df = pd.concat(
    [pd.DataFrame({"group": label, "F1": arr}) for label, arr in zip(group_labels, group_arrays)],
    ignore_index=True,
)
dunn_raw = sp.posthoc_dunn(long_df, val_col="F1", group_col="group", p_adjust=None)

control_raw_p = dunn_raw["control"].drop("control")
_, control_holm_p, _, _ = multipletests(control_raw_p.to_numpy(), method="holm")
control_comparison = pd.Series(control_holm_p, index=control_raw_p.index).sort_values()
print("\nControl vs. each cell (Dunnett-style, Holm-corrected over these 15 comparisons only):")
print(control_comparison.round(4))

treatment_labels = [f"{n},{d}" for n in NUM_LOCATIONS for d in DISTS]
treatment_pairs = [(a, b) for i, a in enumerate(treatment_labels) for b in treatment_labels[i + 1 :]]
treatment_raw_p = [dunn_raw.loc[a, b] for a, b in treatment_pairs]
_, treatment_holm_p, _, _ = multipletests(treatment_raw_p, method="holm")
treatment_comparison = pd.Series(
    treatment_holm_p, index=[f"{a} vs {b}" for a, b in treatment_pairs]
).sort_values()
print("\nTreatment-vs-treatment pairwise comparisons (Holm-corrected over these 105 comparisons only):")
print(treatment_comparison.round(4))


def create_profile_plot():
    """Mean F1 per num_locations across distance bounds, averaged over the 5 independent location draws."""
    _, ax = plt.subplots(nrows=1, figsize=(10, 8))
    ax.set_title("Profile Plot (redesigned experiment)")
    for n in NUM_LOCATIONS:
        means = [np.mean(groups[f"{n},{d}"]) for d in DISTS]
        ax.plot(DISTS, means, marker="o", linestyle="--", label=str(n))
    ax.axhline(y=np.mean(groups["control"]), color="grey", linestyle=":", label="control (1 location)")
    ax.set_xlabel("Distance Bounds (miles)")
    ax.set_ylabel("Mean F1 Score (averaged over 5 independent location draws)")
    ax.legend(title="Number of Locations")


create_profile_plot()
plt.show()
