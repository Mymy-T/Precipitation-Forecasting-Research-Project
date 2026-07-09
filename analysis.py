import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scikit_posthocs as sp
from scipy.stats import kruskal

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

# --- Omnibus test -----------------------------------------------------------------------------
# Kruskal-Wallis (rank-based) instead of parametric two-way ANOVA. At n=3-5 per group, Shapiro-Wilk /
# Levene pre-checks (as the original paper ran) have very little power to actually detect non-normality
# or unequal variance, so committing up front to a method that doesn't assume either is more honest than
# "test assumptions, then pick a method" at this sample size.
stat, p = kruskal(*group_arrays)
print(f"\nKruskal-Wallis: H = {stat:.4f}, p = {p:.5f} (n groups = {len(group_arrays)})")

# --- Post-hoc ----------------------------------------------------------------------------------
# Dunn's test with Holm correction replaces both Dunnett's test (control vs. each treatment cell) and
# Tukey's HSD (pairwise among treatment cells) from the original pipeline -- one rank-based procedure
# that handles the unequal group sizes here (control n=3, every other cell n=5) without difficulty.
long_df = pd.concat(
    [pd.DataFrame({"group": label, "F1": arr}) for label, arr in zip(group_labels, group_arrays)],
    ignore_index=True,
)
dunn_result = sp.posthoc_dunn(long_df, val_col="F1", group_col="group", p_adjust="holm")
print("\nDunn's test (Holm-corrected) p-value matrix:")
print(dunn_result.round(4))

print("\nControl vs. each cell (Dunnett-style contrasts), sorted by p-value:")
print(dunn_result["control"].drop("control").sort_values().round(4))


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
