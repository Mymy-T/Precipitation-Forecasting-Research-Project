import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scikit_posthocs as sp
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy.stats import levene, shapiro
from statsmodels.stats.multitest import multipletests

RESULTS_PATH = "results.csv"
BASELINE_RESULTS_PATH = "baseline_results.csv"
NUM_LOCATIONS = [4, 7, 10]
DISTS = [50, 75, 100, 125, 150]

df = pd.read_csv(os.getcwd() + "/" + RESULTS_PATH)

control_df = df[df["num_locations"] == 1]
treatment_df = df[df["num_locations"] != 1]

# Avg across seeds
draw_means = treatment_df.groupby(["num_locations", "dist", "draw_num"])["F1"].mean().reset_index()

# Control has only 1 draw, so its 3 replicates are its 3 seeds directly
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


# MARK: ANOVA
anova_model = smf.ols("F1 ~ C(num_locations)*C(dist)", data=draw_means).fit()

# Assumption checking for ANOVA table
cell_groups = [draw_means[ (draw_means["num_locations"] == n) & (draw_means["dist"] == d) ]["F1"].to_numpy() for n in NUM_LOCATIONS for d in DISTS]

levene_stat, levene_p = levene(*cell_groups, center="median")
shapiro_stat, shapiro_p = shapiro(anova_model.resid)

print("\nAssumption checks (treatment cells only):")
print(f"  Levene's test (homoscedasticity across 15 cells): stat={levene_stat:.4f}, p={levene_p:.4f}")
print(f"  Shapiro-Wilk (normality of ANOVA residuals):      stat={shapiro_stat:.4f}, p={shapiro_p:.4f}")

print("\nParametric two-way ANOVA (factorial omnibus, treatment cells only, draw-averaged F1):")
print(sm.stats.anova_lm(anova_model, typ=2).round(4))


# # Permutation-based two-way ANOVA; only if homoscedasticity or normality are violated
# def permutation_two_way_anova(data, n_perm=9999, seed=0):
#     rng = np.random.default_rng(seed)

#     def f_stats(f1_values):
#         model = smf.ols("F1 ~ C(num_locations)*C(dist)", data=data.assign(F1=f1_values)).fit()
#         return sm.stats.anova_lm(model, typ=2)["F"].drop("Residual")

#     observed_f = f_stats(data["F1"].to_numpy())
#     perm_f = pd.DataFrame([f_stats(rng.permutation(data["F1"].to_numpy())) for _ in range(n_perm)])
#     perm_p = ((perm_f >= observed_f).sum() + 1) / (n_perm + 1)
#     return pd.DataFrame({"F": observed_f, "perm_p": perm_p})


# print("\nPermutation two-way ANOVA (assumption-free robustness check, treatment cells only, draw-averaged F1):")
# print(permutation_two_way_anova(draw_means).round(4))


# MARK: Pairwise omparisons
long_df = pd.concat(
    [pd.DataFrame({"group": label, "F1": arr}) for label, arr in zip(group_labels, group_arrays)],
    ignore_index=True,
)

# Assumption checking to select Dunn's test (rank-based) or Dunnett's test (assumes normality)
group_levene_stat, group_levene_p = levene(*group_arrays, center="median")

group_model = smf.ols("F1 ~ C(group)", data=long_df).fit()
group_shapiro_stat, group_shapiro_p = shapiro(group_model.resid)

print("\nAssumption checks (control & 15 treatment cells):")
print(f"  Levene's test (homoscedasticity across 16 groups):    stat={group_levene_stat:.4f}, p={group_levene_p:.4f}")
print(f"  Shapiro-Wilk (normality of one-way ANOVA residuals):  stat={group_shapiro_stat:.4f}, p={group_shapiro_p:.4f}")

# Control comparisons
dunn_raw = sp.posthoc_dunn(long_df, val_col="F1", group_col="group", p_adjust=None)

control_raw_p = dunn_raw["control"].drop("control")
_, control_holm_p, _, _ = multipletests(control_raw_p.to_numpy(), method="holm")
control_comparison = pd.Series(control_holm_p, index=control_raw_p.index).sort_values()

print("\nControl vs. each cell (Dunnett-style, Holm-corrected over these 15 comparisons):")
print(control_comparison.round(4))


# Treatment comparisons
treatment_labels = [f"{n},{d}" for n in NUM_LOCATIONS for d in DISTS]
treatment_pairs = [(a, b) for i, a in enumerate(treatment_labels) for b in treatment_labels[i + 1 :]]
treatment_raw_p = [dunn_raw.loc[a, b] for a, b in treatment_pairs]
_, treatment_holm_p, _, _ = multipletests(treatment_raw_p, method="holm")
treatment_comparison = pd.Series(
    treatment_holm_p, index=[f"{a} vs {b}" for a, b in treatment_pairs]
).sort_values()
print("\nTreatment-vs-treatment pairwise comparisons (Holm-corrected over these 105 comparisons):")
print(treatment_comparison.round(4))


# MARK: Baseline comparison 
# This is descriptive only since fitting is deterministic
baseline_df = pd.read_csv(os.getcwd() + "/" + BASELINE_RESULTS_PATH)
baseline_control_f1 = baseline_df[baseline_df["num_locations"] == 1]["F1"].iloc[0]
baseline_cell_means = ( baseline_df[baseline_df["num_locations"] != 1].groupby(["num_locations", "dist"])["F1"].mean() )

print(f"\nBaseline (logistic regression) control F1 (deterministic, single value): {baseline_control_f1:.4f}")
print("LSTM vs. baseline mean F1 per cell (descriptive comparison, no significance test):")
for n in NUM_LOCATIONS:
    for d in DISTS:
        b_mean = baseline_cell_means[(n, d)]
        lstm_mean = np.mean(groups[f"{n},{d}"])
        beats = "beats" if lstm_mean > b_mean else "does not beat"
        print(f"  {n},{d}: logistic={b_mean:.4f}, LSTM={lstm_mean:.4f} ({beats} logistic)")


# MARK: Profile Plot
# Two hue families (one per model): blue for the LSTM, orange for the
LSTM_SHADES = {4: "#86b6ef", 7: "#2a78d6", 10: "#104281"}
LSTM_BASE = "#0d366b"
BASELINE_SHADES = {4: "#f6bba4", 7: "#eb6834", 10: "#b04e27"}
BASELINE_BASE = "#81391d"


def create_profile_plot():
    """Profile plot of mean F1 per num_locations across distance bounds, averaged over the 5 location draws"""
    _, ax = plt.subplots(nrows=1, figsize=(10, 8))
    ax.set_title("Profile Plot")

    for n in NUM_LOCATIONS:
        means = [np.mean(groups[f"{n},{d}"]) for d in DISTS]
        ax.plot(DISTS, means, marker="o", linestyle="--", color=LSTM_SHADES[n], label=f"{n} (LSTM)")

    for n in NUM_LOCATIONS:
        baseline_means = [baseline_cell_means[(n, d)] for d in DISTS]
        ax.plot(DISTS, baseline_means, marker="s", linestyle="--", color=BASELINE_SHADES[n], label=f"{n} (logistic)")

    ax.axhline(y=np.mean(groups["control"]), color=LSTM_BASE, linestyle=":", linewidth=2, label="control (1 location, LSTM)")
    ax.axhline(y=baseline_control_f1, color=BASELINE_BASE, linestyle=":", linewidth=2, label="control (1 location, logistic)")

    ax.set_xlabel("Distance Bounds (miles)")
    ax.set_ylabel("Mean F1 Score (averaged over 5 location draws)")
    ax.legend(title="Number of Locations")

create_profile_plot()
plt.show()
