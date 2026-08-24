import os
from itertools import combinations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy.stats import dunnett, levene, shapiro
from statsmodels.stats.multicomp import pairwise_tukeyhsd

RESULTS_PATH = "results.csv"
BASELINE_RESULTS_PATH = "baseline_results.csv"
REPORT_PATH = "analysis_report.md"
PROFILE_PLOT_PATH = "profile_plot.png"
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


# MARK: Pairwise comparisons
long_df = pd.concat([pd.DataFrame({"group": label, "F1": arr}) for label, arr in zip(group_labels, group_arrays)], ignore_index=True)

# Assumption check for Dunnett/Tukey
group_levene_stat, group_levene_p = levene(*group_arrays, center="median")

group_model = smf.ols("F1 ~ C(group)", data=long_df).fit()
group_shapiro_stat, group_shapiro_p = shapiro(group_model.resid)

print("\nAssumption checks (control & 15 treatment cells):")
print(f"  Levene's test (homoscedasticity across 16 groups):    stat={group_levene_stat:.4f}, p={group_levene_p:.4f}")
print(f"  Shapiro-Wilk (normality of one-way ANOVA residuals):  stat={group_shapiro_stat:.4f}, p={group_shapiro_p:.4f}")

# Control comparisons (Dunnett)
dunnett_result = dunnett(*group_arrays[1:], control=groups["control"])
control_comparison = pd.Series(dunnett_result.pvalue, index=group_labels[1:]).sort_values()

print("\nControl vs. each cell (Dunnett's test):")
print(control_comparison.round(4))


# Treatment comparisons (Tukey HSD)
treatment_long_df = long_df[long_df["group"] != "control"]
tukey_result = pairwise_tukeyhsd(treatment_long_df["F1"], treatment_long_df["group"])
pair_labels = [f"{a} vs {b}" for a, b in combinations(tukey_result.groupsunique, 2)]
treatment_comparison = pd.Series(tukey_result.pvalues, index=pair_labels).sort_values()
print("\nTreatment-vs-treatment pairwise comparisons (Tukey's HSD):")
print(treatment_comparison.round(4))


# MARK: Comparison to baseline
baseline_df = pd.read_csv(os.getcwd() + "/" + BASELINE_RESULTS_PATH)
baseline_control_f1 = baseline_df[baseline_df["num_locations"] == 1]["F1"].iloc[0]
baseline_treatment_df = baseline_df[baseline_df["num_locations"] != 1]
baseline_cell_means = baseline_treatment_df.groupby(["num_locations", "dist"])["F1"].mean()

print(f"\nBaseline (logistic regression) control F1 (deterministic, single value): {baseline_control_f1:.4f}")
print("LSTM vs. baseline mean F1 per cell (descriptive comparison, no significance test):")
for n in NUM_LOCATIONS:
    for d in DISTS:
        b_mean = baseline_cell_means[(n, d)]
        lstm_mean = np.mean(groups[f"{n},{d}"])
        beats = "beats" if lstm_mean > b_mean else "does not beat"
        print(f"  {n},{d}: logistic={b_mean:.4f}, LSTM={lstm_mean:.4f} ({beats} logistic)")


# MARK: Baseline ANOVA
baseline_anova_model = smf.ols("F1 ~ C(num_locations)*C(dist)", data=baseline_treatment_df).fit()

baseline_cell_groups = [
    baseline_treatment_df[(baseline_treatment_df["num_locations"] == n) & (baseline_treatment_df["dist"] == d)]["F1"].to_numpy()
    for n in NUM_LOCATIONS
    for d in DISTS
]
baseline_levene_stat, baseline_levene_p = levene(*baseline_cell_groups, center="median")
baseline_shapiro_stat, baseline_shapiro_p = shapiro(baseline_anova_model.resid)

print("\nAssumption checks (baseline treatment cells only):")
print(f"  Levene's test (homoscedasticity across 15 cells): stat={baseline_levene_stat:.4f}, p={baseline_levene_p:.4f}")
print(f"  Shapiro-Wilk (normality of ANOVA residuals):      stat={baseline_shapiro_stat:.4f}, p={baseline_shapiro_p:.4f}")

print("\nParametric two-way ANOVA (baseline, treatment cells only):")
print(sm.stats.anova_lm(baseline_anova_model, typ=2).round(4))


# MARK: Combined ANOVA
combined_df = pd.concat(
    [
        draw_means.assign(model="LSTM"),
        baseline_treatment_df[["num_locations", "dist", "draw_num", "F1"]].assign(model="baseline"),
    ],
    ignore_index=True,
)
combined_model = smf.ols("F1 ~ (C(model) + C(num_locations) + C(dist)) ** 2", data=combined_df).fit()

combined_groups = [
    combined_df[(combined_df["model"] == m) & (combined_df["num_locations"] == n) & (combined_df["dist"] == d)]["F1"].to_numpy()
    for m in ["LSTM", "baseline"]
    for n in NUM_LOCATIONS
    for d in DISTS
]

# Assumption checks
combined_levene_stat, combined_levene_p = levene(*combined_groups, center="median")
combined_shapiro_stat, combined_shapiro_p = shapiro(combined_model.resid)

print("\nAssumption checks (combined LSTM & baseline, treatment cells only):")
print(f"  Levene's test (homoscedasticity across model & cell groups): stat={combined_levene_stat:.4f}, p={combined_levene_p:.4f}")
print(f"  Shapiro-Wilk (normality of ANOVA residuals):                 stat={combined_shapiro_stat:.4f}, p={combined_shapiro_p:.4f}")

print("\nCombined LSTM-vs-baseline two-way ANOVA")
print(sm.stats.anova_lm(combined_model, typ=2).round(4))


# MARK: Profile Plot
# Two hue families (one per model): blue for the LSTM, orange for the baseline/logistic model
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
plt.savefig(os.getcwd() + "/" + PROFILE_PLOT_PATH, dpi=150, bbox_inches="tight")
plt.show()


# MARK: Markdown report export
def export_report_to_markdown(filepath=REPORT_PATH, plot_path=PROFILE_PLOT_PATH):
    """Write all computed results to a formatted Markdown report, mirroring the console output above."""
    lines = ["# Precipitation LSTM Analysis Report", ""]

    lines += ["## Per-Cell Summary", ""]
    summary_df = pd.DataFrame(
        [
            {
                "group": label,
                "n": len(arr),
                "mean": round(np.mean(arr), 4),
                "sd": round(np.std(arr, ddof=1), 4) if len(arr) > 1 else float("nan"),
            }
            for label, arr in zip(group_labels, group_arrays)
        ]
    )
    lines += [summary_df.to_markdown(index=False), ""]

    lines += ["## LSTM: Parametric Two-Way ANOVA (treatment cells only, draw-averaged F1)", ""]
    lines += [
        "**Assumption checks (treatment cells only):**",
        f"- Levene's test (homoscedasticity across 15 cells): stat={levene_stat:.4f}, p={levene_p:.4f}",
        f"- Shapiro-Wilk (normality of ANOVA residuals): stat={shapiro_stat:.4f}, p={shapiro_p:.4f}",
        "",
    ]
    lstm_anova_table = sm.stats.anova_lm(anova_model, typ=2).round(4)
    lstm_anova_table.index.name = "Term"
    lines += [lstm_anova_table.to_markdown(), ""]

    lines += ["## Assumption Checks (control & 15 treatment cells)", ""]
    lines += [
        f"- Levene's test (homoscedasticity across 16 groups): stat={group_levene_stat:.4f}, p={group_levene_p:.4f}",
        f"- Shapiro-Wilk (normality of one-way ANOVA residuals): stat={group_shapiro_stat:.4f}, p={group_shapiro_p:.4f}",
        "",
    ]

    lines += ["## Control vs. Each Cell (Dunnett's Test)", ""]
    control_table = control_comparison.rename("p-value").round(4)
    control_table.index.name = "group"
    lines += [control_table.to_markdown(), ""]

    lines += ["## Treatment-vs-Treatment Pairwise Comparisons (Tukey's HSD)", ""]
    treatment_table = treatment_comparison.rename("p-value").round(4)
    treatment_table.index.name = "comparison"
    lines += [treatment_table.to_markdown(), ""]

    lines += ["## Baseline (Logistic Regression) Comparison", ""]
    lines += [f"Baseline control F1 (deterministic, single value): **{baseline_control_f1:.4f}**", ""]
    baseline_compare_df = pd.DataFrame(
        [
            {
                "num_locations": n,
                "dist": d,
                "logistic": round(baseline_cell_means[(n, d)], 4),
                "LSTM": round(np.mean(groups[f"{n},{d}"]), 4),
                "LSTM beats logistic": np.mean(groups[f"{n},{d}"]) > baseline_cell_means[(n, d)],
            }
            for n in NUM_LOCATIONS
            for d in DISTS
        ]
    )
    lines += [baseline_compare_df.to_markdown(index=False), ""]

    lines += ["## Baseline: Parametric Two-Way ANOVA (treatment cells only)", ""]
    lines += [
        f"- Levene's test (homoscedasticity across 15 cells): stat={baseline_levene_stat:.4f}, p={baseline_levene_p:.4f}",
        f"- Shapiro-Wilk (normality of ANOVA residuals): stat={baseline_shapiro_stat:.4f}, p={baseline_shapiro_p:.4f}",
        "",
    ]
    baseline_anova_table = sm.stats.anova_lm(baseline_anova_model, typ=2).round(4)
    baseline_anova_table.index.name = "Term"
    lines += [baseline_anova_table.to_markdown(), ""]

    lines += ["## Combined LSTM-vs-Baseline Two-Way ANOVA", ""]
    lines += [
        f"- Levene's test (homoscedasticity across model x cell groups): stat={combined_levene_stat:.4f}, p={combined_levene_p:.4f}",
        f"- Shapiro-Wilk (normality of ANOVA residuals): stat={combined_shapiro_stat:.4f}, p={combined_shapiro_p:.4f}",
        "",
    ]
    combined_anova_table = sm.stats.anova_lm(combined_model, typ=2).round(4)
    combined_anova_table.index.name = "Term"
    lines += [combined_anova_table.to_markdown(), ""]

    lines += ["## Profile Plot", "", f"![Profile Plot]({plot_path})", ""]

    with open(os.getcwd() + "/" + filepath, "w") as f:
        f.write("\n".join(lines))
    print(f"\nMarkdown report written to {filepath}")


export_report_to_markdown()
