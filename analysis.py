import math
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import re
from scipy.stats import shapiro, levene, dunnett, tukey_hsd
import statsmodels.api as sm
from statsmodels.formula.api import ols, rlm


def create_residual_vs_fit_graph(groups):
    """Residual vs fit plot - graphically check for homogeneity of variance"""
    _, ax = plt.subplots(nrows=1, figsize=(10, 8))
    # Residual vs Fit - Homogeneity of variance
    means = [] # There will be duplicate values, 10 for each group
    residuals = []
    mean_residuals = []
    ranges = [] # IQRs of residuals, not means
    for group in groups:
        mean = np.mean(group)
        current_residuals = []
        for x in group:
            means.append(mean)
            current_residuals.append(x - mean)
        ql, qh = np.percentile(current_residuals, [20, 80])
        range = qh - ql
        ranges.append(range)
        mean_residuals.append(np.median(current_residuals))
        for curr_res in current_residuals:
            residuals.append(curr_res)
    unique_means = list(set(means))
    ranges = [x/2 for x in ranges] # Halve each iqr for plotting
    ax.set_title("Residual vs Fitted Values")
    ax.scatter(means, residuals, color="blue")
    ax.errorbar(unique_means, mean_residuals, yerr=ranges, fmt="o", ecolor="red", color="red", capsize=0)
    ax.axhline(y=0, color="green", linestyle="--")
    ax.set_xlabel("Fitted Values (Group Means)")
    ax.set_ylabel("Residuals")

def create_profile_plot(num_locs, dists, results):
    """Num_locs distinguished by color, dists distinguished by x axis"""
    c_label = "Number of Locations"
    x_label = "Distance Bounds (miles)"
    c_vars = num_locs
    x_vars = dists
    # num_locations_pattern = r"^\d"
    # distance_pattern = r"\d+$"
    # nums_locations = []
    # dists = []
    # for key, val in results.items():
    #     num_locations = int(re.search(num_locations_pattern, key).group())
    #     dist = int(re.search(distance_pattern, key).group())
    #     nums_locations.append(num_locations)
    #     dists.append(dist)


    fig, ax = plt.subplots(nrows=1, figsize=(10, 8))
    ax.set_title("Profile Plot")
    for c_var in c_vars:
        means = []
        for x_var in x_vars:
            key = f"{c_var},{x_var}"
            mean = np.mean(results[key])
            means.append(mean)
        ax.plot(x_vars, means, marker="o", linestyle="--", label=f"{c_var}")
    ax.set_xlabel(x_label)
    ax.set_ylabel("Mean F1 Score")
    ax.legend(title=c_label)

df = pd.read_csv(os.getcwd() + "/results.csv")
# df = df[df["Num_Loc"] != 1]


### ANOVA Analysis Setup

num_locs = [4, 7, 10]
dists = [50, 75, 100, 125, 150]
results = {}
for num_loc in num_locs:
    for dist in dists:
        df_subset = df[df["Num_Loc"] == num_loc]
        df_subset = df_subset[df_subset["Dist"] == dist]
        df_subset = df_subset["F1"]
        results[f"{num_loc},{dist}"] = df_subset.to_numpy()
groups = list(results.values())

# Check Assumptions
# Homogeneity of variance
levene_p = levene(*groups).pvalue
print(f"Levene's test resulted in a p value of {levene_p}")
create_residual_vs_fit_graph(groups)
# Normality
shapiro_p_vals = [shapiro(group).pvalue for group in groups]
non_normal_count = 0
for p in shapiro_p_vals:
    if p < 0.05:
        non_normal_count += 1
print(f"Shapiro-Wilks test resulted in a p value < 0.05 for {non_normal_count} out of {len(shapiro_p_vals)} groups")

# Dunnett's test against control
control_subset = df[df["Num_Loc"] == 1]
control = control_subset["F1"].to_numpy()
print(dunnett(*groups, control=control))

# Create ANOVA Table
df = df[df["Num_Loc"] != 1] # Drop the control group
lm = ols('F1 ~ C(Num_Loc)*C(Dist)', data=df).fit()
table = sm.stats.anova_lm(lm)
print(table)
create_profile_plot(num_locs, dists, results)

# Tukey's HSD Analysis
print(tukey_hsd(*groups))
for i in range(len(groups)):
    print(f"Group #{i}: {np.mean(groups[i])}")


plt.show()