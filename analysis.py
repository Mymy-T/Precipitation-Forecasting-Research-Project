import math
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
from scipy.stats import shapiro, levene, dunnett, tukey_hsd
import statsmodels.api as sm
from statsmodels.formula.api import ols, rlm
# from statsmodels.stats.outliers_influence import variance_inflation_factor

df = pd.read_csv(os.getcwd() + "/results.csv")
# df = df[df["Num_Loc"] != 1]


### ANOVA Analysis

num_locs = [4, 7, 10]
dists = [50, 75, 100, 125, 150]
groups = []
for num_loc in num_locs:
    for dist in dists:
        df_subset = df[df["Num_Loc"] == num_loc]
        df_subset = df_subset[df_subset["Dist"] == dist]
        df_subset = df_subset["F1"]
        groups.append(df_subset.to_numpy())

# Check Assumptions
# Homogeneity of variance
levene_p = levene(*groups).pvalue
print(f"Levene's test resulted in a p value of {levene_p}")
# Normality
shapiro_p_vals = [shapiro(group).pvalue for group in groups]
non_normal_count = 0
for p in shapiro_p_vals:
    if p < 0.05:
        non_normal_count += 1
print(f"Shapiro-Wilks test resulted in a p value < 0.05 for {non_normal_count} out of {len(shapiro_p_vals)} groups")

# Plots for checking assumptions
fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(10, 8))
# Residual vs Fit - Homogeneity of variance
means = []
residuals = []
for group in groups:
    mean = np.mean(group)
    for x in group:
        means.append(mean)
        residuals.append(x - mean)
axes[0].set_title("Residual vs Fitted Values")
axes[0].scatter(means, residuals, color="blue")
axes[0].axhline(y=0, color="red", linestyle="--")
axes[0].set_xlabel("Fitted Values")
axes[0].set_ylabel("Residuals")

# Dunnett's test against control
control_subset = df[df["Num_Loc"] == 1]
control = control_subset["F1"].to_numpy()
print(dunnett(*groups, control=control))

# Create ANOVA Table
df = df[df["Num_Loc"] != 1] # Drop the control group
lm = ols('F1 ~ C(Num_Loc)*C(Dist)', data=df).fit()
table = sm.stats.anova_lm(lm)
print(table)

# Tukey's HSD Analysis
print(tukey_hsd(*groups))
for i in range(len(groups)):
    print(f"Group #{i}: {np.mean(groups[i])}")


plt.show()

### REGRESSION ANALYSIS

# # # Transformation of y (F1)
# # df["F1"] = df["F1"].apply(lambda x: math.sqrt(x))

# # model = ols('F1 ~ Num_Loc*Dist', data=df).fit()
# model = rlm('F1 ~ Num_Loc*Dist', data=df).fit()
# print(model.summary())
# residuals = model.resid
# fit_values = model.fittedvalues

# # Shapiro Wilk test
# print(shapiro(residuals))

# # VIF
# X = sm.tools.tools.add_constant(df)
# vif = pd.DataFrame()
# vif["Feature"] = X.columns
# vif['VIF'] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])] 
# print(vif)

# print(residuals)

# fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(12, 8))

# # Residual fit plot
# axes[0].set_title("Residual vs Fitted Values")
# axes[0].scatter(fit_values, residuals, color="blue")
# axes[0].axhline(y=0, color="red", linestyle="--")
# axes[0].set_xlabel("Fitted Values")
# axes[0].set_ylabel("Residuals")

# # QN plot
# axes[1].set_title("QN Plot")
# sm.qqplot(residuals, fit=True, line='45', ax=axes[1])
# axes[1].set_xlabel("Expected Normal")
# axes[1].set_ylabel("Residuals")


# # Histogram
# axes[2].set_title("Histogram of Residuals")
# axes[2].hist(residuals, bins=30)
# axes[2].set_xlabel("Residual")
# axes[2].set_ylabel("Frequency")

# fig.tight_layout()
# plt.show()
