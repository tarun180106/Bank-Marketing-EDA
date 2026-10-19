
import os
import pandas as pd
import numpy as np

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import seaborn as sns

# --------------------------------------------------
# 1. LOAD DATASET
# --------------------------------------------------

df = pd.read_csv("data/bank-full.csv", sep=";")
os.makedirs("outputs", exist_ok=True)
sns.set_theme(style="whitegrid")

print("=" * 60)
print("ASSESSMENT 4: BIVARIATE ANALYSIS")
print("=" * 60)

print("\nDataset shape:", df.shape)
print("\nFirst 5 rows:")
print(df.head())


def save_graph(filename):
    plt.tight_layout()
    plt.savefig(
        os.path.join("outputs", filename),
        dpi=150,
        bbox_inches="tight"
    )
    plt.close()
    print("Graph saved:", filename)


# --------------------------------------------------
# 2. CONTINGENCY TABLE: TWO CATEGORICAL VARIABLES
# --------------------------------------------------

print("\n--- Contingency Table: Job vs Subscription ---")

contingency = pd.crosstab(df["job"], df["y"])
print(contingency)

# --------------------------------------------------
# 3. STACKED BAR CHART
# --------------------------------------------------

contingency.plot(kind="bar", stacked=True, figsize=(10, 6))
plt.title("Subscription by Job (Stacked Bar Chart)")
plt.xlabel("Job")
plt.ylabel("Number of Customers")
plt.xticks(rotation=45, ha="right")
save_graph("qp4_stacked_bar.png")

# --------------------------------------------------
# 4. GROUPED BAR CHART
# --------------------------------------------------

contingency.plot(kind="bar", stacked=False, figsize=(10, 6))
plt.title("Subscription by Job (Grouped Bar Chart)")
plt.xlabel("Job")
plt.ylabel("Number of Customers")
plt.xticks(rotation=45, ha="right")
save_graph("qp4_grouped_bar.png")

# --------------------------------------------------
# 5. SCATTER PLOT: TWO NUMERICAL VARIABLES
# --------------------------------------------------

plt.figure(figsize=(8, 5))
sns.scatterplot(
    data=df.sample(min(2000, len(df)), random_state=42),
    x="age",
    y="duration",
    hue="y",
    alpha=0.6
)
plt.title("Age vs Call Duration")
plt.xlabel("Age")
plt.ylabel("Call Duration (seconds)")
save_graph("qp4_scatter_age_duration.png")

# --------------------------------------------------
# 6. LINE PLOT
# --------------------------------------------------

# Mean call duration by age
mean_duration = df.groupby("age", observed=True)["duration"].mean()

plt.figure(figsize=(9, 5))
plt.plot(mean_duration.index, mean_duration.values, marker="o")
plt.title("Average Call Duration by Age")
plt.xlabel("Age")
plt.ylabel("Average Call Duration (seconds)")
save_graph("qp4_line_age_duration.png")

# --------------------------------------------------
# 7. HEATMAP: CONTINGENCY TABLE
# --------------------------------------------------

plt.figure(figsize=(9, 6))
sns.heatmap(contingency, annot=True, fmt="d", cmap="YlGnBu")
plt.title("Heatmap of Job vs Subscription")
plt.xlabel("Subscribed")
plt.ylabel("Job")
save_graph("qp4_contingency_heatmap.png")

# --------------------------------------------------
# 8. BAR CHART COMPARING NUMERICAL VARIABLES
# --------------------------------------------------

mean_values = df[["age", "duration", "campaign"]].mean()

plt.figure(figsize=(8, 5))
mean_values.plot(kind="bar")
plt.title("Mean of Selected Numerical Variables")
plt.xlabel("Variable")
plt.ylabel("Mean Value")
save_graph("qp4_numeric_means_bar.png")

# --------------------------------------------------
# 9. DENSITY PLOT BY SUBSCRIPTION
# --------------------------------------------------

plt.figure(figsize=(8, 5))
sns.kdeplot(
    data=df,
    x="age",
    hue="y",
    common_norm=False
)
plt.title("Age Density by Subscription")
plt.xlabel("Age")
plt.ylabel("Density")
save_graph("qp4_age_density_by_subscription.png")

# --------------------------------------------------
# 10. BOXPLOT
# --------------------------------------------------

plt.figure(figsize=(8, 5))
sns.boxplot(data=df, x="y", y="duration")
plt.title("Call Duration by Subscription")
plt.xlabel("Subscribed")
plt.ylabel("Call Duration (seconds)")
save_graph("qp4_duration_boxplot.png")

# --------------------------------------------------
# 11. VIOLIN PLOT
# --------------------------------------------------

plt.figure(figsize=(8, 5))
sns.violinplot(data=df, x="y", y="age")
plt.title("Age Distribution by Subscription")
plt.xlabel("Subscribed")
plt.ylabel("Age")
save_graph("qp4_age_violinplot.png")

# --------------------------------------------------
# 12. COMBINED VIOLIN AND BOX PLOT
# --------------------------------------------------

plt.figure(figsize=(8, 5))
sns.violinplot(
    data=df,
    x="y",
    y="age",
    inner=None,
    color="lightblue"
)
sns.boxplot(
    data=df,
    x="y",
    y="age",
    width=0.18,
    showfliers=False
)
plt.title("Combined Violin and Box Plot of Age")
plt.xlabel("Subscribed")
plt.ylabel("Age")
save_graph("qp4_combined_violin_boxplot.png")

# --------------------------------------------------
# 13. CORRELATION MATRIX
# --------------------------------------------------

numeric_df = df.select_dtypes(include=np.number)
correlation = numeric_df.corr()

print("\n--- Correlation Matrix ---")
print(correlation.round(3))

plt.figure(figsize=(12, 9))
sns.heatmap(
    correlation,
    annot=True,
    fmt=".2f",
    cmap="coolwarm",
    center=0
)
plt.title("Correlation Matrix of Numerical Variables")
save_graph("qp4_correlation_matrix.png")

# --------------------------------------------------
# 14. FINAL SUMMARY
# --------------------------------------------------

print("\nAssessment 4 completed successfully.")
print("Check the outputs folder for the generated graphs.")
