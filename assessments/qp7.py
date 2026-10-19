
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
print("ASSESSMENT 7: 1D, 2D AND MULTIDIMENSIONAL ANALYSIS")
print("=" * 60)

print("\nDataset shape:", df.shape)


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
# 2. ONE-DIMENSIONAL ANALYSIS: DESCRIPTIVE STATISTICS
# --------------------------------------------------

print("\n--- One-Dimensional Descriptive Statistics ---")

numeric_df = df.select_dtypes(include=np.number)

stats = pd.DataFrame({
    "Mean": numeric_df.mean(),
    "Median": numeric_df.median(),
    "Minimum": numeric_df.min(),
    "Maximum": numeric_df.max(),
    "Range": numeric_df.max() - numeric_df.min(),
    "Variance": numeric_df.var(),
    "Standard Deviation": numeric_df.std(),
    "IQR": numeric_df.quantile(0.75) - numeric_df.quantile(0.25),
    "Skewness": numeric_df.skew(),
    "Kurtosis": numeric_df.kurtosis()
})

print(stats.round(3))


# --------------------------------------------------
# 3. FREQUENCY AND RELATIVE FREQUENCY TABLE
# --------------------------------------------------

print("\n--- Frequency Table: Job ---")

job_frequency = df["job"].value_counts()
job_relative_frequency = df["job"].value_counts(normalize=True)

job_table = pd.DataFrame({
    "Frequency": job_frequency,
    "Relative Frequency": job_relative_frequency,
    "Percentage": job_relative_frequency * 100
})

print(job_table.round(3))

# Bar chart
plt.figure(figsize=(10, 6))
job_frequency.sort_values().plot(kind="barh")
plt.title("Frequency of Customers by Job")
plt.xlabel("Frequency")
plt.ylabel("Job")
save_graph("qp7_job_frequency.png")

# Pie chart
plt.figure(figsize=(8, 8))
job_frequency.plot(kind="pie", autopct="%1.1f%%")
plt.title("Proportion of Customers by Job")
plt.ylabel("")
save_graph("qp7_job_pie.png")


# --------------------------------------------------
# 4. NUMERICAL FREQUENCY TABLE USING BINS
# --------------------------------------------------

print("\n--- Age Frequency Table ---")

df["age_group"] = pd.cut(
    df["age"],
    bins=[0, 25, 35, 45, 55, 65, 100],
    labels=["0-25", "26-35", "36-45", "46-55", "56-65", "66+"]
)

age_frequency = df["age_group"].value_counts().sort_index()
print(age_frequency)

plt.figure(figsize=(8, 5))
age_frequency.plot(kind="bar")
plt.title("Customer Age Group Frequency")
plt.xlabel("Age Group")
plt.ylabel("Frequency")
plt.xticks(rotation=0)
save_graph("qp7_age_group_frequency.png")

# Histogram
plt.figure(figsize=(8, 5))
sns.histplot(data=df, x="age", bins=20, kde=True)
plt.title("Histogram of Customer Age")
plt.xlabel("Age")
plt.ylabel("Frequency")
save_graph("qp7_age_histogram.png")

# Boxplot
plt.figure(figsize=(8, 5))
sns.boxplot(data=df, x="age")
plt.title("Boxplot of Customer Age")
plt.xlabel("Age")
save_graph("qp7_age_boxplot.png")


# --------------------------------------------------
# 5. TWO-DIMENSIONAL ANALYSIS: CROSS-TABULATION
# --------------------------------------------------

print("\n--- Cross-tabulation: Job vs Subscription ---")

cross_table = pd.crosstab(df["job"], df["y"])
print(cross_table)

print("\nRow proportions:")
print(pd.crosstab(
    df["job"], df["y"], normalize="index"
).round(3))

print("\nColumn proportions:")
print(pd.crosstab(
    df["job"], df["y"], normalize="columns"
).round(3))

# Stacked bar chart
cross_table.plot(kind="bar", stacked=True, figsize=(10, 6))
plt.title("Job vs Subscription: Stacked Bar Chart")
plt.xlabel("Job")
plt.ylabel("Number of Customers")
plt.xticks(rotation=45, ha="right")
save_graph("qp7_job_subscription_stacked.png")

# Grouped bar chart
cross_table.plot(kind="bar", stacked=False, figsize=(10, 6))
plt.title("Job vs Subscription: Grouped Bar Chart")
plt.xlabel("Job")
plt.ylabel("Number of Customers")
plt.xticks(rotation=45, ha="right")
save_graph("qp7_job_subscription_grouped.png")


# --------------------------------------------------
# 6. GROUPED STATISTICS
# --------------------------------------------------

print("\n--- Average Duration by Subscription ---")

group_stats = df.groupby("y", observed=True)["duration"].agg(
    ["mean", "median", "std"]
)

print(group_stats.round(2))

print("\n--- Age Statistics by Job ---")

age_by_job = df.groupby("job", observed=True)["age"].agg(
    ["mean", "median", "std"]
)

print(age_by_job.round(2))


# --------------------------------------------------
# 7. SIDE-BY-SIDE BOXPLOT AND VIOLIN PLOT
# --------------------------------------------------

plt.figure(figsize=(8, 5))
sns.boxplot(data=df, x="y", y="duration")
plt.title("Call Duration by Subscription")
plt.xlabel("Subscribed")
plt.ylabel("Duration (seconds)")
save_graph("qp7_duration_boxplot_by_subscription.png")

plt.figure(figsize=(8, 5))
sns.violinplot(data=df, x="y", y="duration")
plt.title("Call Duration Distribution by Subscription")
plt.xlabel("Subscribed")
plt.ylabel("Duration (seconds)")
save_graph("qp7_duration_violin_by_subscription.png")


# --------------------------------------------------
# 8. CORRELATION: PEARSON AND SPEARMAN
# --------------------------------------------------

print("\n--- Pearson Correlation ---")

pearson_corr = df[["age", "duration", "campaign", "previous"]].corr(
    method="pearson"
)

print(pearson_corr.round(3))

print("\n--- Spearman Correlation ---")

spearman_corr = df[["age", "duration", "campaign", "previous"]].corr(
    method="spearman"
)

print(spearman_corr.round(3))

# Scatter plot with regression line
plt.figure(figsize=(8, 5))
sns.regplot(
    data=df.sample(min(2000, len(df)), random_state=42),
    x="age",
    y="duration",
    scatter_kws={"alpha": 0.35}
)
plt.title("Age vs Call Duration with Regression Line")
plt.xlabel("Age")
plt.ylabel("Duration (seconds)")
save_graph("qp7_age_duration_regression.png")


# --------------------------------------------------
# 9. MULTIDIMENSIONAL ANALYSIS
# Summary by job and subscription
# --------------------------------------------------

print("\n--- Duration Summary by Job and Subscription ---")

multi_summary = df.groupby(
    ["job", "y"], observed=True
)["duration"].agg(["mean", "median", "std"])

print(multi_summary.round(2))

# Boxplot by job and subscription
plt.figure(figsize=(12, 6))
sns.boxplot(
    data=df,
    x="job",
    y="duration",
    hue="y"
)
plt.title("Call Duration by Job and Subscription")
plt.xlabel("Job")
plt.ylabel("Duration (seconds)")
plt.xticks(rotation=45, ha="right")
save_graph("qp7_job_duration_subscription_boxplot.png")

# Grouped bar chart: average duration
mean_duration = df.groupby(
    ["job", "y"], observed=True
)["duration"].mean().unstack()

mean_duration.plot(kind="bar", figsize=(12, 6))
plt.title("Average Call Duration by Job and Subscription")
plt.xlabel("Job")
plt.ylabel("Average Duration (seconds)")
plt.xticks(rotation=45, ha="right")
save_graph("qp7_grouped_mean_duration.png")


# --------------------------------------------------
# 10. SCATTER PLOT WITH A CATEGORICAL VARIABLE
# --------------------------------------------------

sample_df = df.sample(min(2000, len(df)), random_state=42)

plt.figure(figsize=(9, 6))
sns.scatterplot(
    data=sample_df,
    x="age",
    y="duration",
    hue="y",
    style="contact",
    alpha=0.6
)
plt.title("Age vs Duration by Subscription and Contact Type")
plt.xlabel("Age")
plt.ylabel("Duration (seconds)")
save_graph("qp7_multivariable_scatter.png")


# --------------------------------------------------
# 11. CORRELATION BY SUBSCRIPTION CATEGORY
# --------------------------------------------------

print("\n--- Correlation by Subscription ---")

for category in df["y"].dropna().unique():
    subset = df[df["y"] == category]
    correlation = subset["age"].corr(subset["duration"])
    print(
        "Subscription =", category,
        "| Pearson correlation between age and duration =",
        round(correlation, 3)
    )


# --------------------------------------------------
# 12. CORRELATION MATRIX HEATMAP
# --------------------------------------------------

plt.figure(figsize=(10, 7))
sns.heatmap(
    numeric_df.corr(),
    annot=True,
    fmt=".2f",
    cmap="coolwarm",
    center=0
)
plt.title("Correlation Matrix of Numerical Variables")
save_graph("qp7_correlation_heatmap.png")


# --------------------------------------------------
# 13. PAIRS PLOT
# --------------------------------------------------

pair_data = df[
    ["age", "duration", "campaign", "previous", "y"]
].sample(min(1000, len(df)), random_state=42)

pair_plot = sns.pairplot(
    pair_data,
    vars=["age", "duration", "campaign", "previous"],
    hue="y",
    corner=True,
    diag_kind="hist",
    plot_kws={"alpha": 0.4, "s": 12}
)

pair_plot.fig.suptitle("Pairs Plot of Numerical Variables", y=1.02)
pair_plot.fig.savefig(
    "outputs/qp7_pairs_plot.png",
    dpi=150,
    bbox_inches="tight"
)
plt.close(pair_plot.fig)
print("Graph saved: qp7_pairs_plot.png")


# --------------------------------------------------
# 14. FINAL SUMMARY
# --------------------------------------------------

print("\nAssessment 7 completed successfully.")
print("Check the outputs folder for all generated graphs.")
