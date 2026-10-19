
import os
import pandas as pd
import numpy as np

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# --------------------------------------------------
# 1. LOAD DATASET
# --------------------------------------------------

df = pd.read_csv("data/bank-full.csv", sep=";")
os.makedirs("outputs", exist_ok=True)
sns.set_theme(style="whitegrid")

print("=" * 60)
print("ASSESSMENT 9: PRINCIPAL COMPONENT ANALYSIS")
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
# 2. SELECT NUMERICAL VARIABLES
# --------------------------------------------------

numeric_df = df.select_dtypes(include=np.number).copy()

print("\nNumerical columns:")
print(numeric_df.columns.tolist())

print("\nNumber of numerical columns:", numeric_df.shape[1])

# Handle missing and infinite values if present
numeric_df = numeric_df.replace([np.inf, -np.inf], np.nan)
numeric_df = numeric_df.fillna(numeric_df.median())

# Remove constant columns because they contain no variation
numeric_df = numeric_df.loc[
    :, numeric_df.nunique() > 1
]

print("\nShape after cleaning:", numeric_df.shape)

# --------------------------------------------------
# 3. STANDARDIZE THE DATA
# --------------------------------------------------

scaler = StandardScaler()
X_scaled = scaler.fit_transform(numeric_df)

print("\nStandardization completed.")
print("Standardized data shape:", X_scaled.shape)

# --------------------------------------------------
# 4. APPLY PCA
# --------------------------------------------------

pca = PCA()
pca_scores = pca.fit_transform(X_scaled)

print("\nPCA completed.")
print("Number of principal components:", pca.n_components_)

# --------------------------------------------------
# 5. EXPLAINED VARIANCE SUMMARY
# --------------------------------------------------

explained_variance = pca.explained_variance_ratio_ * 100
cumulative_variance = np.cumsum(explained_variance)

variance_table = pd.DataFrame({
    "Principal Component": [
        "PC" + str(i + 1)
        for i in range(len(explained_variance))
    ],
    "Explained Variance (%)": explained_variance,
    "Cumulative Variance (%)": cumulative_variance
})

print("\n--- Explained Variance Summary ---")
print(variance_table.round(3).to_string(index=False))

# --------------------------------------------------
# 6. SCREE PLOT
# --------------------------------------------------

plt.figure(figsize=(9, 5))

plt.plot(
    range(1, len(explained_variance) + 1),
    explained_variance,
    marker="o"
)

plt.title("PCA Scree Plot")
plt.xlabel("Principal Component")
plt.ylabel("Explained Variance (%)")
plt.xticks(range(1, len(explained_variance) + 1))
plt.ylim(0, 60)

save_graph("qp9_scree_plot.png")

# --------------------------------------------------
# 7. CUMULATIVE EXPLAINED VARIANCE
# --------------------------------------------------

plt.figure(figsize=(9, 5))

plt.plot(
    range(1, len(cumulative_variance) + 1),
    cumulative_variance,
    marker="o"
)

plt.axhline(
    y=80,
    linestyle="--",
    label="80% variance"
)

plt.axhline(
    y=90,
    linestyle="--",
    label="90% variance"
)

plt.title("Cumulative Explained Variance")
plt.xlabel("Number of Principal Components")
plt.ylabel("Cumulative Explained Variance (%)")
plt.xticks(range(1, len(cumulative_variance) + 1))
plt.legend()

save_graph("qp9_cumulative_variance.png")

# --------------------------------------------------
# 8. FIRST SIX PCA SCORES
# --------------------------------------------------

score_columns = [
    "PC" + str(i + 1)
    for i in range(pca_scores.shape[1])
]

scores_df = pd.DataFrame(
    pca_scores,
    columns=score_columns,
    index=df.index
)

print("\n--- First Six PCA Scores ---")
print(scores_df.head(6).round(4))

scores_df.head(6).to_csv(
    "outputs/qp9_first_six_pca_scores.csv",
    index=False
)

print("\nFirst six PCA scores saved.")

# --------------------------------------------------
# 9. PC1 VS PC2 SCATTER PLOT
# Color represents subscription outcome
# --------------------------------------------------

plot_df = pd.DataFrame({
    "PC1": pca_scores[:, 0],
    "PC2": pca_scores[:, 1],
    "Subscription": df["y"].astype(str)
})

sample_df = plot_df.sample(
    min(3000, len(plot_df)),
    random_state=42
)

plt.figure(figsize=(9, 6))

sns.scatterplot(
    data=sample_df,
    x="PC1",
    y="PC2",
    hue="Subscription",
    alpha=0.6
)

plt.title("PCA: PC1 vs PC2 by Subscription")
plt.xlabel(
    f"PC1 ({explained_variance[0]:.2f}% variance)"
)
plt.ylabel(
    f"PC2 ({explained_variance[1]:.2f}% variance)"
)

save_graph("qp9_pc1_pc2_scatter.png")

# --------------------------------------------------
# 10. PCA BIPLOT
# Show observations and feature loading directions
# --------------------------------------------------

loadings = pca.components_.T[:, :2]

# Use a sample of observations to keep the plot readable
sample_positions = np.random.RandomState(42).choice(
    len(pca_scores),
    size=min(1000, len(pca_scores)),
    replace=False
)

plt.figure(figsize=(11, 8))

sns.scatterplot(
    x=pca_scores[sample_positions, 0],
    y=pca_scores[sample_positions, 1],
    hue=df.iloc[sample_positions]["y"].astype(str).to_numpy(),
    alpha=0.35,
    legend=True
)

# Scale loading arrows for visibility
x_range = np.ptp(pca_scores[sample_positions, 0])
y_range = np.ptp(pca_scores[sample_positions, 1])

arrow_scale = min(x_range, y_range) * 0.35

for i, feature in enumerate(numeric_df.columns):
    x_loading = loadings[i, 0] * arrow_scale
    y_loading = loadings[i, 1] * arrow_scale

    plt.arrow(
        0, 0,
        x_loading,
        y_loading,
        color="black",
        alpha=0.8,
        head_width=0.08,
        length_includes_head=True
    )

    plt.text(
        x_loading * 1.08,
        y_loading * 1.08,
        feature,
        fontsize=8,
        ha="center",
        va="center"
    )

plt.title("PCA Biplot: Observations and Feature Loadings")
plt.xlabel(
    f"PC1 ({explained_variance[0]:.2f}% variance)"
)
plt.ylabel(
    f"PC2 ({explained_variance[1]:.2f}% variance)"
)

save_graph("qp9_biplot.png")

# --------------------------------------------------
# 11. PCA LOADINGS TABLE
# --------------------------------------------------

loadings_table = pd.DataFrame(
    pca.components_.T,
    index=numeric_df.columns,
    columns=score_columns
)

print("\n--- PCA Loadings for PC1 and PC2 ---")
print(loadings_table[["PC1", "PC2"]].round(4))

loadings_table.to_csv(
    "outputs/qp9_pca_loadings.csv"
)

print("\nPCA loadings saved.")

# --------------------------------------------------
# 12. SAVE ALL PCA SCORES
# --------------------------------------------------

scores_df.to_csv(
    "outputs/qp9_all_pca_scores.csv",
    index=False
)

# --------------------------------------------------
# 13. FINAL SUMMARY
# --------------------------------------------------

print("\n--- Final Summary ---")
print("Original dataset shape:", df.shape)
print("Numerical features used:", numeric_df.shape[1])
print("Principal components:", pca.n_components_)
print(
    "Variance explained by first two components:",
    round(cumulative_variance[1], 2),
    "%"
)

print("\nAssessment 9 completed successfully.")
print("Check the outputs folder for all graphs and CSV files.")
