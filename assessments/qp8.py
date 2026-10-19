
import os
import pandas as pd
import numpy as np

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, AgglomerativeClustering
from scipy.cluster.hierarchy import linkage, dendrogram
from scipy.spatial.distance import pdist

# --------------------------------------------------
# 1. LOAD DATASET
# --------------------------------------------------

df = pd.read_csv("data/bank-full.csv", sep=";")
os.makedirs("outputs", exist_ok=True)
sns.set_theme(style="whitegrid")

print("=" * 60)
print("ASSESSMENT 8: K-MEANS AND HIERARCHICAL CLUSTERING")
print("=" * 60)

print("\nDataset shape:", df.shape)
print("\nTarget variable distribution:")
print(df["y"].value_counts())

# Use numerical features only for clustering
features = [
    "age",
    "duration",
    "campaign",
    "previous",
    "emp.var.rate",
    "cons.price.idx",
    "cons.conf.idx",
    "euribor3m",
    "nr.employed"
]

X = df[features].copy()
X = X.replace([np.inf, -np.inf], np.nan)
X = X.fillna(X.median())

print("\nFeatures used for clustering:")
print(features)

# Standardize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

print("\nStandardization completed.")
print("Scaled data shape:", X_scaled.shape)


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
# 2. K-MEANS WITH K = 2
# --------------------------------------------------

print("\n--- K-Means Clustering: k = 2 ---")

kmeans = KMeans(
    n_clusters=2,
    random_state=42,
    n_init=10
)

clusters = kmeans.fit_predict(X_scaled)

df["kmeans_cluster"] = clusters

print("\nCluster counts:")
print(df["kmeans_cluster"].value_counts().sort_index())

print("\nCluster centers in standardized feature space:")
centers = pd.DataFrame(
    kmeans.cluster_centers_,
    columns=features
)
print(centers.round(3))

print("\nCluster centers in original feature units:")
original_centers = pd.DataFrame(
    scaler.inverse_transform(kmeans.cluster_centers_),
    columns=features
)
print(original_centers.round(3))

print("\nInertia / WCSS for k=2:", round(kmeans.inertia_, 3))


# --------------------------------------------------
# 3. VISUALIZE K-MEANS CLUSTERS
# Use age and duration for a 2D view
# --------------------------------------------------

plt.figure(figsize=(9, 6))

sample_indices = df.sample(
    min(2000, len(df)),
    random_state=42
).index

sns.scatterplot(
    x=df.loc[sample_indices, "age"],
    y=df.loc[sample_indices, "duration"],
    hue=df.loc[sample_indices, "kmeans_cluster"].astype(str),
    palette="Set1",
    alpha=0.6
)

plt.title("K-Means Clusters (k = 2)")
plt.xlabel("Age")
plt.ylabel("Call Duration (seconds)")
save_graph("qp8_kmeans_clusters.png")


# --------------------------------------------------
# 4. WCSS FOR K = 1 TO 10
# --------------------------------------------------

print("\n--- Elbow Method: WCSS ---")

wcss = []
k_values = range(1, 11)

for k in k_values:
    model = KMeans(
        n_clusters=k,
        random_state=42,
        n_init=10
    )
    model.fit(X_scaled)
    wcss.append(model.inertia_)
    print("k =", k, "| WCSS =", round(model.inertia_, 2))

plt.figure(figsize=(8, 5))
plt.plot(list(k_values), wcss, marker="o")
plt.title("Elbow Method for Choosing k")
plt.xlabel("Number of Clusters (k)")
plt.ylabel("WCSS / Inertia")
plt.xticks(list(k_values))
save_graph("qp8_elbow_method.png")


# --------------------------------------------------
# 5. APPLY K-MEANS WITH AN ALTERNATIVE K
# k = 3 for comparison
# --------------------------------------------------

print("\n--- K-Means Clustering: k = 3 ---")

kmeans3 = KMeans(
    n_clusters=3,
    random_state=42,
    n_init=10
)

clusters3 = kmeans3.fit_predict(X_scaled)
df["kmeans_cluster_3"] = clusters3

print("\nCluster counts for k=3:")
print(df["kmeans_cluster_3"].value_counts().sort_index())


# --------------------------------------------------
# 6. HIERARCHICAL CLUSTERING
# Use a sample because hierarchical clustering is
# computationally expensive for 41,000+ rows.
# --------------------------------------------------

print("\n--- Hierarchical Clustering ---")

sample_size = min(500, len(X_scaled))
rng = np.random.RandomState(42)
sample_indices_h = rng.choice(
    len(X_scaled),
    size=sample_size,
    replace=False
)

X_hierarchical = X_scaled[sample_indices_h]

print("Hierarchical sample size:", sample_size)

# Euclidean distance and Ward linkage
distance_matrix = pdist(
    X_hierarchical,
    metric="euclidean"
)

Z = linkage(
    X_hierarchical,
    method="ward",
    metric="euclidean"
)

print("Euclidean distance calculation completed.")
print("Ward linkage completed.")


# --------------------------------------------------
# 7. DENDROGRAM
# --------------------------------------------------

plt.figure(figsize=(12, 6))

dendrogram(
    Z,
    truncate_mode="lastp",
    p=30,
    leaf_rotation=90,
    leaf_font_size=9,
    show_contracted=True
)

plt.title("Hierarchical Clustering Dendrogram (Ward Linkage)")
plt.xlabel("Cluster or Sample Index")
plt.ylabel("Euclidean Distance")
save_graph("qp8_dendrogram.png")


# --------------------------------------------------
# 8. CUT HIERARCHICAL TREE INTO 3 CLUSTERS
# --------------------------------------------------

hierarchical = AgglomerativeClustering(
    n_clusters=3,
    metric="euclidean",
    linkage="ward"
)

hierarchical_clusters = hierarchical.fit_predict(
    X_hierarchical
)

hierarchical_results = pd.DataFrame({
    "Sample_Index": sample_indices_h,
    "Hierarchical_Cluster": hierarchical_clusters
})

print("\nHierarchical cluster counts:")
print(
    hierarchical_results["Hierarchical_Cluster"]
    .value_counts()
    .sort_index()
)

print("\nFirst 10 hierarchical assignments:")
print(hierarchical_results.head(10))

# Visualize hierarchical clusters using age and duration
hierarchical_plot_df = pd.DataFrame(
    scaler.inverse_transform(X_hierarchical),
    columns=features
)
hierarchical_plot_df["cluster"] = hierarchical_clusters.astype(str)

plt.figure(figsize=(9, 6))

sns.scatterplot(
    data=hierarchical_plot_df,
    x="age",
    y="duration",
    hue="cluster",
    palette="Set2",
    alpha=0.7
)

plt.title("Hierarchical Clustering (3 Clusters)")
plt.xlabel("Age")
plt.ylabel("Call Duration (seconds)")
save_graph("qp8_hierarchical_clusters.png")


# --------------------------------------------------
# 9. SAVE CLUSTER ASSIGNMENTS
# --------------------------------------------------

df.to_csv(
    "outputs/qp8_kmeans_cluster_assignments.csv",
    index=False
)

hierarchical_results.to_csv(
    "outputs/qp8_hierarchical_cluster_assignments.csv",
    index=False
)

print("\nCluster assignments saved to outputs folder.")


# --------------------------------------------------
# 10. FINAL SUMMARY
# --------------------------------------------------

print("\nAssessment 8 completed successfully.")
print("Check the outputs folder for graphs and CSV files.")
