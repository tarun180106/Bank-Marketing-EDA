
import os
import pandas as pd
import numpy as np

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import seaborn as sns
from pandas.plotting import parallel_coordinates

# --------------------------------------------------
# 1. LOAD DATASET
# --------------------------------------------------

df = pd.read_csv("data/bank-full.csv", sep=";")
os.makedirs("outputs", exist_ok=True)
sns.set_theme(style="whitegrid")

print("=" * 60)
print("ASSESSMENT 5: MULTIVARIATE ANALYSIS")
print("=" * 60)

print("\nDataset shape:", df.shape)
print("\nNumerical columns:")
print(df.select_dtypes(include=np.number).columns.tolist())


def save_graph(filename):
    plt.tight_layout()
    plt.savefig(
        os.path.join("outputs", filename),
        dpi=150,
        bbox_inches="tight"
    )
    plt.close()
    print("Graph saved:", filename)


# Use a sample to make plots faster and easier to read
sample_df = df.sample(min(2000, len(df)), random_state=42)

# --------------------------------------------------
# 2. SCATTER PLOT: THREE VARIABLES
# Age, duration, and subscription
# --------------------------------------------------

plt.figure(figsize=(9, 6))

sns.scatterplot(
    data=sample_df,
    x="age",
    y="duration",
    hue="y",
    alpha=0.6
)

plt.title("Age vs Call Duration by Subscription")
plt.xlabel("Age")
plt.ylabel("Call Duration (seconds)")
save_graph("qp5_scatter_three_variables.png")

# --------------------------------------------------
# 3. SCATTER PLOT: FOUR VARIABLES
# Age, duration, campaign, and subscription
# Color represents subscription; marker represents contact
# --------------------------------------------------

plt.figure(figsize=(10, 6))

sns.scatterplot(
    data=sample_df,
    x="age",
    y="duration",
    hue="y",
    style="contact",
    alpha=0.7
)

plt.title("Multivariable Scatter Plot")
plt.xlabel("Age")
plt.ylabel("Call Duration (seconds)")
save_graph("qp5_scatter_color_shape.png")

# --------------------------------------------------
# 4. SCATTER PLOT WITH COLOR AND SIZE
# --------------------------------------------------

plt.figure(figsize=(10, 6))

sns.scatterplot(
    data=sample_df,
    x="age",
    y="duration",
    hue="y",
    size="campaign",
    sizes=(20, 180),
    alpha=0.6
)

plt.title("Age vs Duration: Color and Size Represent Extra Variables")
plt.xlabel("Age")
plt.ylabel("Call Duration (seconds)")
save_graph("qp5_scatter_color_size.png")

# --------------------------------------------------
# 5. BUBBLE PLOT
# Age vs duration, bubble size = campaign
# --------------------------------------------------

plt.figure(figsize=(10, 6))

sns.scatterplot(
    data=sample_df,
    x="age",
    y="duration",
    size="campaign",
    sizes=(20, 250),
    alpha=0.5,
    legend="brief"
)

plt.title("Bubble Plot: Age vs Duration")
plt.xlabel("Age")
plt.ylabel("Call Duration (seconds)")
save_graph("qp5_bubble_plot.png")

# --------------------------------------------------
# 6. FACET GRID
# Separate plots for subscription categories
# --------------------------------------------------

facet = sns.relplot(
    data=sample_df,
    x="age",
    y="duration",
    col="y",
    kind="scatter",
    alpha=0.5,
    height=4,
    aspect=1.2
)

facet.set_axis_labels("Age", "Call Duration (seconds)")
facet.set_titles("Subscription = {col_name}")
facet.fig.suptitle("Age vs Duration by Subscription", y=1.05)

facet.fig.savefig(
    "outputs/qp5_facet_grid.png",
    dpi=150,
    bbox_inches="tight"
)
plt.close(facet.fig)
print("Graph saved: qp5_facet_grid.png")

# --------------------------------------------------
# 7. PAIRS PLOT
# Select four numerical variables
# --------------------------------------------------

pair_columns = [
    "age",
    "duration",
    "campaign",
    "previous"
]

pair_df = sample_df[pair_columns + ["y"]].copy()

pair_plot = sns.pairplot(
    pair_df,
    vars=pair_columns,
    hue="y",
    corner=True,
    diag_kind="hist",
    plot_kws={"alpha": 0.4, "s": 15}
)

pair_plot.fig.suptitle(
    "Pairs Plot of Numerical Variables",
    y=1.02
)

pair_plot.fig.savefig(
    "outputs/qp5_pairs_plot.png",
    dpi=150,
    bbox_inches="tight"
)
plt.close(pair_plot.fig)
print("Graph saved: qp5_pairs_plot.png")

# --------------------------------------------------
# 8. PARALLEL COORDINATES PLOT
# Scale numerical variables so they can be compared
# --------------------------------------------------

parallel_columns = [
    "age",
    "duration",
    "campaign",
    "previous"
]

parallel_df = sample_df[parallel_columns + ["y"]].copy()

# Normalize each numerical column to the range 0 to 1
for col in parallel_columns:
    minimum = parallel_df[col].min()
    maximum = parallel_df[col].max()

    if maximum != minimum:
        parallel_df[col] = (
            (parallel_df[col] - minimum) / (maximum - minimum)
        )
    else:
        parallel_df[col] = 0

# Take a smaller sample for readability
parallel_sample = parallel_df.sample(
    min(150, len(parallel_df)),
    random_state=42
)

plt.figure(figsize=(10, 6))

parallel_coordinates(
    parallel_sample,
    class_column="y",
    cols=parallel_columns,
    alpha=0.35
)

plt.title("Parallel Coordinates Plot by Subscription")
plt.xlabel("Numerical Variables")
plt.ylabel("Normalized Value (0 to 1)")
save_graph("qp5_parallel_coordinates.png")

# --------------------------------------------------
# 9. SUMMARY
# --------------------------------------------------

print("\nAssessment 5 completed successfully.")
print("All graphs are saved in the outputs folder.")
