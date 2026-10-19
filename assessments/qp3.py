
import os
import pandas as pd
import numpy as np

# Use a non-GUI backend to avoid the Tcl/Tk error
import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import seaborn as sns

# --------------------------------------------------
# 1. SETUP
# --------------------------------------------------

df = pd.read_csv("data/bank-full.csv", sep=";")

os.makedirs("outputs", exist_ok=True)

sns.set_theme(style="whitegrid")

print("=" * 60)
print("ASSESSMENT 3: DATA CLEANING AND UNIVARIATE ANALYSIS")
print("=" * 60)

print("\nOriginal dataset shape:", df.shape)
print("\nFirst 5 rows:")
print(df.head())


# Helper function to save every graph with a unique name
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
# 2. MISSING VALUE ANALYSIS
# --------------------------------------------------

print("\n--- Missing Value Analysis ---")

missing = df.isnull().sum()
print("\nMissing values in each column:")
print(missing)

print("\nTotal missing values:", int(df.isnull().sum().sum()))

# Remove columns with more than 70% missing values
threshold = 0.70 * len(df)
columns_to_remove = df.columns[df.isnull().sum() > threshold]

clean_df = df.drop(columns=columns_to_remove).copy()

print("\nColumns removed (>70% missing):", list(columns_to_remove))
print("Shape after removing columns:", clean_df.shape)

# --------------------------------------------------
# 3. MISSING VALUE IMPUTATION
# --------------------------------------------------

print("\n--- Missing Value Imputation ---")

# Numerical columns: mean and median imputation
numeric_cols = clean_df.select_dtypes(include=np.number).columns.tolist()

for col in numeric_cols:
    if clean_df[col].isnull().any():
        clean_df[col] = clean_df[col].fillna(clean_df[col].mean())

for col in numeric_cols:
    if clean_df[col].isnull().any():
        clean_df[col] = clean_df[col].fillna(clean_df[col].median())

# Categorical columns: mode imputation
categorical_cols = clean_df.select_dtypes(
    include=["object", "category"]
).columns.tolist()

for col in categorical_cols:
    if clean_df[col].isnull().any():
        mode_values = clean_df[col].mode()
        if not mode_values.empty:
            clean_df[col] = clean_df[col].fillna(mode_values.iloc[0])

print("Missing values after imputation:",
      int(clean_df.isnull().sum().sum()))

# --------------------------------------------------
# 4. DUPLICATE DETECTION AND REMOVAL
# --------------------------------------------------

print("\n--- Duplicate Detection ---")

duplicate_count = int(clean_df.duplicated().sum())
print("Duplicate rows found:", duplicate_count)

clean_df = clean_df.drop_duplicates().copy()

print("Shape after removing duplicates:", clean_df.shape)

# --------------------------------------------------
# 5. DATA TYPE CONVERSION
# --------------------------------------------------

print("\n--- Data Types ---")
print(clean_df.dtypes)

# Convert categorical columns to category type
for col in categorical_cols:
    if col in clean_df.columns:
        clean_df[col] = clean_df[col].astype("category")

print("\nData types after conversion:")
print(clean_df.dtypes)

# --------------------------------------------------
# 6. TEXT TRANSFORMATION
# --------------------------------------------------

print("\n--- Text Transformation ---")

text_df = clean_df.copy()

for col in text_df.select_dtypes(include=["object"]).columns:
    text_df[col] = text_df[col].str.lower()

print("Text values converted to lowercase where applicable.")

# Demonstrate uppercase conversion on a copy of the job column
if "job" in clean_df.columns:
    job_upper = clean_df["job"].astype(str).str.upper()
    print("\nFirst 5 uppercase job values:")
    print(job_upper.head().to_list())

# --------------------------------------------------
# 7. OUTLIER DETECTION USING IQR
# --------------------------------------------------

print("\n--- Outlier Detection using IQR ---")

if "age" in clean_df.columns:
    Q1 = clean_df["age"].quantile(0.25)
    Q3 = clean_df["age"].quantile(0.75)
    IQR = Q3 - Q1

    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    outliers = clean_df[
        (clean_df["age"] < lower_bound) |
        (clean_df["age"] > upper_bound)
    ]

    print("Q1:", Q1)
    print("Q3:", Q3)
    print("IQR:", IQR)
    print("Lower bound:", lower_bound)
    print("Upper bound:", upper_bound)
    print("Age outliers found:", len(outliers))

    # Histogram of age
    plt.figure(figsize=(8, 5))
    sns.histplot(clean_df["age"], bins=20, kde=False)
    plt.title("Histogram of Customer Age")
    plt.xlabel("Age")
    plt.ylabel("Frequency")
    save_graph("qp3_age_histogram.png")

    # Boxplot of age
    plt.figure(figsize=(8, 5))
    sns.boxplot(x=clean_df["age"])
    plt.title("Boxplot of Customer Age")
    plt.xlabel("Age")
    save_graph("qp3_age_boxplot.png")

# --------------------------------------------------
# 8. HISTOGRAM WITH DENSITY
# --------------------------------------------------

if "duration" in clean_df.columns:
    plt.figure(figsize=(8, 5))
    sns.histplot(clean_df["duration"], bins=30, kde=True)
    plt.title("Distribution of Call Duration")
    plt.xlabel("Call Duration (seconds)")
    plt.ylabel("Frequency")
    save_graph("qp3_duration_histogram_density.png")

# --------------------------------------------------
# 9. VIOLIN PLOT
# --------------------------------------------------

if "age" in clean_df.columns and "y" in clean_df.columns:
    plt.figure(figsize=(8, 5))
    sns.violinplot(data=clean_df, x="y", y="age")
    plt.title("Customer Age Distribution by Subscription")
    plt.xlabel("Subscribed")
    plt.ylabel("Age")
    save_graph("qp3_age_violinplot.png")

# --------------------------------------------------
# 10. DOT PLOT
# --------------------------------------------------

if "age" in clean_df.columns:
    sample_ages = clean_df["age"].dropna().head(30)

    plt.figure(figsize=(8, 5))
    plt.plot(
        sample_ages.to_numpy(),
        np.zeros(len(sample_ages)),
        "o"
    )
    plt.title("Dot Plot of First 30 Customer Ages")
    plt.xlabel("Age")
    plt.yticks([])
    save_graph("qp3_age_dotplot.png")

# --------------------------------------------------
# 11. FREQUENCY, PROPORTION AND PERCENTAGE
# --------------------------------------------------

print("\n--- Frequency Table for Subscription ---")

if "y" in clean_df.columns:
    frequency = clean_df["y"].value_counts()
    proportion = clean_df["y"].value_counts(normalize=True)
    percentage = proportion * 100

    frequency_table = pd.DataFrame({
        "Frequency": frequency,
        "Proportion": proportion,
        "Percentage": percentage
    })

    print(frequency_table)

    # Bar chart
    plt.figure(figsize=(7, 5))
    sns.countplot(data=clean_df, x="y")
    plt.title("Subscription Frequency")
    plt.xlabel("Subscribed")
    plt.ylabel("Number of Customers")
    save_graph("qp3_subscription_bar_chart.png")

    # Percentage bar chart
    plt.figure(figsize=(7, 5))
    (proportion * 100).plot(kind="bar")
    plt.title("Subscription Percentage")
    plt.xlabel("Subscribed")
    plt.ylabel("Percentage (%)")
    save_graph("qp3_subscription_percentage_bar.png")

    # Pie chart
    plt.figure(figsize=(7, 7))
    frequency.plot(
        kind="pie",
        autopct="%1.1f%%",
        startangle=90
    )
    plt.title("Subscription Distribution")
    plt.ylabel("")
    save_graph("qp3_subscription_pie_chart.png")

# --------------------------------------------------
# 12. HORIZONTAL BAR CHART FOR JOB
# --------------------------------------------------

if "job" in clean_df.columns:
    job_counts = clean_df["job"].astype(str).value_counts()

    plt.figure(figsize=(10, 6))
    job_counts.sort_values().plot(kind="barh")
    plt.title("Customer Count by Job")
    plt.xlabel("Number of Customers")
    plt.ylabel("Job")
    save_graph("qp3_job_horizontal_bar.png")

# --------------------------------------------------
# 13. LOLLIPOP CHART
# --------------------------------------------------

if "job" in clean_df.columns:
    job_counts = clean_df["job"].astype(str).value_counts()

    plt.figure(figsize=(10, 6))
    x_positions = np.arange(len(job_counts))

    plt.vlines(
        x=x_positions,
        ymin=0,
        ymax=job_counts.values
    )
    plt.plot(
        x_positions,
        job_counts.values,
        "o"
    )

    plt.xticks(
        x_positions,
        job_counts.index,
        rotation=45,
        ha="right"
    )

    plt.title("Lollipop Chart of Customers by Job")
    plt.xlabel("Job")
    plt.ylabel("Number of Customers")
    save_graph("qp3_job_lollipop_chart.png")

# --------------------------------------------------
# 14. DENSITY PLOT
# --------------------------------------------------

if "age" in clean_df.columns:
    plt.figure(figsize=(8, 5))
    sns.kdeplot(data=clean_df, x="age", fill=True)
    plt.title("Density Plot of Customer Age")
    plt.xlabel("Age")
    plt.ylabel("Density")
    save_graph("qp3_age_density.png")

# --------------------------------------------------
# 15. FACET PLOT
# --------------------------------------------------

if "age" in clean_df.columns and "y" in clean_df.columns:
    grid = sns.displot(
        data=clean_df,
        x="age",
        col="y",
        bins=20,
        height=4,
        aspect=1.2
    )

    grid.set_axis_labels("Age", "Frequency")
    grid.set_titles("Subscription = {col_name}")
    grid.fig.suptitle(
        "Customer Age Distribution by Subscription",
        y=1.05
    )

    grid.fig.savefig(
        "outputs/qp3_age_facet_plot.png",
        dpi=150,
        bbox_inches="tight"
    )
    plt.close(grid.fig)
    print("Graph saved: qp3_age_facet_plot.png")

# --------------------------------------------------
# 16. FINAL SUMMARY
# --------------------------------------------------

print("\n--- Final Summary ---")
print("Original dataset shape:", df.shape)
print("Cleaned dataset shape:", clean_df.shape)
print("Missing values remaining:",
      int(clean_df.isnull().sum().sum()))
print("Duplicate rows remaining:",
      int(clean_df.duplicated().sum()))

print("\nAssessment 3 completed successfully.")
print("Check the outputs folder for the generated graphs.")
