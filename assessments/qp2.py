
import pandas as pd
import numpy as np

# Load the dataset
df = pd.read_csv("data/bank-full.csv", sep=";")

print("1. First 10 records:")
print(df.head(10))

print("\n2. Last 10 records:")
print(df.tail(10))

print("\n3. Number of rows and columns:")
print(df.shape)

print("\n4. Column names:")
print(df.columns.tolist())

print("\n5. Data types:")
print(df.dtypes)

print("\n6. Summary statistics:")
print(df.describe(include="all").T)

# 7. Missing values in each column
print("\n7. Missing values per column:")
print(df.isnull().sum())

# 8. Total missing values
print("\n8. Total missing values:")
print(df.isnull().sum().sum())

# 9. Rows containing at least one missing value
missing_rows = df[df.isnull().any(axis=1)]
print("\n9. Rows containing missing values:")
print(missing_rows.head(10))
print("Total rows with missing values:", len(missing_rows))

# 10. Remove rows containing missing values
cleaned_df = df.dropna().copy()
print("\n10. Shape after removing missing rows:")
print(cleaned_df.shape)

# 11. Mean imputation for a numerical column
mean_df = df.copy()
mean_df["age"] = mean_df["age"].fillna(
    mean_df["age"].mean()
)
print("\n11. Missing values in age after mean imputation:")
print(mean_df["age"].isnull().sum())

# 12. Median imputation for another numerical column
median_df = df.copy()
median_df["duration"] = median_df["duration"].fillna(
    median_df["duration"].median()
)
print("\n12. Missing values in duration after median imputation:")
print(median_df["duration"].isnull().sum())

# 13. Mode imputation for a categorical column
mode_df = df.copy()
mode_value = mode_df["job"].mode()

if not mode_value.empty:
    mode_df["job"] = mode_df["job"].fillna(mode_value[0])

print("\n13. Missing values in job after mode imputation:")
print(mode_df["job"].isnull().sum())

# 14. Forward filling
forward_df = df.copy()
forward_df = forward_df.ffill()

print("\n14. Missing values after forward filling:")
print(forward_df.isnull().sum().sum())

# 15. Backward filling
backward_df = df.copy()
backward_df = backward_df.bfill()

print("\n15. Missing values after backward filling:")
print(backward_df.isnull().sum().sum())

# 16. Replace remaining missing values with constants
constant_df = df.copy()

for col in constant_df.columns:
    if pd.api.types.is_numeric_dtype(constant_df[col]):
        constant_df[col] = constant_df[col].fillna(0)
    else:
        constant_df[col] = constant_df[col].fillna("Unknown")

print("\n16. Missing values after constant replacement:")
print(constant_df.isnull().sum().sum())

# 17. Identify and remove duplicate rows
print("\n17. Duplicate rows before removal:")
print(df.duplicated().sum())

deduplicated_df = df.drop_duplicates().copy()

print("Rows after removing duplicates:", len(deduplicated_df))

# 18. Convert appropriate variables to category type
category_df = df.copy()

categorical_columns = [
    "job", "marital", "education", "default",
    "housing", "loan", "contact", "month",
    "day_of_week", "poutcome", "y"
]

for col in categorical_columns:
    category_df[col] = category_df[col].astype("category")

print("\n18. Updated data types:")
print(category_df.dtypes)

# 19. Rename a column
renamed_df = df.rename(
    columns={"day_of_week": "campaign_day"}
)

print("\n19. Renamed column:")
print(renamed_df.columns.tolist())

# 20. Create a derived column
derived_df = df.copy()
derived_df["age_group"] = pd.cut(
    derived_df["age"],
    bins=[0, 25, 40, 60, 120],
    labels=["Young", "Adult", "Middle-aged", "Senior"]
)

print("\n20. Derived age group:")
print(derived_df[["age", "age_group"]].head(10))

# 21. Summary statistics
print("\n21. Summary of age:")
print(df["age"].describe())

# 22. Minimum and maximum
print("\n22. Minimum age:", df["age"].min())
print("Maximum age:", df["age"].max())

# 23. Range
print("\n23. Age range:", df["age"].max() - df["age"].min())

# 24. Variance
print("\n24. Age variance:", df["age"].var())

# 25. Standard deviation
print("\n25. Age standard deviation:", df["age"].std())

# 26. Interquartile range
q1 = df["age"].quantile(0.25)
q3 = df["age"].quantile(0.75)
iqr = q3 - q1

print("\n26. First quartile:", q1)
print("Third quartile:", q3)
print("Interquartile range:", iqr)

print("\nQP 2 completed successfully.")
