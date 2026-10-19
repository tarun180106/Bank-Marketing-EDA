
import pandas as pd
import numpy as np

# Load the dataset
df = pd.read_csv("data/bank-full.csv", sep=";")

print("Dataset shape:", df.shape)
print("Available columns:", df.columns.tolist())

# ==================================================
# PART A: DATAFRAME OPERATIONS
# ==================================================

# Create three dataframes using existing columns
dataframe1 = df[["age", "job", "duration"]].head(10).copy()
dataframe2 = df[["age", "marital", "education"]].head(10).copy()
dataframe3 = df[["age", "job", "duration"]].iloc[10:20].copy()

# Add a unique row identifier to demonstrate joins
dataframe1.insert(0, "RecordID", range(1, 11))
dataframe2.insert(0, "RecordID", range(1, 11))
dataframe3.insert(0, "RecordID", range(11, 21))

print("\nDataframe 1:")
print(dataframe1)

print("\nDataframe 2:")
print(dataframe2)

print("\nDataframe 3:")
print(dataframe3)

# 1. Append rows of dataframe1 and dataframe3
appended_df = pd.concat(
    [dataframe1, dataframe3],
    ignore_index=True
)

print("\n1. Appended DataFrame:")
print(appended_df)

# 2. Concatenate columns
concatenated_df = pd.concat(
    [
        dataframe1.reset_index(drop=True),
        dataframe2.reset_index(drop=True)
    ],
    axis=1
)

print("\n2. Column-wise Concatenation:")
print(concatenated_df)

# 3. Merge using RecordID
merged_df = pd.merge(
    dataframe1,
    dataframe2,
    on="RecordID",
    how="inner",
    suffixes=("_df1", "_df2")
)

print("\n3. Merged DataFrame:")
print(merged_df)

# 4. Join operations
inner_join = pd.merge(
    dataframe1, dataframe2, on="RecordID",
    how="inner", suffixes=("_df1", "_df2")
)

left_join = pd.merge(
    dataframe1, dataframe2, on="RecordID",
    how="left", suffixes=("_df1", "_df2")
)

right_join = pd.merge(
    dataframe1, dataframe2, on="RecordID",
    how="right", suffixes=("_df1", "_df2")
)

full_outer_join = pd.merge(
    dataframe1, dataframe2, on="RecordID",
    how="outer", suffixes=("_df1", "_df2")
)

print("\nInner Join:")
print(inner_join)

print("\nLeft Join:")
print(left_join)

print("\nRight Join:")
print(right_join)

print("\nFull Outer Join:")
print(full_outer_join)

# ==================================================
# PART B: DATA TRANSFORMATIONS
# ==================================================

# Use the first 10 customer ages as the numerical vector
values = df["age"].head(10).astype(float)

print("\nOriginal numerical vector:")
print(values.to_list())

minimum = values.min()
maximum = values.max()

# 1. Normalize to [-3, +3]
if maximum != minimum:
    normalized_3 = (
        (values - minimum) / (maximum - minimum) * 6 - 3
    )
else:
    normalized_3 = values * 0

print("\nNormalization to [-3, +3]:")
print(normalized_3.to_list())

# Normalize to [-1, +1]
if maximum != minimum:
    normalized_1 = (
        (values - minimum) / (maximum - minimum) * 2 - 1
    )
else:
    normalized_1 = values * 0

print("\nNormalization to [-1, +1]:")
print(normalized_1.to_list())

# Min-Max normalization to [0, 1]
if maximum != minimum:
    min_max = (values - minimum) / (maximum - minimum)
else:
    min_max = values * 0

print("\nMin-Max Normalization [0, 1]:")
print(min_max.to_list())

# 2. Z-score normalization
std = values.std(ddof=0)

if std != 0:
    z_score = (values - values.mean()) / std
else:
    z_score = values * 0

print("\nZ-score Normalization:")
print(z_score.to_list())

# 3. Decimal scaling normalization
max_abs = values.abs().max()
j = 0

while max_abs / (10 ** j) >= 1:
    j += 1

decimal_scaling = values / (10 ** j)

print("\nDecimal Scaling Normalization:")
print(decimal_scaling.to_list())

print("\nQP 1 completed successfully.")
