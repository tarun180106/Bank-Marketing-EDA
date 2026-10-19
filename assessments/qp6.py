
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
print("ASSESSMENT 6: TIME-RELATED PATTERN ANALYSIS")
print("=" * 60)

print("\nDataset shape:", df.shape)
print("\nAvailable columns:")
print(df.columns.tolist())


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
# 2. CUSTOMER COUNTS BY MONTH
# --------------------------------------------------

print("\n--- Customer Count by Month ---")

month_order = [
    "mar", "apr", "may", "jun", "jul", "aug",
    "sep", "oct", "nov", "dec"
]

month_counts = df["month"].value_counts()

available_months = [
    month for month in month_order
    if month in month_counts.index
]

month_counts = month_counts.reindex(available_months)

print(month_counts)

plt.figure(figsize=(9, 5))
month_counts.plot(kind="bar")
plt.title("Number of Customers Contacted by Month")
plt.xlabel("Month")
plt.ylabel("Number of Customers")
plt.xticks(rotation=0)
save_graph("qp6_customers_by_month.png")


# --------------------------------------------------
# 3. SUBSCRIPTIONS BY MONTH
# --------------------------------------------------

print("\n--- Subscription Counts by Month ---")

monthly_subscriptions = pd.crosstab(
    df["month"],
    df["y"]
)

monthly_subscriptions = monthly_subscriptions.reindex(
    available_months
)

print(monthly_subscriptions)

monthly_subscriptions.plot(
    kind="bar",
    stacked=True,
    figsize=(10, 6)
)

plt.title("Subscription Counts by Month")
plt.xlabel("Month")
plt.ylabel("Number of Customers")
plt.xticks(rotation=0)
save_graph("qp6_subscriptions_by_month.png")


# --------------------------------------------------
# 4. SUBSCRIPTION RATE BY MONTH
# --------------------------------------------------

monthly_rate = (
    df.assign(subscribed=(df["y"] == "yes").astype(int))
    .groupby("month", observed=True)["subscribed"]
    .mean()
    .mul(100)
    .reindex(available_months)
)

print("\n--- Subscription Rate by Month (%) ---")
print(monthly_rate.round(2))

plt.figure(figsize=(9, 5))
plt.plot(
    monthly_rate.index,
    monthly_rate.values,
    marker="o"
)

plt.title("Subscription Rate by Month")
plt.xlabel("Month")
plt.ylabel("Subscription Rate (%)")
save_graph("qp6_monthly_subscription_rate.png")


# --------------------------------------------------
# 5. CUSTOMER COUNTS BY DAY OF WEEK
# --------------------------------------------------

print("\n--- Customer Count by Day of Week ---")

day_order = ["mon", "tue", "wed", "thu", "fri"]

day_counts = df["day_of_week"].value_counts()
available_days = [
    day for day in day_order
    if day in day_counts.index
]

day_counts = day_counts.reindex(available_days)

print(day_counts)

plt.figure(figsize=(8, 5))
day_counts.plot(kind="bar")
plt.title("Number of Customers Contacted by Day")
plt.xlabel("Day of Week")
plt.ylabel("Number of Customers")
plt.xticks(rotation=0)
save_graph("qp6_customers_by_day.png")


# --------------------------------------------------
# 6. SUBSCRIPTION RATE BY DAY OF WEEK
# --------------------------------------------------

daily_rate = (
    df.assign(subscribed=(df["y"] == "yes").astype(int))
    .groupby("day_of_week", observed=True)["subscribed"]
    .mean()
    .mul(100)
    .reindex(available_days)
)

print("\n--- Subscription Rate by Day (%) ---")
print(daily_rate.round(2))

plt.figure(figsize=(8, 5))
plt.plot(
    daily_rate.index,
    daily_rate.values,
    marker="o"
)

plt.title("Subscription Rate by Day of Week")
plt.xlabel("Day of Week")
plt.ylabel("Subscription Rate (%)")
save_graph("qp6_subscription_rate_by_day.png")


# --------------------------------------------------
# 7. MONTHLY AVERAGE CALL DURATION
# --------------------------------------------------

monthly_duration = (
    df.groupby("month", observed=True)["duration"]
    .mean()
    .reindex(available_months)
)

print("\n--- Average Call Duration by Month ---")
print(monthly_duration.round(2))

plt.figure(figsize=(9, 5))
plt.plot(
    monthly_duration.index,
    monthly_duration.values,
    marker="o"
)

plt.title("Average Call Duration by Month")
plt.xlabel("Month")
plt.ylabel("Average Duration (seconds)")
save_graph("qp6_average_duration_by_month.png")


# --------------------------------------------------
# 8. MONTH-DAY SUBSCRIPTION RATE HEATMAP
# --------------------------------------------------

heatmap_data = pd.crosstab(
    df["month"],
    df["day_of_week"],
    values=(df["y"] == "yes").astype(int),
    aggfunc="mean"
) * 100

heatmap_data = heatmap_data.reindex(available_months)
heatmap_data = heatmap_data.reindex(
    columns=available_days
)

print("\n--- Subscription Rate by Month and Day (%) ---")
print(heatmap_data.round(2))

plt.figure(figsize=(9, 6))
sns.heatmap(
    heatmap_data,
    annot=True,
    fmt=".1f",
    cmap="YlGnBu"
)

plt.title("Subscription Rate by Month and Day (%)")
plt.xlabel("Day of Week")
plt.ylabel("Month")
save_graph("qp6_month_day_heatmap.png")


# --------------------------------------------------
# 9. LIMITATION OF CHRONOLOGICAL TIME SERIES
# --------------------------------------------------

print("\n--- Time-Series Limitation ---")

if "year" not in df.columns and "date" not in df.columns:
    print(
        "The dataset has no complete date or year column. "
        "Monthly and weekday patterns can be analyzed, "
        "but chronological forecasting such as ARIMA cannot "
        "be justified from these columns alone."
    )


# --------------------------------------------------
# 10. FINAL SUMMARY
# --------------------------------------------------

print("\nAssessment 6 completed successfully.")
print("Check the outputs folder for all generated graphs.")
