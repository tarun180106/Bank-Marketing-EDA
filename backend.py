from pathlib import Path

import pandas as pd
from flask import Flask, jsonify, send_from_directory, request, make_response
import csv
import io
BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "bank-full.csv"
FRONTEND_DIR = BASE_DIR / "frontend"

app = Flask(__name__)


def load_data():
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Dataset not found: {DATA_PATH}")

    return pd.read_csv(DATA_PATH, sep=";")


df = load_data()


# Apply the selected filters to the dataset
def get_filtered_data():
    filtered = df.copy()

    filter_columns = {
        "job": "job",
        "education": "education",
        "marital": "marital",
        "contact": "contact"
    }

    for parameter, column in filter_columns.items():
        selected_value = request.args.get(parameter, "all")

        if selected_value and selected_value != "all":
            filtered = filtered[filtered[column] == selected_value]

    return filtered


@app.route("/")
def home():
    return send_from_directory(FRONTEND_DIR, "index.html")


@app.route("/css/<path:filename>")
def css_files(filename):
    return send_from_directory(FRONTEND_DIR / "css", filename)


@app.route("/js/<path:filename>")
def js_files(filename):
    return send_from_directory(FRONTEND_DIR / "js", filename)


# Provide dropdown options for the frontend
@app.route("/api/filter-options")
def filter_options():
    return jsonify({
        "job": sorted(df["job"].dropna().unique().tolist()),
        "education": sorted(df["education"].dropna().unique().tolist()),
        "marital": sorted(df["marital"].dropna().unique().tolist()),
        "contact": sorted(df["contact"].dropna().unique().tolist())
    })
@app.route("/subscriptions")
def subscription_page():
    return send_from_directory(
        FRONTEND_DIR, "subscriptions.html"
    )

# Overview KPI cards
@app.route("/api/overview")
def overview():
    data = get_filtered_data()

    total_customers = len(data)
    subscribers = int((data["y"] == "yes").sum())

    subscription_rate = (
        subscribers / total_customers * 100
        if total_customers else 0
    )

    average_age = (
        round(float(data["age"].mean()), 2)
        if total_customers else 0
    )

    return jsonify({
        "total_customers": total_customers,
        "subscribers": subscribers,
        "subscription_rate": round(subscription_rate, 2),
        "average_age": average_age
    })


# Subscription outcomes
@app.route("/api/subscriptions")
def subscriptions():
    data = get_filtered_data()

    counts = data["y"].value_counts().reindex(
        ["no", "yes"], fill_value=0
    )

    return jsonify({
        "labels": counts.index.tolist(),
        "values": counts.astype(int).tolist()
    })


# Subscription rates by job
@app.route("/api/jobs")
def jobs():
    data = get_filtered_data()

    job_summary = (
        data.assign(subscribed=(data["y"] == "yes").astype(int))
        .groupby("job")["subscribed"]
        .mean()
        .mul(100)
        .sort_values(ascending=False)
    )

    return jsonify({
        "labels": job_summary.index.tolist(),
        "values": job_summary.round(2).tolist()
    })


# Monthly subscription outcomes
@app.route("/api/monthly")
def monthly():
    data = get_filtered_data()

    month_order = [
        "mar", "apr", "may", "jun", "jul",
        "aug", "sep", "oct", "nov", "dec"
    ]

    counts = pd.crosstab(data["month"], data["y"])
    counts = counts.reindex(
        index=month_order, columns=["no", "yes"], fill_value=0
    )

    return jsonify({
        "labels": counts.index.tolist(),
        "no": counts["no"].astype(int).tolist(),
        "yes": counts["yes"].astype(int).tolist()
    })


# Customer age distribution
@app.route("/api/age-distribution")
def age_distribution():
    data = get_filtered_data()

    bins = [0, 25, 35, 45, 55, 65, 120]
    labels = ["18-25", "26-35", "36-45", "46-55", "56-65", "66+"]

    age_groups = pd.cut(
        data["age"],
        bins=bins,
        labels=labels,
        include_lowest=True
    )

    counts = age_groups.value_counts().reindex(
        labels, fill_value=0
    )

    return jsonify({
        "labels": counts.index.tolist(),
        "values": counts.astype(int).tolist()
    })


# Average call duration by subscription outcome
@app.route("/api/duration")
def duration():
    data = get_filtered_data()

    summary = (
        data.groupby("y")["duration"]
        .mean()
        .reindex(["no", "yes"])
    )

    values = [
        round(float(value), 2) if pd.notna(value) else 0
        for value in summary.tolist()
    ]

    return jsonify({
        "labels": ["Not Subscribed", "Subscribed"],
        "values": values
    })
@app.route("/api/insights")
def insights():
    data = get_filtered_data()

    total = len(data)
    subscribed = data[data["y"] == "yes"]
    not_subscribed = data[data["y"] == "no"]

    subscription_rate = (
        len(subscribed) / total * 100 if total else 0
    )

    # Subscription rates by job
    job_rates = (
        data.assign(subscribed=(data["y"] == "yes").astype(int))
        .groupby("job")["subscribed"]
        .mean()
        .mul(100)
        .sort_values(ascending=False)
    )

    # Subscription rates by education
    education_rates = (
        data.assign(subscribed=(data["y"] == "yes").astype(int))
        .groupby("education")["subscribed"]
        .mean()
        .mul(100)
        .sort_values(ascending=False)
    )

    avg_duration_yes = (
        float(subscribed["duration"].mean())
        if len(subscribed) else 0
    )

    avg_duration_no = (
        float(not_subscribed["duration"].mean())
        if len(not_subscribed) else 0
    )

    top_job = (
        {"name": str(job_rates.index[0]),
         "rate": round(float(job_rates.iloc[0]), 2)}
        if not job_rates.empty else None
    )

    top_education = (
        {"name": str(education_rates.index[0]),
         "rate": round(float(education_rates.iloc[0]), 2)}
        if not education_rates.empty else None
    )

    return jsonify({
        "total_customers": total,
        "subscribers": len(subscribed),
        "subscription_rate": round(subscription_rate, 2),
        "top_job": top_job,
        "top_education": top_education,
        "average_duration_subscribed": round(avg_duration_yes, 2),
        "average_duration_not_subscribed": round(avg_duration_no, 2)
    })

@app.route("/api/export")
def export_csv():
    data = get_filtered_data()

    output = io.StringIO()
    data.to_csv(output, index=False, sep=";")

    response = make_response(output.getvalue())
    response.headers["Content-Disposition"] = (
        "attachment; filename=bank_marketing_filtered.csv"
    )
    response.headers["Content-Type"] = "text/csv; charset=utf-8"

    return response
if __name__ == "__main__":
    app.run(debug=True, port=5000)