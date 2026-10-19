async function fetchJSON(url) {
    const response = await fetch(url);

    if (!response.ok) {
        throw new Error(`Request failed: ${url}`);
    }

    return response.json();
}

function createChart(canvasId, config) {
    const canvas = document.getElementById(canvasId);

    if (!canvas) return;

    const existing = Chart.getChart(canvas);

    if (existing) existing.destroy();

    new Chart(canvas, config);
}

async function loadSubscriptionChart() {
    const data = await fetchJSON("/api/subscriptions");

    createChart("subscriptionChart", {
        type: "doughnut",
        data: {
            labels: data.labels,
            datasets: [{
                data: data.values,
                backgroundColor: ["#9aadd1", "#36c991"],
                borderWidth: 0
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            cutout: "65%",
            plugins: {
                legend: { position: "bottom" }
            }
        }
    });
}

async function loadJobChart() {
    const data = await fetchJSON("/api/jobs");

    createChart("jobChart", {
        type: "bar",
        data: {
            labels: data.labels,
            datasets: [{
                label: "Subscription rate (%)",
                data: data.values,
                backgroundColor: "#6688eb",
                borderRadius: 5
            }]
        },
        options: {
            indexAxis: "y",
            responsive: true,
            maintainAspectRatio: false,
            scales: {
                x: {
                    beginAtZero: true,
                    title: {
                        display: true,
                        text: "Subscription rate (%)"
                    }
                }
            },
            plugins: {
                legend: { display: false }
            }
        }
    });
}

async function loadDurationChart() {
    const data = await fetchJSON("/api/duration");

    createChart("durationChart", {
        type: "bar",
        data: {
            labels: data.labels,
            datasets: [{
                label: "Average duration (seconds)",
                data: data.values,
                backgroundColor: ["#9aadd1", "#36c991"],
                borderRadius: 5
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            scales: {
                y: { beginAtZero: true }
            },
            plugins: {
                legend: { display: false }
            }
        }
    });
}

async function initializeSubscriptions() {
    try {
        await Promise.all([
            loadSubscriptionChart(),
            loadJobChart(),
            loadDurationChart()
        ]);
    } catch (error) {
        console.error("Subscription page failed to load:", error);
    }
}

initializeSubscriptions();