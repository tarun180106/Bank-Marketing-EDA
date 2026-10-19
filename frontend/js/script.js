const chartInstances = {};

async function fetchJSON(url) {
    const response = await fetch(url);

    if (!response.ok) {
        throw new Error(`Request failed: ${url} (${response.status})`);
    }

    return response.json();
}

// Build the API URL using current filter selections
function getFilteredURL(endpoint) {
    const params = new URLSearchParams();

    const filters = {
        job: "jobFilter",
        education: "educationFilter",
        marital: "maritalFilter",
        contact: "contactFilter"
    };

    for (const key in filters) {
        const element = document.getElementById(filters[key]);

        if (element && element.value !== "all") {
            params.set(key, element.value);
        }
    }

    const query = params.toString();

    return query ? `${endpoint}?${query}` : endpoint;
}

function showNumber(value) {
    return Number(value).toLocaleString("en-IN");
}

// Safely create or replace a chart
function createChart(canvasId, config) {
    const canvas = document.getElementById(canvasId);

    if (!canvas) {
        throw new Error(`Canvas not found: ${canvasId}`);
    }

    if (chartInstances[canvasId]) {
        chartInstances[canvasId].destroy();
    }

    const existingChart = Chart.getChart(canvas);

    if (existingChart) {
        existingChart.destroy();
    }

    chartInstances[canvasId] = new Chart(canvas, config);
}

// Load dropdown options from Flask
async function loadFilterOptions() {
    const options = await fetchJSON("/api/filter-options");

    const filterConfig = [
        { key: "job", id: "jobFilter" },
        { key: "education", id: "educationFilter" },
        { key: "marital", id: "maritalFilter" },
        { key: "contact", id: "contactFilter" }
    ];

    for (let i = 0; i < filterConfig.length; i++) {
        const config = filterConfig[i];
        const select = document.getElementById(config.id);

        if (!select) {
            throw new Error(`Filter dropdown not found: ${config.id}`);
        }

        for (let j = 0; j < options[config.key].length; j++) {
            const value = options[config.key][j];
            const option = document.createElement("option");

            option.value = value;
            option.textContent = value.charAt(0).toUpperCase() + value.slice(1);

            select.appendChild(option);
        }
    }
}

// Load overview KPI cards
async function loadOverview() {
    const data = await fetchJSON(getFilteredURL("/api/overview"));

    document.getElementById("totalCustomers").textContent =
        showNumber(data.total_customers);

    document.getElementById("subscribers").textContent =
        showNumber(data.subscribers);

    document.getElementById("subscriptionRate").textContent =
        `${data.subscription_rate.toFixed(2)}%`;

    document.getElementById("averageAge").textContent =
        `${data.average_age.toFixed(1)} years`;
}

// Subscription outcomes
async function loadSubscriptionChart() {
    const data = await fetchJSON(getFilteredURL("/api/subscriptions"));

    createChart("subscriptionChart", {
        type: "doughnut",
        data: {
            labels: data.labels,
            datasets: [{
                data: data.values,
                backgroundColor: ["#4776e6", "#36c991"],
                borderWidth: 0,
                hoverOffset: 7
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            cutout: "66%",
            plugins: {
                legend: { position: "bottom" }
            }
        }
    });
}

// Job-wise subscription rates
async function loadJobChart() {
    const data = await fetchJSON(getFilteredURL("/api/jobs"));

    createChart("jobChart", {
        type: "bar",
        data: {
            labels: data.labels,
            datasets: [{
                label: "Subscription rate (%)",
                data: data.values,
                backgroundColor: "#6688eb",
                borderRadius: 5,
                maxBarThickness: 28
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
                },
                y: {
                    ticks: {
                        font: { size: 11 }
                    }
                }
            },
            plugins: {
                legend: { display: false }
            }
        }
    });
}

// Monthly subscription outcomes
async function loadMonthlyChart() {
    const data = await fetchJSON(getFilteredURL("/api/monthly"));

    createChart("monthlyChart", {
        type: "bar",
        data: {
            labels: data.labels,
            datasets: [
                {
                    label: "Not Subscribed",
                    data: data.no,
                    backgroundColor: "#9aadd1"
                },
                {
                    label: "Subscribed",
                    data: data.yes,
                    backgroundColor: "#36c991"
                }
            ]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            scales: {
                x: { stacked: true },
                y: { stacked: true, beginAtZero: true }
            }
        }
    });
}

// Customer age distribution
async function loadAgeChart() {
    const data = await fetchJSON(getFilteredURL("/api/age-distribution"));

    createChart("ageChart", {
        type: "bar",
        data: {
            labels: data.labels,
            datasets: [{
                label: "Customers",
                data: data.values,
                backgroundColor: "#6688eb",
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

// Average call duration
async function loadDurationChart() {
    const data = await fetchJSON(getFilteredURL("/api/duration"));

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

// Refresh all dashboard data using the selected filters
async function refreshDashboard() {
    const errorMessage = document.querySelector(".error-message");

    if (errorMessage) {
        errorMessage.remove();
    }

    try {
        await Promise.all([
            loadOverview(),
            loadSubscriptionChart(),
            loadJobChart(),
            loadMonthlyChart(),
            loadAgeChart(),
            loadDurationChart(),
            loadInsights()
        ]);
    } catch (error) {
        console.error("Dashboard loading failed:", error);

        if (!document.querySelector(".error-message")) {
            document.querySelector(".topbar").insertAdjacentHTML(
                "afterend",
                '<p class="error-message">Unable to update dashboard. Check the Flask server and browser console.</p>'
            );
        }
    }
}

// Set up filters and initialize the dashboard
async function initializeDashboard() {
    try {
        await loadFilterOptions();

        const filterIds = [
            "jobFilter",
            "educationFilter",
            "maritalFilter",
            "contactFilter"
        ];

        for (let i = 0; i < filterIds.length; i++) {
            document.getElementById(filterIds[i]).addEventListener(
                "change",
                refreshDashboard
            );
        }

        document.getElementById("resetFilters").addEventListener(
            "click",
            async function () {
                for (let i = 0; i < filterIds.length; i++) {
                    document.getElementById(filterIds[i]).value = "all";
                }

                await refreshDashboard();
            }
        );
        document.getElementById("exportCSV").addEventListener(
            "click",
            function () {
                window.location.href = getFilteredURL("/api/export");
            }
        );

        await refreshDashboard();
    } catch (error) {
        console.error("Dashboard initialization failed:", error);

        if (!document.querySelector(".error-message")) {
            document.querySelector(".topbar").insertAdjacentHTML(
                "afterend",
                '<p class="error-message">Dashboard initialization failed. Check the Flask server, HTML element IDs, and browser console.</p>'
            );
        }
    }
}

async function loadInsights() {
    const data = await fetchJSON(getFilteredURL("/api/insights"));
    const container = document.getElementById("insightsContainer");

    const cards = [];

    cards.push(`
        <article class="insight-card">
            <h3>Campaign Conversion</h3>
            <div class="insight-value">${data.subscription_rate}%</div>
            <p>${showNumber(data.subscribers)} out of
            ${showNumber(data.total_customers)} selected customers subscribed.</p>
        </article>
    `);

    if (data.top_job) {
        cards.push(`
            <article class="insight-card">
                <h3>Highest Job Subscription Rate</h3>
                <div class="insight-value">${data.top_job.rate}%</div>
                <p>The ${data.top_job.name} group has the highest
                observed subscription rate among the selected data.</p>
            </article>
        `);
    }

    if (data.top_education) {
        cards.push(`
            <article class="insight-card">
                <h3>Education Segment</h3>
                <div class="insight-value">${data.top_education.rate}%</div>
                <p>The ${data.top_education.name} education group has
                the highest observed subscription rate in this selection.</p>
            </article>
        `);
    }

    cards.push(`
        <article class="insight-card">
            <h3>Average Call Duration</h3>
            <div class="insight-value">${data.average_duration_subscribed} sec</div>
            <p>Average duration for subscribed customers, compared with
            ${data.average_duration_not_subscribed} seconds for customers
            who did not subscribe. This is an association, not proof of causation.</p>
        </article>
    `);

    container.innerHTML = cards.join("");
}
initializeDashboard();