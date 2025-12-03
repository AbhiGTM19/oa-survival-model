// Tab Switching
function openTab(tabName) {
    const tabs = document.getElementsByClassName('tab-content');
    for (let i = 0; i < tabs.length; i++) {
        tabs[i].classList.remove('active');
    }

    const btns = document.getElementsByClassName('tab-btn');
    for (let i = 0; i < btns.length; i++) {
        btns[i].classList.remove('active');
    }

    document.getElementById(tabName).classList.add('active');
    event.currentTarget.classList.add('active');
}

// Chart Instance
let survivalChart = null;

// Form Submission
document.getElementById('analysisForm').addEventListener('submit', async function (e) {
    e.preventDefault();

    const form = e.target;
    const formData = new FormData(form);
    const loading = document.getElementById('loading');
    const results = document.getElementById('results');

    // UI State
    loading.classList.remove('hidden');
    results.classList.add('hidden');

    try {
        const response = await fetch('/api/predict', {
            method: 'POST',
            body: formData
        });

        if (!response.ok) {
            throw new Error('Analysis failed');
        }

        const data = await response.json();
        currentAnalysisData = data; // Store for report
        updateUI(data);

    } catch (error) {
        alert('Error: ' + error.message);
    } finally {
        loading.classList.add('hidden');
        results.classList.remove('hidden');
    }
});

function updateUI(data) {
    // 1. Risk Display
    const riskDiv = document.getElementById('riskDisplay');
    const riskClass = data.risk_class;
    let cssClass = 'risk-card-low';
    if (riskClass === 'High') cssClass = 'risk-card-high';
    else if (riskClass === 'Moderate') cssClass = 'risk-card-moderate';

    riskDiv.innerHTML = `
        <div class="${cssClass}">
            <div class="risk-label">${riskClass} Risk</div>
            <div class="risk-score-val">${data.risk_score.toFixed(1)}</div>
            <div style="margin-top: 10px; font-size: 0.9rem; color: #4B5563;">
                Estimated probability of rapid progression based on multi-modal analysis.
            </div>
        </div>
    `;

    // 2. Images
    document.getElementById('imgOriginal').src = `data:image/png;base64,${data.images.original}`;
    document.getElementById('imgCounterfactual').src = `data:image/png;base64,${data.images.counterfactual}`;
    document.getElementById('imgHeatmap').src = `data:image/png;base64,${data.images.heatmap}`;

    // 3. Findings
    const findingsDiv = document.getElementById('findings');
    findingsDiv.innerHTML = '<h4>Detected Structural Deviations</h4>';
    data.findings.forEach(item => {
        const cleanItem = item.replace(/\*\*/g, '');
        findingsDiv.innerHTML += `<div class="finding-item">${cleanItem}</div>`;
    });

    // 4. Chart
    renderChart(data.survival_curve);
}

function renderChart(dataPoints) {
    const ctx = document.getElementById('survivalChart').getContext('2d');

    if (survivalChart) {
        survivalChart.destroy();
    }

    const labels = dataPoints.map(p => p.x.toFixed(1));
    const values = dataPoints.map(p => p.y);

    // Population Average Baseline (Approximation)
    const baseline = dataPoints.map(p => 1.0 - Math.exp(-0.0001 * (p.x * 365)));
    // The new chart expects data points as objects with x and y properties
    const survivalData = dataPoints;

    survivalChart = new Chart(ctx, {
        type: 'line',
        data: {
            datasets: [{
                label: 'Surgery-Free Probability',
                data: survivalData,
                borderColor: '#EF4444', // Solid Red
                backgroundColor: 'rgba(239, 68, 68, 0.1)',
                borderWidth: 3,
                tension: 0.4,
                pointRadius: 0,
                pointHoverRadius: 6,
                fill: true
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: {
                    display: false
                },
                tooltip: {
                    mode: 'index',
                    intersect: false,
                    backgroundColor: 'rgba(15, 23, 42, 0.9)',
                    titleFont: { family: 'Inter', size: 13 },
                    bodyFont: { family: 'Inter', size: 13 },
                    padding: 10,
                    cornerRadius: 8,
                    displayColors: false
                }
            },
            scales: {
                x: {
                    type: 'linear',
                    position: 'bottom',
                    title: { display: true, text: 'Years from Baseline', font: { family: 'Inter', weight: 600 } },
                    grid: { display: false }
                },
                y: {
                    min: 0,
                    max: 1,
                    title: { display: true, text: 'Survival Probability', font: { family: 'Inter', weight: 600 } },
                    grid: { color: '#E2E8F0', borderDash: [4, 4] }
                }
            },
            interaction: {
                mode: 'nearest',
                axis: 'x',
                intersect: false
            }
        }
    });
}

// Report Download
document.getElementById('downloadReportBtn').addEventListener('click', async function () {
    if (!currentAnalysisData) {
        alert("Please run an analysis first.");
        return;
    }

    const btn = this;
    const originalText = btn.innerText;
    btn.innerText = "Generating PDF...";
    btn.disabled = true;

    try {
        // Prepare data for report
        const formData = new FormData(document.getElementById('analysisForm'));
        const patientData = {
            'ID': formData.get('patient_id'),
            'Age': formData.get('age'),
            'Sex': formData.get('sex'),
            'BMI': formData.get('bmi'),
            'KL Grade': formData.get('kl_grade'),
            'WOMAC': formData.get('womac'),
            'COMP': parseFloat(formData.get('bio_comp')),
            'CTX': parseFloat(formData.get('bio_ctx'))
        };

        const reportReq = {
            patient_data: patientData,
            risk_analysis: {
                score: currentAnalysisData.risk_score,
                class: currentAnalysisData.risk_class,
                prob_5yr: (currentAnalysisData.survival_curve.find(p => p.x >= 5)?.y * 100).toFixed(1) + '%' || 'N/A'
            },
            findings: currentAnalysisData.findings,
            images: currentAnalysisData.images
        };

        // Add graph image
        const canvas = document.getElementById('survivalChart');
        reportReq.images.graph = canvas.toDataURL('image/png').split(',')[1];

        const response = await fetch('/api/report', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify(reportReq)
        });

        if (!response.ok) throw new Error("Report generation failed");

        const blob = await response.blob();
        const url = window.URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = "OA_Prognosis_Report.pdf";
        document.body.appendChild(a);
        a.click();
        window.URL.revokeObjectURL(url);
        document.body.removeChild(a);

    } catch (error) {
        alert("Error generating report: " + error.message);
    } finally {
        btn.innerText = originalText;
        btn.disabled = false;
    }
});

let currentAnalysisData = null;
