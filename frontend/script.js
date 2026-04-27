const form = document.getElementById('fraudForm');
const resultPanel = document.getElementById('result');
const fillSampleBtn = document.getElementById('fillSampleBtn');

function renderResult({ label, probability, risk }) {
  const riskClass = risk.toLowerCase();
  const heading = label === 'Fraud' ? '⚠️ Fraud Alert' : '✅ Transaction Looks Normal';

  resultPanel.className = `result-panel ${riskClass}`;
  resultPanel.innerHTML = `
    <p class="result-title">${heading}</p>
    <div class="metric">${(probability * 100).toFixed(2)}%</div>
    <p class="result-text">Fraud Probability</p>
    <p class="result-meta"><strong>Risk Level:</strong> ${risk}</p>
    <p class="result-meta"><strong>Model Decision:</strong> ${label}</p>
  `;
}

function renderError(message) {
  resultPanel.className = 'result-panel high';
  resultPanel.innerHTML = `
    <p class="result-title">Unable to process request</p>
    <p class="result-text">${message}</p>
  `;
}

fillSampleBtn.addEventListener('click', () => {
  document.getElementById('amount').value = 4500;
  document.getElementById('hour_of_day').value = 3;
  document.getElementById('day_of_week').value = 6;
  document.getElementById('hours_since_last_txn').value = 0.5;
  document.getElementById('txn_count_24h').value = 5;
  document.getElementById('txn_count_7d').value = 25;
  document.getElementById('amount_deviation').value = 4.5;
  document.getElementById('is_high_amount').value = 1;
  document.getElementById('is_unusual_location').value = 1;
  document.getElementById('location_changed').value = 1;
});

form.addEventListener('submit', async (event) => {
  event.preventDefault();

  const data = {
    amount: Number(document.getElementById('amount').value),
    hour_of_day: Number(document.getElementById('hour_of_day').value),
    day_of_week: Number(document.getElementById('day_of_week').value),
    hours_since_last_txn: Number(document.getElementById('hours_since_last_txn').value),
    txn_count_24h: Number(document.getElementById('txn_count_24h').value),
    txn_count_7d: Number(document.getElementById('txn_count_7d').value),
    amount_deviation: Number(document.getElementById('amount_deviation').value),
    is_high_amount: Number(document.getElementById('is_high_amount').value),
    is_unusual_location: Number(document.getElementById('is_unusual_location').value),
    location_changed: Number(document.getElementById('location_changed').value),
  };

  resultPanel.className = 'result-panel idle';
  resultPanel.innerHTML = `
    <p class="result-title">Analyzing transaction...</p>
    <p class="result-text">The model is checking behavioral and location-based patterns.</p>
  `;

  try {
    const response = await fetch('http://127.0.0.1:5000/predict', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(data),
    });

    const result = await response.json();

    if (!response.ok || !result.success) {
      throw new Error(result.error || 'Unknown server error');
    }

    renderResult(result);
  } catch (error) {
    renderError(error.message + '. Make sure the Flask backend is running on port 5000.');
  }
});
