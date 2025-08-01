const axios = require('axios');

async function sendRequest() {
    const payload = {
        statuses_count: parseInt(document.getElementById('statuses_count').value),
        followers_count: parseInt(document.getElementById('followers_count').value),
        friends_count: parseInt(document.getElementById('friends_count').value),
        favourites_count: parseInt(document.getElementById('favourites_count').value),
        listed_count: parseInt(document.getElementById('listed_count').value),
        lang: (document.getElementById('lang').value ?? "").trim() || "NaN",
        time_zone: (document.getElementById('time_zone').value ?? "").trim() || "NaN",
        location: (document.getElementById('location').value ?? "").trim() || "NaN"
    };

    const resultEl = document.getElementById('result');
    const loadingEl = document.getElementById('loading');
    resultEl.innerText = "";
    loadingEl.style.display = 'flex';

    try {
        const res = await axios.post('http://localhost:8000/predict', payload);
        resultEl.innerText = `Result: ${res.data.prediction} - ${(res.data.confidence * 100).toFixed(2)}%`;
    } catch (err) {
        resultEl.innerText = `Error: ${err.message}`;
    } finally {
        loadingEl.style.display = 'none';
    }
}