const axios = require('axios');

async function sendRequest() {
    const payload = {
        statuses_count: parseInt(document.getElementById('statuses_count').value),
        followers_count: parseInt(document.getElementById('followers_count').value),
        friends_count: parseInt(document.getElementById('friends_count').value),
        favourites_count: parseInt(document.getElementById('favourites_count').value),
        listed_count: parseInt(document.getElementById('listed_count').value),
        lang: document.getElementById('lang').value,
        time_zone: document.getElementById('time_zone').value,
        location: document.getElementById('location').value
    };

    try {
        const res = await axios.post('http://localhost:8000/predict', payload);
        document.getElementById('result').innerText = `Kết quả: ${res.data.prediction}`;
    } catch (err) {
        document.getElementById('result').innerText = `Lỗi: ${err.message}`;
    }
}
