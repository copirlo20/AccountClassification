const axios = require('axios');

async function checkUser() {
    document.getElementById('info').style.display = 'none';
    document.getElementById('result').style.display = 'none';
    const userId = document.getElementById('user_id').value.trim();
    if (!userId) return alert("Please enter a user ID");
    document.getElementById('loading').style.display = 'flex';
    try {
        const userRes = await axios.get(`http://localhost:8000/user/${userId}`);
        const userData = userRes.data;
        if (!userData || Object.keys(userData).length === 0) {
            alert("User not found.");
            return;
        }
        document.getElementById('statuses_count').value = userData.statuses_count || 0;
        document.getElementById('followers_count').value = userData.followers_count || 0;
        document.getElementById('friends_count').value = userData.friends_count || 0;
        document.getElementById('favourites_count').value = userData.favourites_count || 0;
        document.getElementById('listed_count').value = userData.listed_count || 0;
        document.getElementById('lang').value = userData.lang || '';
        document.getElementById('time_zone').value = userData.time_zone || '';
        document.getElementById('location').value = userData.location || '';

        const friendsRes = await axios.get(`http://localhost:8000/friends/${userId}`);
        const friendsData = friendsRes.data;
        const friendsList = friendsData.friends || [];
        document.getElementById('friends_list').value = friendsList.join(', ') || 'No friends found';
        document.getElementById('info').style.display = 'block';

        const fullList = [parseInt(userId), ...friendsList];
        const predictRes = await axios.post(`http://localhost:8000/predict`, {
            users_id: fullList
        });
        const { prediction, confidence } = predictRes.data;
        showPredictionResult(prediction, confidence);
    } catch (error) {
        console.error(error);
        alert(error.response.data.detail);
    } finally {
        document.getElementById('loading').style.display = 'none';
    }
}

function showPredictionResult(prediction, confidence) {
    const resultDiv = document.getElementById('result');
    resultDiv.style.display = 'block';
    resultDiv.classList.remove('result-real', 'result-fake');
    if (prediction.toLowerCase() === 'real') {
        resultDiv.style.color = 'green';
        resultDiv.classList.add('result-real');
    } else if (prediction.toLowerCase() === 'fake') {
        resultDiv.style.color = 'red';
        resultDiv.classList.add('result-fake');
    }
    resultDiv.innerHTML = `<strong>Prediction:</strong> ${prediction} - ${confidence}`;
}