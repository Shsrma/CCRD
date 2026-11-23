// config.js
// Change this URL when deploying to the cloud
const API_BASE_URL = "http://localhost:8000";

// Get token from localStorage
const token = localStorage.getItem('access_token');

// Redirect to login if token is missing
if (!token && window.location.pathname !== "/login.html") {
    window.location.href = 'login.html';
}
