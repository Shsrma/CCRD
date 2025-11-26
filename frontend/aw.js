const token = localStorage.getItem('access_token');
// ...
fetch("http://localhost:8000/alerts", { 
    headers: { 'Authorization': `Bearer ${token}` } 
})