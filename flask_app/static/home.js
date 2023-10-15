const form = document.querySelector('form');

// Function to handle form submission
function handleFormSubmission(event) {
    event.preventDefault(); // Prevent default form submission

    // Show the loading spinner
    document.getElementById('loading-spinner').style.display = 'block';

    // Start by sending a request to /generate_analysis
    fetch('/generate_analysis', {
        method: 'POST',
    })
    .then(response => response.json())
    .then(data => {
        if (data.status === "success") {
            // If /generate_analysis is successful, send a request to /generate_model
            return fetch('/generate_model', {
                method: 'POST'
            });
        } else {
            throw new Error('Error generating analysis.');
        }
    })
    .then(response => response.json())
    .then(data => {
        if (data.status === "success") {
            // If /generate_model is successful, send a background request to /generate_power_ranks
            return fetch('/generate_power_ranks', {
                method: 'POST'
            });
        } else {
            throw new Error(data.error || 'Error generating model.');
        }
    })
    .then(() => {
        // After all fetch calls are successful, redirect to /view_analysis
        window.location.href = '/view_analysis'; 
    })
    .catch(error => {
        console.error('Error:', error);
        alert('An error occurred. Please try again.');
        document.getElementById('loading-spinner').style.display = 'none';
    });
} 

// Attach the handleFormSubmission function to the form's submit event
form.addEventListener('submit', handleFormSubmission);
