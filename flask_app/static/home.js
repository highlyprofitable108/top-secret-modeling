document.addEventListener('DOMContentLoaded', (event) => {
    const form = document.querySelector('form');

    // Function to handle form submission
    function handleFormSubmission(event) {
        event.preventDefault();
        
        // Show the loading spinner
        showLoadingSpinner();

        // Start by sending a request to /generate_analysis
        fetch('/generate_model', {
            method: 'POST',
        })
        .then(response => response.json())
        .then(data => {
            if (data.status === "success") {
                // If /generate_model is successful, send a request to /generate_power_ranks
                return fetch('/generate_power_ranks', {
                    method: 'POST'
                });
            } else {
                throw new Error(data.error || 'Error generating model.');
            }
        })
        .then(response => response.json())
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

    function showLoadingSpinner() {
        // You can adjust the values here based on the specifics of column.js if needed
        let countdown = 450;

        document.getElementById('loading-spinner').style.display = 'block';

        // Calculate the estimated completion time
        let currentDate = new Date();
        let completionDate = new Date(currentDate.getTime() + (countdown * 1000)); // Convert seconds to milliseconds
        
        let hours = completionDate.getHours();
        let minutes = completionDate.getMinutes().toString().padStart(2, '0');
        let ampm = hours >= 12 ? 'PM' : 'AM';
        hours = hours % 12;
        hours = hours ? hours : 12; // the hour '0' should be '12'
        
        let completionTime = `${hours}:${minutes} ${ampm}`;
        
        // Display the estimated completion time
        document.getElementById('countdown-timer').textContent = "Estimated completion time: " + completionTime;

        return true; // Show the spinner
    }


    // Attach the handleFormSubmission function to the form's submit event
    form.addEventListener('submit', handleFormSubmission);

});