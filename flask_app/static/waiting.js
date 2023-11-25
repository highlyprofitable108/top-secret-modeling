document.addEventListener('DOMContentLoaded', function() {
    let taskId = new URLSearchParams(window.location.search).get('task_id');
    console.log(`Initial Task ID retrieved: ${taskId}`); // Logging the initial retrieved task ID

    const statusElement = document.getElementById('task-status');
    const resultsLink = document.getElementById('results-link');
    const logContainer = document.getElementById('log-messages');

    // Check if essential elements exist
    if (!statusElement || !resultsLink || !logContainer) {
        console.error('One or more required elements are missing.');
        return; // Exit the function if elements are missing
    }

    // Function to check task status
    function checkTaskStatus() {
        console.log(`Checking status for task ID: ${taskId}`); // Logging each status check for current task ID
        fetch(`/task_status/${taskId}?_=${Date.now()}`)
            .then(response => {
                if (!response.ok) {
                    throw new Error('Network response was not ok');
                }
                return response.json();
            })
            .then(data => {
                console.log(`Status response for task ID ${taskId}:`, data); // Logging the status response for current task ID

                if (['PROGRESS', 'PENDING', 'INITIALIZING', 'PROCESSING', 'FINALIZING'].includes(data.state)) {
                    const capitalizedState = data.state.charAt(0).toUpperCase() + data.state.slice(1).toLowerCase();
                    const statusText = data.info ? `Current Status: ${capitalizedState}` : 'Current Status: Loading...';
                    statusElement.textContent = statusText;
                } else if (data.state === 'SUCCESS') {
                    console.log(`Updated task ID to: ${taskId}`);
                    console.log(`OLD task ID to: ${data.task_id}`);

                    if (data.task_id && data.task_id !== taskId) { 
                        console.log(`Current task ID ${taskId} completed. Moving to next task in chain: ${data.task_id}`);
                        taskId = data.task_id;
                        console.log(`Updated task ID to: ${taskId}`);
                        statusElement.textContent = `Moving to next task: ${taskId}`;
                        setTimeout(checkTaskStatus, 15000);
                    } else {
                        console.log('Final task in the chain completed');
                        statusElement.textContent = 'Task completed successfully!';
                        resultsLink.querySelector('a').href = `/view_analysis?task_id=${taskId}`;
                        resultsLink.classList.remove('hidden');
                    }
                } else if (data.state === 'FAILURE') {
                    const errorText = data.info ? `Task failed: ${data.info}` : 'Task failed: An error occurred.';
                    statusElement.textContent = errorText;
                } else {
                    const defaultStatus = data.status ? `Status: ${data.status}` : 'Status: Unknown';
                    statusElement.textContent = defaultStatus;
                }
            })
            .catch(error => {
                statusElement.textContent = 'Error fetching status. Please refresh the page.';
                console.error('Error:', error);
            });
    } 
    
    // Start the initial status check and poll every 5 seconds
    setInterval(checkTaskStatus, 5000);

    // Function to fetch and display log messages
    function fetchAndDisplayLastLog() {
        fetch('/get_logs')
            .then(response => response.json())
            .then(logs => {
                if (logs.length > 0) {
                    logContainer.textContent = logs[logs.length - 1];
                } else {
                    logContainer.textContent = 'No log messages available.';
                }
            })
            .catch(error => {
                console.error('Error fetching logs:', error);
                logContainer.textContent = 'Error fetching log messages.';
            });
    }

    // Call fetchAndDisplayLastLog periodically
    setInterval(fetchAndDisplayLastLog, 5000); // Fetch logs every 5 seconds
});
