document.addEventListener('DOMContentLoaded', function() {
    let taskId = new URLSearchParams(window.location.search).get('task_id');
    console.log(`Task ID retrieved: ${taskId}`); // Added logging
    const statusElement = document.getElementById('task-status');
    const resultsLink = document.getElementById('results-link');

    // Function to check task status
    function checkTaskStatus() {
        console.log(`Checking status for task: ${taskId}`); // Added logging
        fetch(`/task_status/${taskId}?_=${Date.now()}`)
            .then(response => {
                if (!response.ok) {
                    throw new Error('Network response was not ok');
                }
                return response.json();
            })
            .then(data => {
                console.log(`Status response for task ID ${taskId}:`, data); // Added logging
                if (['PROGRESS', 'INITIALIZING', 'PROCESSING', 'FINALIZING'].includes(data.state)) {
                    const statusText = data.info ? `Current Status: ${data.info}` : 'Current Status: Loading...';
                    statusElement.textContent = statusText;
                } else if (data.state === 'SUCCESS') {
                    if (data.task_id && data.task_id !== taskId) {
                        console.log(`Moving to next task in chain: ${data.task_id}`); // Debugging log
                        taskId = data.task_id; // Update taskId with the new task ID
                        statusElement.textContent = `Moving to next task: ${taskId}`;
                        setTimeout(checkTaskStatus, 15000); // Delay for 15 seconds before checking the new task
                    } else {
                        console.log('Final task in the chain completed'); // Debugging log
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
});
