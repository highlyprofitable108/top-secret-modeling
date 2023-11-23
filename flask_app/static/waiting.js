document.addEventListener('DOMContentLoaded', function() {
    let taskId = new URLSearchParams(window.location.search).get('task_id');
    const statusElement = document.getElementById('task-status');
    const resultsLink = document.getElementById('results-link');
  
    // Function to check task status
    function checkTaskStatus() {
        fetch(`/task_status/${taskId}`)
            .then(response => response.json())
            .then(data => {
                if (data.state === 'PROGRESS') {
                    statusElement.textContent = `Current Status: ${data.status} (${data.current}/${data.total})`;
                } else if (data.state === 'SUCCESS') {
                    if (data.result && data.result.task_id) {
                        // Update taskId to the ID of the next task in the chain
                        taskId = data.result.task_id;
                        // Continue checking the status of the new task
                        setTimeout(checkTaskStatus, 5000);
                    } else {
                        // Final task completed
                        statusElement.textContent = 'Task completed successfully!';
                        resultsLink.querySelector('a').href = `/view_analysis?task_id=${taskId}`;
                        resultsLink.classList.remove('hidden');
                    }
                } else {
                    statusElement.textContent = `Status: ${data.status}`;
                }
            })
            .catch(error => {
                statusElement.textContent = 'Error fetching status. Please refresh the page.';
                console.error('Error:', error);
            });
    }
    
    // Start the initial status check
    checkTaskStatus();    
  
    // Poll the task status every 15 seconds
    setInterval(checkTaskStatus, 1500);
});
