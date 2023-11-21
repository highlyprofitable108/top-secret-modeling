document.addEventListener('DOMContentLoaded', (event) => {
    // Get references to DOM elements
    const selectedColumnsList = document.getElementById('selectedColumnsList');
    const checkboxes = document.querySelectorAll('.column-checkbox');
    const selectAllButton = document.getElementById('selectAll');
    const selectRandomButton = document.getElementById('selectRandom');
    const form = document.querySelector('form');

    // Add event listener to update the list of selected columns whenever a checkbox is changed
    checkboxes.forEach(checkbox => {
        checkbox.addEventListener('change', () => {
            updateSelectedColumnsList();
        });
    });

    // Add event listener to select all checkboxes when the "Select All" button is clicked
    selectAllButton.addEventListener('click', () => {
        checkboxes.forEach(checkbox => {
            checkbox.checked = true;
        });
        updateSelectedColumnsList();
    });

    // Add event listener to select a random number of checkboxes when the "Random Selection" button is clicked
    selectRandomButton.addEventListener('click', () => {
        const randomCount = Math.floor(Math.random() * checkboxes.length);
        checkboxes.forEach(checkbox => {
            checkbox.checked = false;
        });
        for (let i = 0; i < randomCount; i++) {
            const randomIndex = Math.floor(Math.random() * checkboxes.length);
            checkboxes[randomIndex].checked = true;
        }
        updateSelectedColumnsList();
    });

    // Add event listeners to category "Select All" checkboxes
    const categorySelectAllCheckboxes = document.querySelectorAll('.category-select-all');
    categorySelectAllCheckboxes.forEach(checkbox => {
        checkbox.addEventListener('change', (event) => {
            const category = event.target.getAttribute('data-category');
            const categoryCheckboxes = document.querySelectorAll(`.column-checkbox[data-category="${category}"]`);
            categoryCheckboxes.forEach(box => {
                box.checked = event.target.checked;
            });
            updateSelectedColumnsList(); // Update the list of selected columns
        });
    });

    // Function to update the list of selected columns
    function updateSelectedColumnsList() {
        const selectedColumns = {};
        checkboxes.forEach(box => {
            if (box.checked) {
                const [category, ...columnParts] = box.value.split('.');
                const column = columnParts.join('.');
                if (!selectedColumns[category]) {
                    selectedColumns[category] = [];
                }
                selectedColumns[category].push(column.replace(/_/g, ' ').replace(/Totals\./i, '').replace(/\./g, ' ').replace(/\b\w/g, l => l.toUpperCase()));
            }
        });

        const selectedColumnsText = Object.entries(selectedColumns)
            .map(([category, columns]) => {
                return `<strong>${category.replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase()).replace('Totals.', '')}:</strong> ${columns.join(', ')}`;
            })
            .join('<br>');

        selectedColumnsList.innerHTML = selectedColumnsText || 'None';
    }

    function handleFormSubmission(event, isQuickTest = false) {
        event.preventDefault(); // Prevent default form submission
    
        // Check if any checkboxes are selected
        const checkboxesSelected = Array.from(checkboxes).some(checkbox => checkbox.checked);
        if (!checkboxesSelected) {
            alert('Please select at least one column before running the simulation.');
            return; // Stop the function if no checkboxes are selected
        }
    
        // Call the showLoadingSpinner function
        showLoadingSpinner();
    
        // Prepare the data to be sent to the server
        const formData = new FormData(form);
        formData.append('quick_test', String(isQuickTest));
        console.log('FormData:', Array.from(formData.entries()));

        // Send form data to /process_columns
        fetch('/process_columns', {
            method: 'POST',
            body: formData
        })
        .then(response => response.json())
        .then(data => {
            if (data.success) {
                // If /process_columns is successful, send a request to /generate_model
                return fetch('/generate_model', { method: 'POST' });
            } else {
                throw new Error('Error processing columns.');
            }
        })
        .then(response => response.json())
        .then(data => {
            if (data.status === "success") {
                // If /generate_model is successful, poll for task status
                return pollTaskStatus(data.task_id, '/sim_runner');
            } else {
                throw new Error(data.error || 'Error generating model.');
            }
        })
        .catch(error => {
            console.error('Error:', error);
            alert('An error occurred. Please try again.');
            document.getElementById('loading-spinner').style.display = 'none';
        });
    }
    
    function pollTaskStatus(taskId, nextEndpoint) {
        // Poll the task status every few seconds
        const pollInterval = setInterval(() => {
            fetch(`/task_status/${taskId}`)
                .then(response => response.json())
                .then(data => {
                    if (data.status === 'SUCCESS') {
                        clearInterval(pollInterval);
                        if (nextEndpoint) {
                            // Start the next task
                            startNextTask(nextEndpoint);
                        } else {
                            // Final step: redirect to /view_analysis
                            window.location.href = '/view_analysis';
                        }
                    } else if (data.status === 'FAILURE') {
                        clearInterval(pollInterval);
                        alert('Simulation failed. Please try again.');
                        document.getElementById('loading-spinner').style.display = 'none';
                    }
                    // Handle other statuses if necessary
                })
                .catch(error => {
                    console.error('Error:', error);
                    clearInterval(pollInterval);
                    alert('An error occurred while checking the task status.');
                    document.getElementById('loading-spinner').style.display = 'none';
                });
        }, 15000); // Poll every 15 seconds, adjust as needed
    }
    
    function startNextTask(endpoint) {
        const formData = new FormData(form); // Reconstruct formData
        formData.append('quick_test', String(isQuickTest)); // Make sure to append quick_test
    
        // Start the next task and poll for its completion
        fetch(endpoint, { method: 'POST', body: new FormData(form) })
            .then(response => response.json())
            .then(data => {
                if (data.status === "success") {
                    // If the task is successfully started, poll for its status
                    pollTaskStatus(data.task_id);
                } else {
                    throw new Error(data.error || 'Error starting the next task.');
                }
            })
            .catch(error => {
                console.error('Error:', error);
                alert('An error occurred. Please try again.');
                document.getElementById('loading-spinner').style.display = 'none';
            });
    }    

    function showLoadingSpinner() {
        document.getElementById('loading-spinner').style.display = 'block';

        return true; // Show the spinner
    }

    // Attach the handleFormSubmission function to the form's submit event
    form.addEventListener('submit', handleFormSubmission);

    // Add event listener for the "Quick Test" button
    const quickTestButton = document.getElementById('quickTestButton');
    if (quickTestButton) {
        quickTestButton.addEventListener('click', (event) => {
            // Call handleFormSubmission with isQuickTest set to true
            handleFormSubmission(event, true);
        });
    }

    // Add event listener to clear all selections when the "Clear Selection" button is clicked
    document.getElementById('clearSelection').addEventListener('click', function () {
        document.querySelectorAll('.form-check-input').forEach(function (checkbox) {
            checkbox.checked = false;
        });
        document.getElementById('selectedColumnsList').textContent = 'None';
    });
});

document.addEventListener('DOMContentLoaded', function () {
    const searchInput = document.getElementById('searchInput');
    const columnContainers = document.querySelectorAll('.column-container');
    const noResults = document.getElementById('noResults');

    searchInput.addEventListener('input', function () {
        const query = this.value.toLowerCase().trim();

        let hasResults = false;

        columnContainers.forEach(container => {
            const label = container.querySelector('label').textContent.toLowerCase();

            if (label.includes(query)) {
                container.style.display = 'flex';
                hasResults = true;
            } else {
                container.style.display = 'none';
            }
        });

        noResults.style.display = hasResults ? 'none' : 'block';
    });
});
