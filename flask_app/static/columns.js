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
        
        // Call the showLoadingSpinner function
        showLoadingSpinner();
    
        // Prepare the data to be sent to the server
        const formData = new FormData(form);
        formData.append('quick_test', isQuickTest); // Add the quick_test parameter to the form data
    
        // First, send form data to /process_columns
        fetch('/process_columns', {
            method: 'POST',
            body: formData
        })
        .then(response => response.json())
        .then(data => {
            if (data.success) {
                // If /process_columns is successful, send a request to /generate_model
                return fetch('/generate_model', {
                    method: 'POST'
                });
            } else {
                throw new Error('Error processing columns.');
            }
        })
        .then(response => response.json())
        .then(data => {
            if (data.status === "success") {
                // If /generate_power_ranks is successful, send a request to /sim_runner
                return fetch('/sim_runner', {
                    method: 'POST',
                    body: formData // Send the updated formData with the quick_test parameter
                });
            } else {
                throw new Error(data.error || 'Error generating power ranks.');
            }
        })
        .then(() => {
            // After /sim_runner call is successful, redirect to /view_analysis
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
        let countdown = 900;

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

    // Add event listener for the "Quick Test" button
    const quickTestButton = document.getElementById('quickTestButton');
    if (quickTestButton) {
        quickTestButton.addEventListener('click', (event) => {
            // Call handleFormSubmission with isQuickTest set to true
            handleFormSubmission(event, true);
        });
    }

    // Add event listener to clear all selections when the "Clear Selection" button is clicked
    document.getElementById('clearSelection').addEventListener('click', function() {
        document.querySelectorAll('.form-check-input').forEach(function(checkbox) {
            checkbox.checked = false;
        });
        document.getElementById('selectedColumnsList').textContent = 'None';
    });
});

document.addEventListener('DOMContentLoaded', function() {
    const searchInput = document.getElementById('searchInput');
    const columnContainers = document.querySelectorAll('.column-container');
    const noResults = document.getElementById('noResults');

    searchInput.addEventListener('input', function() {
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
