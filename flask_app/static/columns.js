let globalIsQuickTest = false;  // Global variable to store the state of isQuickTest

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
        globalIsQuickTest = isQuickTest;  // Set the global variable
        event.preventDefault(); // Prevent default form submission
    
        // Check if any checkboxes are selected
        const checkboxesSelected = Array.from(checkboxes).some(checkbox => checkbox.checked);
        if (!checkboxesSelected) {
            alert('Please select at least one column before running the simulation.');
            return; // Stop the function if no checkboxes are selected
        }
    
        // Prepare the data to be sent to the server
        const formData = new FormData(form);
        formData.append('quick_test', String(isQuickTest));
    
        // Send form data to /process_columns
        fetch('/process_columns', {
            method: 'POST',
            body: formData
        })
        .then(response => response.json())
        .then(data => {
            if (data.success) {
                // If /process_columns is successful, send a request to /execute_combined_task
                return fetch('/execute_combined_task', { 
                    method: 'POST', 
                    body: formData 
                });
            } else {
                throw new Error('Error processing columns.');
            }
        })
        .then(response => response.json())
        .then(data => {
            if (data.status === "success") {
                // If /execute_combined_task is successful, poll for task status
                window.location.href = `/waiting?task_id=${data.task_id}`;

            } else {
                throw new Error(data.error || 'Error executing combined task.');
            }
        })
        .catch(error => {
            console.error('Error:', error);
            alert('An error occurred. Please try again.');
        });
    }

    // Attach the handleFormSubmission function to the form's submit event
    form.addEventListener('submit', handleFormSubmission);

    // Add event listener for the "Quick Test" button
    const quickTestButton = document.getElementById('quickTestButton');
    if (quickTestButton) {
        quickTestButton.addEventListener('click', (event) => {
            // Set the value of the quick_test input field to 'true' when Quick Test is clicked
            document.getElementById('quick_test_input').value = 'true';

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
