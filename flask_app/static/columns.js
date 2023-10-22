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

    function handleFormSubmission(event) {
        event.preventDefault(); // Prevent default form submission
    
        // Show the loading spinner
        document.getElementById('loading-spinner').style.display = 'block';
    
        // First, send form data to /process_columns
        fetch('/process_columns', {
            method: 'POST',
            body: new FormData(form)
        })
        .then(response => response.json())
        .then(data => {
            if (data.success) {
                // If /process_columns is successful, send a request to /generate_analysis
                return fetch('/generate_analysis', {
                    method: 'POST'
                });
            } else {
                throw new Error('Error processing columns.');
            }
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

    // Attach the handleFormSubmission function to the form's submit event
    form.addEventListener('submit', handleFormSubmission);

    // Add event listener to clear all selections when the "Clear Selection" button is clicked
    document.getElementById('clearSelection').addEventListener('click', function() {
        document.querySelectorAll('.form-check-input').forEach(function(checkbox) {
            checkbox.checked = false;
        });
        document.getElementById('selectedColumnsList').textContent = 'None';
    });
});