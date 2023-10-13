function showTab(tabName) {
    // Hide all tab contents
    let tabContents = document.querySelectorAll('.tab-content');
    tabContents.forEach(tab => {
        tab.classList.add('hidden');
    });

    // Deactivate all tab buttons
    let tabButtons = document.querySelectorAll('.tab-button');
    tabButtons.forEach(button => {
        button.classList.remove('active');
    });

    // Show the selected tab content and activate the button
    document.getElementById(tabName).classList.remove('hidden');
    document.querySelector(`.tab-button[onclick="showTab('${tabName}')"]`).classList.add('active');
}