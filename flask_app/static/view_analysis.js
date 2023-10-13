function showTab(tabName) {
    // Hide all tab contents
    let tabContents = document.querySelectorAll('.tab-content');
    tabContents.forEach(tab => {
        tab.classList.add('hidden');
    });

    // Reset all tab buttons to default style
    let tabButtons = document.querySelectorAll('.tab-button');
    tabButtons.forEach(button => {
        button.classList.remove('active');
        button.classList.remove('bg-blue-500');
        button.classList.add('bg-blue-200');
        button.classList.remove('text-white');
        button.classList.add('text-gray-700');
    });

    // Show the selected tab content and activate the button
    document.getElementById(tabName).classList.remove('hidden');
    const activeButton = document.querySelector(`.tab-button[onclick="showTab('${tabName}')"]`);
    activeButton.classList.add('active');
    activeButton.classList.add('bg-blue-500');
    activeButton.classList.remove('bg-blue-200');
    activeButton.classList.add('text-white');
    activeButton.classList.remove('text-gray-700');

    // If the heatmap tab is being shown, trigger a resize for the Plotly chart
    if (contentId === 'interactiveHeatmapContent' && typeof Plotly !== 'undefined') {
        var heatmapDiv = document.getElementById('interactiveHeatmapContent'); // Replace with the actual ID of your Plotly div
        Plotly.Plots.resize(heatmapDiv);
    }
}
