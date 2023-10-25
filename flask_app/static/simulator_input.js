function validateForm() {
    // Action Selection
    const randomHistoricalRadio = document.getElementById('randomHistorical');
    const nextWeekRadio = document.getElementById('nextWeek');
    const customMatchupsRadio = document.getElementById('customMatchupsRadio');

    if (!randomHistoricalRadio.checked && !nextWeekRadio.checked && !customMatchupsRadio.checked) {
        alert('Please select an action.');
        return false;
    }

    // Simulation Iterations
    const simIterations = document.getElementById('simIterations').value;
    if (simIterations && (simIterations < 1 || simIterations > 10000)) {
        alert('Number of Sim Iterations should be between 1 and 10,000.');
        return false;
    }

    // Number of Random Historical Games
    if (randomHistoricalRadio.checked) {
        const numRandomGames = document.getElementById('numRandomGames').value;
        if (!numRandomGames || numRandomGames < 1 || numRandomGames > 500) {
            alert('Number of Random Historical Games should be between 1 and 500.');
            return false;
        }
    }

    // Custom Matchups
    if (customMatchupsRadio.checked) {
        const homeTeamSelects = document.querySelectorAll('[name^="homeTeam"]');
        const awayTeamSelects = document.querySelectorAll('[name^="awayTeam"]');
        let isAllSelected = true;
        let selectedTeams = [];

        homeTeamSelects.forEach(select => {
            if (!select.value || select.value === 'None' || select.value === 'Null') isAllSelected = false;
            if (selectedTeams.includes(select.value)) {
                alert('Each team can only be selected once.');
                isAllSelected = false;
            } else if (select.value && select.value !== 'None' && select.value !== 'Null') {
                selectedTeams.push(select.value);
            }
        });

        awayTeamSelects.forEach(select => {
            if (!select.value || select.value === 'None' || select.value === 'Null') isAllSelected = false;
            if (selectedTeams.includes(select.value)) {
                alert('Each team can only be selected once.');
                isAllSelected = false;
            } else if (select.value && select.value !== 'None' && select.value !== 'Null') {
                selectedTeams.push(select.value);
            }
        });

        if (!isAllSelected) {
            return true;
        }
    }

    return true;

}

document.querySelector('form').addEventListener('submit', function(e) {
    e.preventDefault();

    if (validateForm()) {
        showLoadingSpinner();

        setTimeout(function() {
            e.target.submit();
        }, 100);
    }
});

document.addEventListener('DOMContentLoaded', function() {
    const homeTeamSelects = document.querySelectorAll('[name^="homeTeam"]');
    const awayTeamSelects = document.querySelectorAll('[name^="awayTeam"]');

    function updateTeamAvailability() {
        let selectedTeams = [];

        homeTeamSelects.forEach(select => {
            if (select.value) selectedTeams.push(select.value);
        });
        awayTeamSelects.forEach(select => {
            if (select.value) selectedTeams.push(select.value);
        });

        homeTeamSelects.forEach(select => {
            select.querySelectorAll('option').forEach(option => {
                if (selectedTeams.includes(option.value) && option.value !== select.value) {
                    option.disabled = true;
                } else {
                    option.disabled = false;
                }
            });
        });

        awayTeamSelects.forEach(select => {
            select.querySelectorAll('option').forEach(option => {
                if (selectedTeams.includes(option.value) && option.value !== select.value) {
                    option.disabled = true;
                } else {
                    option.disabled = false;
                }
            });
        });
    }

    homeTeamSelects.forEach(select => {
        select.addEventListener('change', updateTeamAvailability);
    });

    awayTeamSelects.forEach(select => {
        select.addEventListener('change', updateTeamAvailability);
    });

    // Show/Hide Custom Matchups Section based on radio button selection
    document.getElementById('customMatchupsRadio').addEventListener('change', function() {
        if (this.checked) {
            document.getElementById('customMatchupsSection').style.display = 'block';
            document.getElementById('randomHistoricalGamesSection').style.display = 'none';
        }
    });

    document.getElementById('randomHistorical').addEventListener('change', function() {
        if (this.checked) {
            document.getElementById('customMatchupsSection').style.display = 'none';
            document.getElementById('randomHistoricalGamesSection').style.display = 'block';
        }
    });

    document.getElementById('nextWeek').addEventListener('change', function() {
        if (this.checked) {
            document.getElementById('customMatchupsSection').style.display = 'none';
            document.getElementById('randomHistoricalGamesSection').style.display = 'none';
        }
    });
});

function showLoadingSpinner() {
    const numIterations = document.getElementById('simIterations').value || 1000; // Default to 1 if not provided
    const randomHistoricalRadio = document.getElementById('randomHistorical');
    const customMatchupsRadio = document.getElementById('customMatchupsRadio');
    let randomSubset = 16; // Default value

    if (randomHistoricalRadio.checked) {
        randomSubset = document.getElementById('numRandomGames').value || 1; // Default to 1 if not provided
    }

    if (customMatchupsRadio.checked) {
        let matchupCount = 0;
        for (let i = 1; i <= 16; i++) {
            const homeTeam = document.querySelector(`[name="homeTeam${i}"]`).value;
            const awayTeam = document.querySelector(`[name="awayTeam${i}"]`).value;
            
            // Check if both home and away teams are selected and they are not 'None'
            if (homeTeam && awayTeam && homeTeam !== 'None' && awayTeam !== 'None') {
                matchupCount++;
            }
        }
        randomSubset = matchupCount; // Set the randomSubset to the count of manual matchups
    }

    // Calculate timeRequired based on the formula
    let timeRequired = (numIterations * randomSubset) / 5000 * 60;
    let countdown = timeRequired

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
