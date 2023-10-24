function showLoadingSpinner() {
    const customMatchupsRadio = document.getElementById('customMatchupsRadio');
    if (customMatchupsRadio.checked) {
        const homeTeamSelects = document.querySelectorAll('[name^="homeTeam"]');
        const awayTeamSelects = document.querySelectorAll('[name^="awayTeam"]');
        let isAllSelected = true;

        homeTeamSelects.forEach(select => {
            if (!select.value) isAllSelected = false;
        });
        awayTeamSelects.forEach(select => {
            if (!select.value) isAllSelected = false;
        });

        if (!isAllSelected) {
            alert('Please fill out all fields before submitting.');
            return false; // Do not show the spinner
        }
    }
    document.getElementById('loading-spinner').style.display = 'block';
    return true; // Show the spinner
}

document.querySelector('form').addEventListener('submit', function(e) {
    if (!showLoadingSpinner()) {
        e.preventDefault(); // Prevent form submission if showLoadingSpinner returns false
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
        }
    });

    document.getElementById('randomHistorical').addEventListener('change', function() {
        if (this.checked) {
            document.getElementById('customMatchupsSection').style.display = 'none';
        }
    });

    document.getElementById('nextWeek').addEventListener('change', function() {
        if (this.checked) {
            document.getElementById('customMatchupsSection').style.display = 'none';
        }
    });
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

