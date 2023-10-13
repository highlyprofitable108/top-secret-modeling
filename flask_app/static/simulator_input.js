function showLoadingSpinner() {
    document.getElementById('loading-spinner').style.display = 'block';
}
        
document.getElementById('homeTeam').addEventListener('change', function() {
    const awayTeamSelect = document.getElementById('awayTeam');
    for (let i = 0; i < awayTeamSelect.options.length; i++) {
        awayTeamSelect.options[i].disabled = false;
        if (awayTeamSelect.options[i].value === this.value) {
            awayTeamSelect.options[i].disabled = true;
        }
    }
});

document.getElementById('awayTeam').addEventListener('change', function() {
    const homeTeamSelect = document.getElementById('homeTeam');
    for (let i = 0; i < homeTeamSelect.options.length; i++) {
        homeTeamSelect.options[i].disabled = false;
        if (homeTeamSelect.options[i].value === this.value) {
            homeTeamSelect.options[i].disabled = true;
        }
    }
});

document.querySelector('form').addEventListener('submit', function(e) {
    const homeTeam = document.getElementById('homeTeam').value;
    const awayTeam = document.getElementById('awayTeam').value;

    if (!homeTeam || !awayTeam) {
        e.preventDefault();
        alert('Please fill out all fields before submitting.');
    }
});
