import os
import requests
import sqlite3
from bs4 import BeautifulSoup


# Define the base URL and directory path
BASE_URL = 'https://www.pro-football-reference.com'
DIRECTORY_PATH = '/Users/michaelfuscoletti/Desktop'

# Connect to SQLite database
db_path = os.path.join(DIRECTORY_PATH, 'nfl_data.db')
conn = sqlite3.connect(db_path)
cursor = conn.cursor()


# Create tables
def create_tables():
    # Game Info table
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS GameInfo (
        GameID INTEGER PRIMARY KEY,
        Date TEXT,
        Time TEXT,
        Stadium TEXT
    )''')

    # Team Stats table
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS TeamStats (
        StatID INTEGER PRIMARY KEY,
        GameID INTEGER,
        TeamID INTEGER,
        FirstDowns INTEGER,
        RushingFirstDowns INTEGER,
        PassingFirstDowns INTEGER,
        FirstDownsFromPenalties INTEGER,
        ThirdDownEfficiency TEXT,
        FourthDownEfficiency TEXT,
        TotalPlays INTEGER,
        TotalYards INTEGER,
        TotalDrives INTEGER,
        YardsPerPlay REAL,
        PassingYards INTEGER,
        RushingYards INTEGER,
        RedZoneAttempts INTEGER,
        Penalties INTEGER,
        Turnovers INTEGER,
        DefensiveSpecialTeamsTDs INTEGER,
        PossessionTime TEXT
    )''')

    # Passing, Rushing, & Receiving table
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS PlayerStats (
        StatID INTEGER PRIMARY KEY,
        GameID INTEGER,
        PlayerID INTEGER,
        FantasyPoints REAL,
        PassingAttempts INTEGER,
        PassingCompletions INTEGER,
        PassingYards INTEGER,
        PassingTDs INTEGER,
        Interceptions INTEGER,
        RushingAttempts INTEGER,
        RushingYards INTEGER,
        RushingTDs INTEGER,
        Targets INTEGER,
        Receptions INTEGER,
        ReceivingYards INTEGER,
        ReceivingTDs INTEGER
    )''')

    # Advanced Passing table
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS AdvancedPassing (
        StatID INTEGER PRIMARY KEY,
        GameID INTEGER,
        PlayerID INTEGER,
        TimesSacked INTEGER,
        YardsLostToSacks INTEGER,
        QBRating REAL,
        AirYards INTEGER,
        YardsAfterCatch INTEGER,
        TimesHit INTEGER,
        ThrowAways INTEGER,
        AggressiveThrows INTEGER,
        TightWindowThrows INTEGER,
        RedZoneAttempts INTEGER
    )''')

    # Advanced Rushing table
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS AdvancedRushing (
        StatID INTEGER PRIMARY KEY,
        GameID INTEGER,
        PlayerID INTEGER,
        BrokenTackles INTEGER,
        YardsAfterContact INTEGER,
        RushingFirstDowns INTEGER,
        RushingTDsInside20 INTEGER,
        RushingTDsInside10 INTEGER,
        RushingTDsInside5 INTEGER
    )''')

    # Advanced Receiving table
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS AdvancedReceiving (
        StatID INTEGER PRIMARY KEY,
        GameID INTEGER,
        PlayerID INTEGER,
        AirYards INTEGER,
        YardsAfterCatch INTEGER,
        BrokenTackles INTEGER,
        Drops INTEGER,
        TightWindowReceptions INTEGER,
        ReceivingFirstDowns INTEGER,
        ReceivingTDsInside20 INTEGER,
        ReceivingTDsInside10 INTEGER,
        ReceivingTDsInside5 INTEGER
    )''')

    # Snap Counts table
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS SnapCounts (
        StatID INTEGER PRIMARY KEY,
        GameID INTEGER,
        PlayerID INTEGER,
        OffensiveSnaps INTEGER,
        DefensiveSnaps INTEGER,
        SpecialTeamsSnaps INTEGER
    )''')

    # Commit the changes
    conn.commit()


# Call the function to create tables
create_tables()


def extract_game_info(url):
    response = requests.get(url)
    soup = BeautifulSoup(response.content, 'html.parser')

    # Extracting data
    date_time = soup.find('div', {'class': 'scorebox_meta'}).find('div').text
    date, time = date_time.split(', ', 1)
    stadium = soup.find(
        'div', {'class': 'scorebox_meta'}
    ).find_all('div')[1].text

    # Storing data in a dictionary
    game_details = {
        'Date': date,
        'Time': time,
        'Stadium': stadium,
    }

    return game_details


# Sample usage
url = "https://www.pro-football-reference.com/boxscores/202011010cin.htm"
print(extract_game_info(url))


def extract_team_stats(url):
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    }
    response = requests.get(url, headers=headers)
    soup = BeautifulSoup(response.content, 'html.parser')
    
    # Find the team stats table
    stats_div = soup.find('div', {'id': 'div_team_stats'})
    
    if not stats_div:
        return "Team statistics div not found on the webpage."
    
    stats_table = stats_div.find('table', {'id': 'team_stats'})
    
    if not stats_table:
        return "Team statistics table not found within the div."
    
    # Extract header (team names)
    header = [th.text for th in stats_table.find('thead').find_all('th') if th.text]
    
    # Extract stats data
    stats_data = {}
    for row in stats_table.find('tbody').find_all('tr'):
        stat_name = row.find('th').text
        stats_values = [td.text for td in row.find_all('td')]
        
        stats_data[stat_name] = {
            header[0]: stats_values[0],
            header[1]: stats_values[1]
        }
    
    return stats_data

url = "https://www.pro-football-reference.com/boxscores/202009130nor.htm"
print(extract_team_stats(url))


"""
def extract_game_info(soup):
    game_info = {}
    game_info_div = soup.find('div', {'class': 'scorebox_meta'})
    details = game_info_div.find_all('div')
    game_info['date'] = details[0].text
    game_info['stadium'] = details[1].text
    game_info['attendance'] = details[2].text.split()[0].replace(',', '')
    game_info['weather'] = details[3].text if len(details) > 3 else None
    return game_info


def extract_team_stats(soup):
    team_stats = {}
    stats_table = soup.find('table', {'id': 'team_stats'})
    for row in stats_table.find_all('tr')[1:]:
        columns = row.find_all('td')
        stat_name = columns[0].text
        away_stat = columns[1].text
        home_stat = columns[2].text
        team_stats[f'away_{stat_name}'] = away_stat
        team_stats[f'home_{stat_name}'] = home_stat
    return team_stats


# Test the functions
response = requests.get(
    'https://www.pro-football-reference.com/boxscores/202009100kan.htm'
)
soup = BeautifulSoup(response.content, 'html.parser')

print(extract_game_info(soup))
print(extract_team_stats(soup))


def extract_advanced_passing(soup):
    passing_data = []
    table = soup.find('table', {'id': 'player_advanced_passing'})
    for row in table.tbody.find_all('tr', class_=lambda x: x != 'thead'):
        player_data = {}
        for td in row.find_all('td'):
            player_data[td['data-stat']] = td.text
        passing_data.append(player_data)
    return passing_data


def extract_advanced_rushing(soup):
    rushing_data = []
    table = soup.find('table', {'id': 'player_advanced_rushing'})
    for row in table.tbody.find_all('tr', class_=lambda x: x != 'thead'):
        player_data = {}
        for td in row.find_all('td'):
            player_data[td['data-stat']] = td.text
        rushing_data.append(player_data)
    return rushing_data


def extract_advanced_receiving(soup):
    receiving_data = []
    table = soup.find('table', {'id': 'player_advanced_receiving'})
    for row in table.tbody.find_all('tr', class_=lambda x: x != 'thead'):
        player_data = {}
        for td in row.find_all('td'):
            player_data[td['data-stat']] = td.text
        receiving_data.append(player_data)
    return receiving_data


# Test the functions
response = requests.get(
    'https://www.pro-football-reference.com/boxscores/202009100kan.htm'
)
soup = BeautifulSoup(response.content, 'html.parser')

print(extract_advanced_passing(soup))
print(extract_advanced_rushing(soup))
print(extract_advanced_receiving(soup))


def extract_snap_counts(soup, team='home'):
    snap_counts_data = []
    table_id = f'snap_counts_{team}'
    table = soup.find('table', {'id': table_id})
    for row in table.tbody.find_all('tr', class_=lambda x: x != 'thead'):
        player_data = {}
        for td in row.find_all('td'):
            player_data[td['data-stat']] = td.text
        snap_counts_data.append(player_data)
    return snap_counts_data


# Test the functions
response = requests.get(
    'https://www.pro-football-reference.com/boxscores/202009100kan.htm'
)
soup = BeautifulSoup(response.content, 'html.parser')

print('Home Team Snap Counts:')
print(extract_snap_counts(soup, 'home'))
print('\nAway Team Snap Counts:')
print(extract_snap_counts(soup, 'away'))


def insert_game_info(game_info):
    cursor.execute('''
    INSERT OR IGNORE INTO Games (Date, Stadium, Attendance, Weather)
    VALUES (?, ?, ?, ?)
    ''', (
        game_info['date'],
        game_info['stadium'],
        game_info['attendance'],
        game_info['weather']
    ))
    conn.commit()


def insert_team_stats(team_stats):
    # Assuming game_id is available after inserting game info
    game_id = cursor.lastrowid
    cursor.execute('''
    INSERT OR IGNORE INTO TeamStats
        (GameID, TotalYards, PassingYards, RushingYards, Turnovers)
    VALUES (?, ?, ?, ?, ?)
    ''', (
        game_id,
        team_stats['away_total_yds'],
        team_stats['away_pass_yds'],
        team_stats['away_rush_yds'],
        team_stats['away_turnovers']
    ))
    cursor.execute('''
    INSERT OR IGNORE INTO TeamStats
        (GameID, TotalYards, PassingYards, RushingYards, Turnovers)
    VALUES (?, ?, ?, ?, ?)
    ''', (
        game_id,
        team_stats['home_total_yds'],
        team_stats['home_pass_yds'],
        team_stats['home_rush_yds'],
        team_stats['home_turnovers']
    ))
    conn.commit()


# Test the integration and insertion
response = requests.get(
    'https://www.pro-football-reference.com/boxscores/202009100kan.htm'
)
soup = BeautifulSoup(response.content, 'html.parser')

# Extract and insert game info
game_info = extract_game_info(soup)
insert_game_info(game_info)

# Extract and insert team stats
team_stats = extract_team_stats(soup)
insert_team_stats(team_stats)


def insert_advanced_passing(passing_data):
    game_id = cursor.lastrowid
    for player in passing_data:
        # Assuming player_id is available after inserting into Players table
        player_id = cursor.execute(
            'SELECT PlayerID FROM Players WHERE PlayerName = ?', (
                player['player'],
            )
        ).fetchone()[0]
        cursor.execute('''
        INSERT OR IGNORE INTO AdvancedPassing (GameID, PlayerID, ...)
        VALUES (?, ?, ...)
        ''', (game_id, player_id, ...))  # Add other columns and values
    conn.commit()


def insert_advanced_rushing(rushing_data):
    game_id = cursor.lastrowid
    for player in rushing_data:
        player_id = cursor.execute(
            'SELECT PlayerID FROM Players WHERE PlayerName = ?', (
                player['player'],
            )
        ).fetchone()[0]
        cursor.execute('''
        INSERT OR IGNORE INTO AdvancedRushing (GameID, PlayerID, ...)
        VALUES (?, ?, ...)
        ''', (game_id, player_id, ...))  # Add other columns and values
    conn.commit()


def insert_advanced_receiving(receiving_data):
    game_id = cursor.lastrowid
    for player in receiving_data:
        player_id = cursor.execute(
            'SELECT PlayerID FROM Players WHERE PlayerName = ?', (
                player['player'],
            )
        ).fetchone()[0]
        cursor.execute('''
        INSERT OR IGNORE INTO AdvancedReceiving (GameID, PlayerID, ...)
        VALUES (?, ?, ...)
        ''', (game_id, player_id, ...))  # Add other columns and values
    conn.commit()


def insert_snap_counts(snap_counts_data, team):
    game_id = cursor.lastrowid
    for player in snap_counts_data:
        player_id = cursor.execute(
            'SELECT PlayerID FROM Players WHERE PlayerName = ?', (
                player['player'],
            )
        ).fetchone()[0]
        cursor.execute('''
        INSERT OR IGNORE INTO SnapCounts (GameID, PlayerID, Team, ...)
        VALUES (?, ?, ?, ...)
        ''', (game_id, player_id, team, ...))  # Add other columns and values
    conn.commit()


# Test the integration and insertion
response = requests.get(
    'https://www.pro-football-reference.com/boxscores/202009100kan.htm'
)
soup = BeautifulSoup(response.content, 'html.parser')

# Extract and insert advanced passing
advanced_passing = extract_advanced_passing(soup)
insert_advanced_passing(advanced_passing)

# Extract and insert advanced rushing
advanced_rushing = extract_advanced_rushing(soup)
insert_advanced_rushing(advanced_rushing)

# Extract and insert advanced receiving
advanced_receiving = extract_advanced_receiving(soup)
insert_advanced_receiving(advanced_receiving)

# Extract and insert snap counts for home and away teams
snap_counts_home = extract_snap_counts(soup, 'home')
insert_snap_counts(snap_counts_home, 'home')
snap_counts_away = extract_snap_counts(soup, 'away')
insert_snap_counts(snap_counts_away, 'away')


def extract_boxscore_data(season, boxscore_url):
    try:
        print(f'Extracting data for {boxscore_url}')
        response = requests.get(boxscore_url)
        soup = BeautifulSoup(response.content, 'html.parser')

        # Extract game date
        game_date_section = soup.find('div', {'class': 'scorebox_meta'})
        game_date = game_date_section.find(
            'div'
        ).text if game_date_section else None

        # Extract player stats from the 'Passing, Rushing, & Receiving Table'
        player_stats_table = soup.find('table', {'id': 'player_offense'})
        if player_stats_table:
            rows = player_stats_table.find_all('tr')
            for row in rows:
                columns = row.find_all('td')
                if columns:
                    player_name_element = row.find('th')
                    player_id = player_name_element[
                        'data-append-csv'
                    ] if player_name_element else 'Unknown'
                    player_name = player_name_element.find(
                        'a'
                    ).text if player_name_element else 'Unknown Player'
                    team = columns[0].text
                    passing_attempts = int(
                        columns[2].text
                    ) if columns[2].text else 0
                    passing_yards = int(
                        columns[3].text
                    ) if columns[3].text else 0
                    passing_tds = int(
                        columns[4].text
                    ) if columns[4].text else 0
                    interceptions = int(
                        columns[5].text
                    ) if columns[5].text else 0
                    rushing_attempts = int(
                        columns[10].text
                    ) if columns[10].text else 0
                    rushing_yards = int(
                        columns[11].text
                    ) if columns[11].text else 0
                    rushing_tds = int(
                        columns[12].text
                    ) if columns[12].text else 0
                    targets = int(
                        columns[14].text
                    ) if columns[14].text else 0
                    receiving_yards = int(
                        columns[16].text
                    ) if columns[16].text else 0
                    receiving_tds = int(
                        columns[17].text
                    ) if columns[17].text else 0
                    fumbles_lost = int(
                        columns[20].text
                    ) if columns[20].text else 0

                    # Insert extracted data into the SQLite database
                    cursor.execute(
                        '''INSERT INTO player_stats (season, game_date,
                        player_id, player_name, team, passing_attempts,
                        passing_yards, passing_tds, interceptions,
                        rushing_attempts, rushing_yards, rushing_tds,
                        targets, receiving_yards, receiving_tds, fumbles_lost)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        ''',
                        (season, game_date, player_id, player_name, team,
                         passing_attempts, passing_yards,
                         passing_tds, interceptions,
                         rushing_attempts, rushing_yards, rushing_tds,
                         targets, receiving_yards, receiving_tds, fumbles_lost)
                    )
        conn.commit()

        # New extraction and insertion logic
        game_info = extract_game_info(soup)
        insert_game_info(game_info)

        team_stats = extract_team_stats(soup)
        insert_team_stats(team_stats)

        print(f'Data extracted and saved for {boxscore_url}')
        time.sleep(5)  # Introduce a delay after each boxscore data extraction
    except Exception as e:
        time.sleep(5)  # Introduce a delay after each boxscore data extraction
        print(f'Error occurred while processing {boxscore_url}: {e}')


# Loop through the 2020, 2021, and 2022 NFL seasons
for season in [2020, 2021, 2022]:
    print(f'Processing season {season}')
    response = requests.get(
        f'https://www.pro-football-reference.com/years/{season}/games.htm'
    )
    soup = BeautifulSoup(response.content, 'html.parser')
    boxscore_links = [
        link['href'] for link in soup.find_all(
            'a', href=True
        ) if 'boxscore' in link.text
    ]
    print(f'Found {len(boxscore_links)} boxscore links for season {season}')

    # Extract data for each boxscore
    for link in boxscore_links:
        boxscore_url = BASE_URL + link
        extract_boxscore_data(season, boxscore_url)

# Close the SQLite database connection
conn.close()
print('Script completed.')
"""
