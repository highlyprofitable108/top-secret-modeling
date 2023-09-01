# Database Schema

## Overview
This document outlines the schema of the SQLite database used in the NFL Data Analysis project.

## Tables

### games
- **id**: Unique identifier for each game.
- **date**: Date of the game.
- **home_team**: Home team's name.
- **away_team**: Away team's name.
- **venue**: Venue of the game.

### teams
- **id**: Unique identifier for each team.
- **name**: Team's name.
- **alias**: Team's alias.
- **conference**: Team's conference.
- **division**: Team's division.

### team_game_stats
- **game_id**: Identifier for the game.
- **team_id**: Identifier for the team.
- **points_scored**: Points scored by the team.
- **yards_gained**: Total yards gained by the team.
- **turnovers**: Total turnovers by the team.

(Note: More columns will be added as the project progresses)
