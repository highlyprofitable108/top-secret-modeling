import requests


class APIHandler:
    def __init__(self, base_url, api_key):
        self.base_url = base_url
        self.api_key = api_key
        self.headers = {
            "Accept": "application/json",
            "Content-Type": "application/json"
        }

    def fetch_game_schedule(self, year, season, endpoint):
        url = f"{self.base_url}{endpoint.format(year=year, season=season)}?api_key={self.api_key}"
        response = requests.get(url, headers=self.headers)
        return response.json()

    def fetch_game_statistics(self, game_id, endpoint):
        url = f"{self.base_url}{endpoint.format(game_id=game_id)}?api_key={self.api_key}"
        response = requests.get(url, headers=self.headers)
        return response.json()
