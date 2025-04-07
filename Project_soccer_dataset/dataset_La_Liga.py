import requests
import csv
import time

API_KEY = '2cd39e52f3a713cb97b4f32db562ce3d'

headers = {
    "x-apisports-key": API_KEY
}

LEAGUE_ID = 140 # La Liga
SEASONS = [2019, 2020, 2021, 2022, 2023, 2024]

with open('laliga_dataset_2019_2024.csv', 'w', newline='', encoding='utf-8') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow([
        'Season', 'Date', 'Round',
        'Home Team', 'Away Team',
        'Goals Home', 'Goals Away',
        'xG Home', 'xG Away', 
        'Corners Home', 'Corners Away',
        'Yellow Cards Home', 'Yellow Cards Away', 
        'Red Cards Home', 'Red Cards Away', 
        'Winner', 
        'Top Score 1', 'Top Score 2', 'Top Score 3', 
        'Top Score 4', 'Top Score 5'
    ])

    for season in SEASONS:
        print(f'Fetching season {season}...')
        url = f"https://v3.football.api-sports.io/fixtures?league={LEAGUE_ID}&season={season}&status=FT"
        response = requests.get(url, headers=headers)
        fixtures = response.json().get('response', [])

        for match in fixtures:
            fixture_id = match['fixture']['id']
            fixture_date = match['fixture']['date']
            round_name = match['league'].get('round', 'N/A')

            home_team = match['teams']['home']['name']
            away_team = match['teams']['away']['name']
            goals_home = match['goals']['home']
            goals_away = match['goals']['away']

            winner = (
                home_team if match['teams']['home']['winner']
                else away_team if match['teams']['away']['winner']
                else 'Draw'
            )


            # Get match statistics (xG, cards, corners)
            stats_url = f"https://v3.football.api-sports.io/fixtures/statistics?fixture={fixture_id}"
            stats_response = requests.get(stats_url, headers=headers)
            stats_data = stats_response.json().get('response', [])

            xg_home = xg_away = 0
            corners_home = corners_away = 0
            yellow_home = yellow_away = red_home = red_away = 0

            for team_stats in stats_data:
                team = team_stats['team']['name']
                for item in team_stats['statistics']:
                    if item['type'] == 'Expected Goals' and item['value'] is not None:
                        if team == home_team:
                            xg_home = item['value']
                        else:
                            xg_away - item['value']
                    elif item['type'] == 'Corner Kicks':
                        if team == home_team:
                            corners_home = item['value']
                        else:
                            corners_away = item['value']
                    elif item['type'] == 'Yellow Cards':
                        if team == home_team:
                            yellow_home = item['value']
                        else:
                            yellow_away = item['value']
                    elif item['type'] == 'Red Cards':
                        if team == home_team:
                            red_home = item['value']
                        else:
                            red_away = item['value']


            # Get odds for correct score from Bet365
            odds_url = f"https://v3.football.api-sports.io/odds?fixture={fixture_id}"
            odds_response = requests.get(odds_url, headers=headers)
            odds_data = odds_response.json().get('response', [])

            top_scores = []
            if odds_data:
                bookmakers = odds_data[0].get('bookmakers', [])
                for bookmaker in bookmakers:
                    if bookmaker['name'].lower() == 'bet365':
                        bets = bookmaker.get('bets', [])
                        for bet in bets:
                            if bet in bets:
                                if bet['name'].lower() == 'correct score':
                                    values = bet.get('values', [])
                                    sorted_bets = sorted(values, key=lambda x: float(x['odd']))
                                    top_scores = [f"{b['value']} ({b['odd']})" for b in sorted_bets[:5]]
                                    break
                        break
            while len(top_scores) < 5:
                top_scores.append("N/A")

            writer.writerow([
                season, fixture_date, round_name,
                home_team, away_team, 
                goals_home, goals_away, 
                xg_home, xg_away,
                corners_home, corners_away,
                yellow_home, yellow_away,
                red_home, red_away,
                winner
                
            ] +top_scores)

            time.sleep(1.2)