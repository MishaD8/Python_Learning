import csv
import time 
import os
from datetime import datetime
import logging

from Project_soccer.src.utils import fetch_api_data, safe_get_value, format_date, logger
from config import LEAGUE_ID, SEASONS, OUTPUT_FILE, RATE_LIMIT_SLEEP

# Check if output file exists to enable resuming
def get_processed_fixtures():
    processed = set()
    if os.path.exists(OUTPUT_FILE):
        try:
            with open(OUTPUT_FILE, 'r', encoding='utf-8') as f:
                reader = csv.reader(f)
                next(reader) # Skip header
                for row in reader:
                    # Create a unique identifier for each fixture
                    if len(row) >= 6: # Ensure we have enough columns
                        fixture_id = f"{row[0]}_{row[3]}_{row[4]}_{row[1]}"
                        processed.add(fixture_id)
            logger.info(f"Found {len(processed)} previously processed fixtures")
        except Exception as e:
            logger.error(f"Error reading existing file: {e}")
    return processed

def process_fixture(match):
    """Process individual fixture data"""
    try:
        fixture_id = match['fixture']['id']
        fixture_date = format_date(match['fixture']['date'])
        round_name = match['league'].get('round', 'N/A')

        home_team = match['teams']['home']['name']
        away_team = match['teams']['away']['name']

        season = match['league']['season']

        # Create a unique identifier for this fixture
        fixture_unique_id = f"{season}_{home_team}_{away_team}_{fixture_date}"

        goals_home = match['goals']['home']
        goals_away = match['goals']['away']

        winner = (
            home_team if match['teams']['home']['winner']
            else away_team if match['teams']['away']['winner']
            else 'Draw'
        )

        # Get match statistics (xG, cards, corners)
        logger.info(f"Fetching stats for {home_team} vs {away_team}")
        stats_url = f"https://v3.football.api-sports.io/fixtures/statistics?fixture={fixture_id}"
        stats_data = fetch_api_data(stats_url)
        stats_response = stats_data.get('response', [])

        xg_home = xg_away = 0
        corners_home = corners_away = 0
        yellow_home = yellow_away = 0
        red_home = red_away = 0

        for team_stats in stats_response:
            team = team_stats['team']['name']
            stats_list = team_stats.get('statistics, []')

            if team == home_team:
                xg_home = safe_get_value(stats_list, 'Expected Goals')
                corners_home = safe_get_value(stats_list, 'Corner Kicks')
                yellow_home = safe_get_value(stats_list, 'Yellow Cards')
                red_home = safe_get_value(stats_list, 'Red Cards')
            else:
                xg_away = safe_get_value(stats_list, 'Expected Goals')
                corners_away = safe_get_value(stats_list, 'Corner Kicks')
                yellow_away = safe_get_value(stats_list, 'Yellow Cards')
                red_away = safe_get_value(stats_list, 'Red Cards')

        # Get odds for correct score
        logger.info(f"Fetching odds for {home_team} vs {away_team}")
        odds_url = f"https://v3.football.api-sports.io/odds?fixture={fixture_id}"
        odds_data = fetch_api_data(odds_url)
        odds_response = odds_data.get('response', [])

        top_scores = []
        if odds_response:
            bookmakers = odds_response[0].get('bookmakers', [])
            for bookmaker in bookmakers:
                if bookmaker['name'].lower() == 'bet365':
                    bets = bookmaker.get('bets', [])
                    for bet in bets:
                        if bet['name'].lower() == 'correct score':
                            values = bet.get('values', [])
                            # Sort by odds (lower odds = higher probability)
                            sorted_bets = sorted(values, key=lambda x: float(x['odd']))
                            top_scores = [f"{b['value']} ({b['odd']})" for b in sorted_bets[:5]]
                            break
                    break
            
        # Ensure we have 5 values for top scores
        while len(top_scores) < 5:
            top_scores.append("N/A")

        row_data = [
            season, fixture_date, round_name,
            home_team, away_team,
            goals_home, goals_away,
            xg_home, xg_away,
            corners_home, corners_away,
            yellow_home, yellow_away,
            red_home, red_away,
            winner
        ] + top_scores

        # Sleep to respect API rate limits
        time.sleep(RATE_LIMIT_SLEEP)

        return fixture_unique_id, row_data
    
    except Exception as e:
        logger.error(f"Error processing fixture {fixture_id}: {e}")
        return None, None
    
def process_fixtures(fixtures, processed_fixtures=None):
    """Process a list of fixtures and write to CSV file"""
    if processed_fixtures is None:
        processed_fixtures = get_processed_fixtures()

    file_exists = os.path.exists(OUTPUT_FILE)
    mode = 'a' if file_exists else 'w'

    processed_count = 0

    with open(OUTPUT_FILE, mode, newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)

        # Write header only if creating a new file
        if not file_exists:
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

        for match in fixtures:
            fixture_unigue_id, row_data = process_fixture(match)

            if not fixture_unigue_id or not row_data:
                continue

            # Skip if already processed
            if fixture_unigue_id in processed_fixtures:
                logger.info(f"Skipping already processed fixture: {row_data[3]} vs {row_data[4]} ({row_data[1]})")
                continue

            writer.writerow(row_data)
            processed_count += 1
            logger.info(f"Processed fixture: {row_data[3]} {row_data[5]}-{row_data[6]} {row_data[4]}")
            csvfile.flush() # Ensure data is written to disk after each fixture

    return processed_count

def main():
    """Main function to collect full seasons data"""
    # Check for existing data to enable resuming
    processed_fixtures = get_processed_fixtures()

    total_fixtures = 0
    processed_count = 0

    for season in SEASONS:
        logger.info(f'Fetching season {season}...')
        url = f"https://v3.football.api-sports.io/fixtures?league={LEAGUE_ID}&season={season}&status=FT"
        data = fetch_api_data(url)
        fixtures = data.get('response', [])
        total_fixtures += len(fixtures)

        logger.info(f"Found {len(fixtures)} fixtures for season {season}")

        season_processed = process_fixture(fixtures, processed_fixtures)
        processed_count += season_processed

    logger.info(f"Data collection complete. Processed {processed_count}/{total_fixtures} fixtures across {len(SEASONS)} seasons")

if __name__ == "__main__":
    logger.info("Starting La Liga data collection script")
    main()
    logger.info("Script execution complete")