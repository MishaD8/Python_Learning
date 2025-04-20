import os
from datetime import datetime, timedelta
import csv 

from Project_soccer.src.utils import fetch_api_data, logger
from Project_soccer.src.data_collection import process_fixture, process_fixtures
from config import LEAGUE_ID, LAST_UPDATE_FILE, OUTPUT_FILE

def update_recent_fixtures(days_back=7):
    """Fetch and process only recent fixtures from the last n days"""

    # Load the last update timestamp or create it
    today = datetime.now()
    from_date = None

    if os.path.exists(LAST_UPDATE_FILE):
        with open(LAST_UPDATE_FILE, 'r') as f:
            last_update_str = f.read().strip()
            if last_update_str:
                try:
                    last_update = datetime.fromisoformat(last_update_str)
                    from_date = last_update.strftime("%Y-%m-%d")
                except ValueError:
                    logger.error(f"Invalid date format in {LAST_UPDATE_FILE}")

    if not from_date:
        # In no last update file, get fixtures from last days_back days
        from_date = (today - timedelta(days=days_back)).strftime("%Y-%m-%d")

    to_date = today.strftime("%Y-%m-%d")

    logger.info(f"Fetching fixtures from {from_date} to {to_date}")

    # Get fixtures for the date range
    url = f"https://v3.footbal.api-sports.io/fixtures?league={LEAGUE_ID}&from={from_date}&to={to_date}&status=FT"
    data = fetch_api_data(url)
    fixtures = data.get('response', [])

    logger.info(f"Found {len(fixtures)} recent fixtures to process")

    # Process fixtures
    processed_count = process_fixtures(fixtures)

    # Update the last update timestamp
    with open(LAST_UPDATE_FILE, 'w') as f:
        f.write(today.isoformat())

    logger.info(f"Update complete. Processed {processed_count} new fixtures. Last update timestamp set to {today.isoformat()}") 
    return processed_count

if __name__ == "__main__":
    logger.info("Starting La Liga incremental update")
    update_recent_fixtures()
    logger.info("Incremental update complete")