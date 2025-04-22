import os

# API Configuration
API_KEY = os.environ.get('API_SPORTS_KEY', '2cd39e52f3a713cb97b4f32db562ce3d')
RATE_LIMIT_SLEEP = 3 # Seconds between API calls

# Data Configuration
LEAGUE_ID = 140 #La Liga
SEASONS = [2019, 2020, 2021, 2022, 2023, 2024]
BASE_DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data')
OUTPUT_FILE = os.path.join(BASE_DATA_DIR, 'laliga_dataset_2019_2024.csv')
PROCESSED_DIR = os.path.join(BASE_DATA_DIR, 'processed')

# Ensure directories exist
os.makedirs(BASE_DATA_DIR, exist_ok=True)
os.makedirs(PROCESSED_DIR, exist_ok=True)

# Logging Configuration
LOG_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'logs')
os.makedirs(LOG_DIR, exist_ok=True)
LOG_FILE = os.path.join(LOG_DIR, 'data_collection.log')

# API Headers
HEADERS = {
    "x-apisports-key": API_KEY
}

# Score patterns we're interested in predicting
SCORE_PATTERNS = [
    {'home_score': 2, 'away_score': 1, 'name': '2-1'},
    {'home_score': 1, 'away_score': 1, 'name': '1-1'},
    {'home_score': 1, 'away_score': 2, 'name': '1-2'}
]

LAST_UPDATE_FILE = os.path.join(BASE_DATA_DIR, 'last_update.txt')