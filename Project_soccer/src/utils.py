import requests
import logging
import time
import sys
from datetime import datetime
from config import HEADERS, LOG_FILE

# Setup logging
def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(LOG_FILE),
            logging.StreamHandler(sys.stdout)
        ]
    )
    return logging.getLogger('soccer_prediction')

logger = setup_logging()

def fetch_api_data(url):
    """Fetch data from API with error handling and retries"""
    max_retries = 3
    for attempt in range(max_retries):
        try:
            logger.info(f"Fetching: {url}")
            response = requests.get(url, headers=HEADERS)
            response.raise_for_status() # Raise exception for HTTP errors
            data = response.json()

            # Check if we've hit rate limits
            if data.get('errors') and 'ratelimit' in str(data.get('errors')).lower():
                logger.warning(f"Rate limit hit. Waiting longer...")
                time.sleep(60) # Wait a full munute before trying again
                continue

            return data
        except requests.exceptions.RequestException as e:
            logger.error(f"API request failed (attempt {attempt+1}/{max_retries}): {e}")
            if attempt < max_retries - 1:
                time.sleep(5 * (attempt + 1)) # Exponential backoff
            else:
                logger.error(f"Failed to fetch data after {max_retries} attempts")
    return {"response": []}
        
def safe_get_value(stats_list, stat_type, default=0):
    """Safely extract a statistic value with proper type handling"""
    for item in stats_list:
        if item['type'] == stat_type:
            value = item.get('value')
            # Handle different value types
            if value is None:
                return default
            if isinstance(value, str):
                try:
                    return float(value.replace('%',''))
                except ValueError:
                    return value
            return value
    return default

def format_date(date_string):
    """Format date string to consistent format"""
    try:
        date_obj = datetime.strptime(date_string, "%Y-%m-%dT%H:%M:%S%z")
        return date_obj.strftime("%Y-%m-%d")
    except ValueError:
        return date_string # Return original if parsing fails
    
