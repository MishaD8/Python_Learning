import requests
import json
import os
from typing import Dict, Any, Optional

class WeatherAPI:
    def __init__(self):
        # Store API key as environment variable for security
        self.access_key = os.getenv('WEATHERSTACK_API_KEY', '88f5ccf1fc67c13ae9557d99439f97db')
        self.base_url = 'http://api.weatherstack.com/'
        
    def get_weather_data(self, endpoint: str, location: str) -> Optional[Dict[str, Any]]:
        """Fetch weather data from the API with error handling."""
        try:
            # Fix query parameter structure
            params = {
                'access_key': self.access_key,
                'query': location
            }
            
            url = f'{self.base_url}{endpoint}'
            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()  # Raises exception for bad status codes
            
            data = response.json()
            
            # Check for API errors
            if 'error' in data:
                print(f"API Error: {data['error']['info']}")
                return None
                
            return data
            
        except requests.exceptions.RequestException as e:
            print(f"Network error: {e}")
            return None
        except json.JSONDecodeError:
            print("Error: Invalid JSON response")
            return None
        except Exception as e:
            print(f"Unexpected error: {e}")
            return None
    
    def display_available_info(self, weather_data: Dict[str, Any]) -> None:
        """Display available weather information keys."""
        if 'current' in weather_data:
            print("\nAvailable weather information:")
            for key in weather_data['current'].keys():
                print(f"- {key}")
            print()
    
    def get_user_input(self, prompt: str, valid_options: list = None) -> str:
        """Get user input with validation."""
        while True:
            user_input = input(prompt).strip()
            if not user_input:
                print("Please enter a valid input.")
                continue
            if valid_options and user_input not in valid_options:
                print(f"Please choose from: {', '.join(valid_options)}")
                continue
            return user_input
    
    def run(self):
        """Main application logic."""
        print("Weather Information System")
        print("=" * 30)
        
        # Get endpoint with validation
        valid_endpoints = ['current', 'historical', 'forecast']
        endpoint = self.get_user_input(
            f"What type of weather data do you want? ({'/'.join(valid_endpoints)}): ",
            valid_endpoints
        )
        
        # Get location
        location = self.get_user_input("What location do you want weather information for? ")
        
        # Fetch weather data
        print("Fetching weather data...")
        weather_data = self.get_weather_data(endpoint, location)
        
        if not weather_data:
            print("Failed to retrieve weather data. Please try again.")
            return
        
        # Display location info if available
        if 'location' in weather_data:
            loc_info = weather_data['location']
            print(f"\nWeather for: {loc_info.get('name', 'Unknown')}, {loc_info.get('country', 'Unknown')}")
        
        # Handle current weather data
        if 'current' in weather_data:
            current_info = weather_data['current']
            self.display_available_info(weather_data)
            
            # Interactive information lookup
            while True:
                info_request = input("What information are you searching for? (or 'quit' to exit): ").strip()
                
                if info_request.lower() in ['quit', 'exit', 'q']:
                    break
                    
                if info_request in current_info:
                    value = current_info[info_request]
                    print(f"{info_request}: {value}")
                else:
                    print("Requested information not available.")
                    print("Available options:")
                    for key in current_info.keys():
                        print(f"  - {key}")
                
                # Ask if user wants to continue
                continue_choice = input("\nWould you like to search for more information? (y/n): ").strip().lower()
                if continue_choice not in ['y', 'yes']:
                    break
        else:
            print("No current weather data available.")
            # Handle other types of data if needed
            print("Available data sections:")
            for key in weather_data.keys():
                print(f"- {key}")

def main():
    """Entry point for the application."""
    try:
        weather_app = WeatherAPI()
        weather_app.run()
    except KeyboardInterrupt:
        print("\nProgram interrupted by user.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

if __name__ == "__main__":
    main()