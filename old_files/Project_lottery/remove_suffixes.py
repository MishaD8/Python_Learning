import pandas as pd
import re

# Load the CSV without headers and set column names
data = pd.read_csv(r'G:\Мой диск\cybersecurity\Python for cybersecurity\Project_dataset\dataset.csv', header=None)
data.columns = ['date', 'num1', 'num2', 'num3', 'num4', 'num5', 'num6', 'bonus']

# Function to clean ordinal suffixes
def clean_date(date_str):
    # Remove ordinal suffixes (like 'st', 'nd', 'rd', 'th')
    cleaned = re.sub(r'(\d+)(st|nd|rd|th)', r'\1', date_str)
    return cleaned

# Apply the function to the 'date' column
data['date'] = data['date'].apply(clean_date)

# Convert cleaned date to datetime
data['date'] = pd.to_datetime(data['date'], errors='coerce')

print(data)