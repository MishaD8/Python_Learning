import pandas as pd

# Test data loading
try:
    data = pd.read_csv(r'G:\Мой диск\cybersecurity\Python for cybersecurity\Project_dataset\dataset.csv')
    print("Data loaded successfully!")
    print(f"Found {len(data)} lottery draws")
    print("First few rows:")
    print(data.head())
except Exception as e:
    print(f"Error loading data: {e}")