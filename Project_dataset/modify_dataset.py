import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the full 5-year dataset
data = pd.read_csv(r'G:\Мой диск\cybersecurity\Python for cybersecurity\Project_dataset\dataset.csv')

# Convert date to datetime
data['data'] = pd.to_datetime(data['date'])

# Sort chronologically
data = data.sort_values('data')

# Add time-based features if desired
data['day_of_week'] = data['data'].dt.dayofweek
data['month'] = data['date'].dt.month

# Extract all lottery numbers including the bonus number for prediction
numeric_data = data[['num1', 'num2', 'num3', 'num4', 'num5', 'num6', 'bonus']]

