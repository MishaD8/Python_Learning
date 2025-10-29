import pandas as pd

# Load the CSV without headers
data = pd.read_csv(r'G:\Мой диск\cybersecurity\Python for cybersecurity\Project_dataset\dataset.csv', header=None)

# Assign column names manually
data.columns = ['date', 'num1', 'num2', 'num3', 'num4', 'num5', 'num6', 'bonus']

print("Column names:", data.columns)
print(data.head())