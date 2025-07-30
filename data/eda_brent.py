import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller

# Load the data
# Assumes the CSV is named 'BrentOilPrices.csv' and is in the same directory

df = pd.read_csv('BrentOilPrices.csv')

# Parse the Date column to datetime
# The format is assumed to be 'day-month-year', e.g., '20-May-87'
df['Date'] = pd.to_datetime(df['Date'], format='%d-%b-%y')

# Sort by date
df = df.sort_values('Date').reset_index(drop=True)

# Quick look at the data
print(df.head())
print(df.info())

# Plot the Price Series
plt.figure(figsize=(14, 6))
plt.plot(df['Date'], df['Price'])
plt.title('Brent Oil Price Over Time')
plt.xlabel('Date')
plt.ylabel('Price (USD/barrel)')
plt.show()

# Check for Missing Values
print("Missing values:\n", df.isnull().sum())

# Summary Statistics
print("Summary statistics:\n", df['Price'].describe())

# Calculate Log Returns
df['LogReturn'] = np.log(df['Price']).diff()

plt.figure(figsize=(14, 6))
plt.plot(df['Date'], df['LogReturn'])
plt.title('Log Returns of Brent Oil Price')
plt.xlabel('Date')
plt.ylabel('Log Return')
plt.show()

# Stationarity Test (ADF)
result = adfuller(df['LogReturn'].dropna())
print('ADF Statistic:', result[0])
print('p-value:', result[1])

# Volatility Clustering Visualization
plt.figure(figsize=(14, 6))
plt.plot(df['Date'], df['LogReturn'].abs())
plt.title('Absolute Log Returns (Volatility Clustering)')
plt.xlabel('Date')
plt.ylabel('Absolute Log Return')
plt.show()