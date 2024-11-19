import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

# Load the CSV file
df = pd.read_csv("gas_sensor_data.csv")

# Convert the timestamp column to datetime format
df['timestamp'] = pd.to_datetime(df['timestamp'])

# Smooth data using moving average
df['smoothed'] = df['gas_level'].rolling(window=5).mean()

# Detect anomalies (values > threshold)
threshold = 400
df['anomaly'] = df['gas_level'] > threshold
print(df.head())

# Prepare data for prediction
X = np.array(df.index).reshape(-1, 1)  # Feature: index (time)
y = df['gas_level']                    # Target: gas level

# Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Linear Regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Predict future gas levels
predicted = model.predict(X_test)

# Evaluate model performance
print("Mean Squared Error:", mean_squared_error(y_test, predicted))

plt.figure(figsize=(10, 6))
plt.plot(df['timestamp'], df['gas_level'], label='Gas Level', color='blue')
plt.plot(df['timestamp'], df['smoothed'], label='Smoothed', color='orange')
plt.axhline(y=threshold, color='red', linestyle='--', label='Threshold')
plt.legend()
plt.title('Gas Levels Over Time')
plt.xlabel('Time')
plt.ylabel('Gas Level')
plt.grid()
plt.show()