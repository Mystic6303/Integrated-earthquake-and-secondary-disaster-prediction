import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random
# Define parameters
start_date = '2023-01-01'
end_date = '2030-12-31'  # Adjust the end date to cover a longer period
num_samples = 30000
random.seed(42)
# Define parameters for sinusoidal curves
num_curves = 50

# Generate synthetic data
dates = pd.date_range(start=start_date, end=end_date, periods=num_samples)
time = np.linspace(0, 30*np.pi, num_samples)

# Create synthetic magnitudes for each curve
magnitudes = []
for _ in range(num_curves):
    amplitude = np.random.uniform(2, 8)  # Amplitude between 2 and 8
    curve_length = num_samples // num_curves
    curve_time = time[_ * curve_length:(_ + 1) * curve_length]
    curve = np.sin(curve_time) * amplitude + np.random.uniform(-0.5, 0.5)
    # Clip the magnitudes to ensure they stay between 2 and 8
    curve = np.clip(curve, 0, 8)
    magnitudes.append(curve)

# Combine the magnitudes to create the dataset
combined_magnitudes = np.concatenate(magnitudes)

# Trim or pad the combined magnitudes to ensure the total number of samples is 20,000
combined_magnitudes = np.resize(combined_magnitudes, (num_samples,))

# Create dataframe
synthetic_df = pd.DataFrame({'Date': dates, 'Magnitude': combined_magnitudes})

# Plot the synthetic data
plt.figure(figsize=(10, 5))
plt.plot(synthetic_df['Date'], synthetic_df['Magnitude'], color='b')
plt.title('Earthquake Magnitudes Over Time')
plt.xlabel('Date')
plt.ylabel('Magnitude')
plt.grid(True)
plt.show()

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

# Prepare data for LSTM
def create_sequences(data, seq_length):
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:i+seq_length])
        y.append(data[i+seq_length])
    return np.array(X), np.array(y)

# Data Preparation
sequence_length = 10  # Adjust this parameter if needed
X, y = create_sequences(synthetic_df['Magnitude'].values, sequence_length)

# Split data into train and test sets
train_size = int(len(X) * 0.8)
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

# Build LSTM model
model = Sequential([
    LSTM(50, input_shape=(X_train.shape[1], 1)),
    Dense(1)
])

# Compile the model
model.compile(optimizer='adam', loss='mse')

# Train the model
model.fit(X_train, y_train, epochs=10, batch_size=32, verbose=1)

# Evaluate the model
train_loss = model.evaluate(X_train, y_train, verbose=0)
test_loss = model.evaluate(X_test, y_test, verbose=0)
print(f'Train Loss: {train_loss}, Test Loss: {test_loss}')

# Predictions
train_predictions = model.predict(X_train)
test_predictions = model.predict(X_test)
input_values = [2.5, 3.0, 3.2, 3.0, 4.1,4.6,4.7,4.9,4.7,5.0]
input_sequence = np.array(input_values).reshape(1, -1)

# Predict using the LSTM model
predictions = model.predict(input_sequence)
print(predictions)
df2=pd.read_csv("query.csv")
df2.head()
# Extract 'mag' column
data = df2['mag'].values

# Data Preparation
sequence_length = 10  # Adjust this parameter if needed

# Prepare sequences
def create_sequences(data, seq_length):
    X = []
    for i in range(len(data) - seq_length):
        X.append(data[i:i+seq_length])
    return np.array(X)

X = create_sequences(data, sequence_length)
# Predictions
predictions = model.predict(X)
# Plotting
plt.figure(figsize=(12, 6))
plt.plot(data, label='Actual', color='blue')
plt.plot(np.arange(sequence_length, len(data)), predictions, label='Predicted', color='red')

plt.title('LSTM Model Predictions')
plt.xlabel('Index')
plt.ylabel('Magnitude')
plt.legend()
plt.show()

import pickle
with open('lstm.pkl', 'wb') as file:
    pickle.dump(model, file)