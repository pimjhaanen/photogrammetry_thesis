import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Step 1: Take the input CSV path
csv_path = "data_inside_thursday/recording_IV10A0.csv"

# Step 2: Load the data
data = pd.read_csv(csv_path)

# Ensure the columns exist in the CSV
if 'recording timestamp' not in data.columns or 'Airspeed in m/s' not in data.columns:
    raise ValueError("CSV file must contain 'timestamp' and 'Airspeed' columns.")

# Step 3: Process the Airspeed column
# Take the average of the Airspeed column and subtract it from the Airspeed values
air_speed_avg = data['Airspeed in m/s'].mean()
data['Airspeed_centered'] = data['Airspeed in m/s'] - air_speed_avg

# Step 4: Apply a low pass filter with alpha = 0.9 using .loc to avoid SettingWithCopyWarning
alpha = 0.95
data['Airspeed_filtered'] = data['Airspeed_centered'].copy()

# Apply low-pass filter using NumPy for efficient iteration
for i in range(1, len(data)):
    data.loc[i, 'Airspeed_filtered'] = alpha * data.loc[i-1, 'Airspeed_filtered'] + (1 - alpha) * data.loc[i, 'Airspeed_centered']

# Step 5: Plot the data in the same plot
plt.figure(figsize=(8, 4))

# Plot: Airspeed fluctuations around 0 and Airspeed with low-pass filter on the same plot
plt.plot(data['recording timestamp'].values, data['Airspeed_centered'].values, label='Measured $V_\infty$')
plt.plot(data['recording timestamp'].values, data['Airspeed_filtered'].values, label='Filtered (Low-pass) $V_\infty$')
plt.axhline(0, color="k", linestyle="--")
# Labels and title
plt.xlabel('Timestamp')
plt.ylabel('$V_\infty$ - $\overline{V_\infty}$ (m/s)')
plt.title('Airspeed Fluctuations')

# Grid and legend
plt.grid(True)
plt.legend()

# Show the plot
plt.tight_layout()
plt.show()

