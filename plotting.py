import pandas as pd
import matplotlib.pyplot as plt

# Load the CSV file
csv_file_path = '/Users/anettvarghese/Downloads/MILimbEEG An EEG Signals Dataset based on Upper and Lower Limb Task During the Execution of Motor and Motorimagery Tasks/S1/S1R1I2_1.csv'  # Replace with your file path
data = pd.read_csv(csv_file_path)

# Use the index as the time axis and select the signal column
time_column = data.index
signal_column = data['1']  # Replace '0' with any other column name like '1', '2', etc., if you want to plot a different signal
# Check if there's a time column
if 'Time' in data.columns:
    # Calculate the time difference between consecutive samples
    time_diffs = data['Time'].diff().dropna()
    # Calculate the average sampling interval
    average_time_diff = time_diffs.mean()
    # Calculate the sampling rate
    sampling_rate = 1 / average_time_diff
else:
    # Alternatively, if there is no time column, you can estimate the sampling rate if you know the duration
    # For example, if the recording duration is 10 seconds:
    duration_seconds = 10  # Replace with actual duration if known
    number_of_samples = len(data)
    sampling_rate = number_of_samples / duration_seconds

print(f"Estimated Sampling Rate: {sampling_rate} Hz")
# Plot the signal
plt.figure(figsize=(10, 6))
plt.plot(time_column, signal_column, label='Signal 0')
plt.xlabel('Time')
plt.ylabel('Signal Value')
plt.title('Signal Plot')
plt.legend()
plt.show()

