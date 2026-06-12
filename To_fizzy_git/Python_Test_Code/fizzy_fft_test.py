
import matplotlib.pyplot as plt
import numpy as np
import csv

window_samples = 128 # Number of samples in each FFT window
#get data from file for now. 
filename = './personal_recs/rec_172544.csv'
orientation_cols = ['Roll', 'Pitch', 'Yaw']
acceleration_cols = ['Acc X', 'Acc Y', 'Acc Z', 'Magnitude']
gyro_cols = ['Gyro X', 'Gyro Y', 'Gyro Z']

# Read CSV and extract available columns with values 
time_data = []
data_dict = {col: [] for col in orientation_cols + acceleration_cols + gyro_cols}
with open(filename, 'r') as csvfile:
    header = next(csv.reader(csvfile))
    available_orientation = [col for col in orientation_cols if col in header]
    available_acceleration = [col for col in acceleration_cols if col in header]
    available_gyro = [col for col in gyro_cols if col in header]
    col_indices = {col: header.index(col) for col in available_orientation + available_acceleration + available_gyro}
    time_index = header.index('timestamp')
    
    for row in csv.reader(csvfile):
        time_data.append(float(row[time_index]))
        for col in available_orientation + available_acceleration + available_gyro:
            data_dict[col].append(float(row[col_indices[col]]))
#Get sample rate from time data
time_diffs = np.diff(time_data) # Time is in ms
#print(time_diffs)
sample_rate = 1 / np.mean(time_diffs) # Convert ms to seconds and calculate sample rate
#print(f"Sample Rate: {sample_rate} Hz")'

# Process data in windows and compute FFT of only linear acceleration magnitude
acc_mag = np.array(data_dict['Acc Y'])

# Create Hanning window
hanning_window = np.hanning(window_samples)

# Function to compute FFT with Hanning window on a data chunk
def compute_fft_windowed(data_chunk, window, sample_rate):
    """
    Compute FFT with Hanning window applied to data chunk.
    
    Args:
        data_chunk: numpy array of samples (must match window length)
        window: Hanning window (or other window function)
        sample_rate: sampling rate in Hz
        
    Returns:
        frequencies: frequency axis (Hz)
        magnitude: magnitude spectrum (normalized)
    """
    # Apply window to data
    windowed_data = data_chunk * window
    
    # Compute FFT
    fft_result = np.fft.fft(windowed_data)
    
    # Compute magnitude spectrum (normalized by window energy)
    magnitude = np.abs(fft_result) / np.sum(window)
    
    # Only keep positive frequencies
    magnitude = magnitude[:len(magnitude)//2]
    
    # Compute frequency axis
    frequencies = np.fft.fftfreq(len(data_chunk), 1/sample_rate)[:len(magnitude)]
    
    return frequencies, magnitude

# Process all windows from recorded data with 50% overlap
overlap_percent = 0.5
step_size = int(window_samples * (1 - overlap_percent))  # 50% overlap = 50% step0
num_windows = (len(acc_mag) - window_samples) // step_size + 1
fft_results = []

for i in range(num_windows):
    start_idx = i * step_size
    end_idx = start_idx + window_samples
    
    if end_idx <= len(acc_mag):  # Only process windows that fit in data
        data_chunk = acc_mag[start_idx:end_idx]
        freq, mag = compute_fft_windowed(data_chunk, hanning_window, sample_rate)
        fft_results.append((freq, mag))

# Plot FFT results
if fft_results:
    plt.figure(figsize=(12, 6))
    # Plot first few windows
    for i in range(len(fft_results)):
        freq, mag = fft_results[i]
        plt.plot(freq, mag, label=f'Window {i+1}', alpha=0.7)
    
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Magnitude')
    plt.title('FFT of Acceleration (Hanning Windowed)')
    plt.legend()
    plt.grid(True)
    plt.show()

# FOR LIVE DATA: Create a sliding window buffer with 50% overlap
# Uncomment and use this pattern for live data processing:
#
# from collections import deque
# 
# class FFTProcessor:
#     def __init__(self, window_size, sample_rate, overlap_percent=0.5):
#         self.window_size = window_size
#         self.sample_rate = sample_rate
#         self.overlap_percent = overlap_percent
#         self.step_size = int(window_size * (1 - overlap_percent))
#         self.buffer = deque(maxlen=window_size)
#         self.hanning = np.hanning(window_size)
#         self.samples_since_last_fft = 0
#     
#     def add_sample(self, value):
#         self.buffer.append(value)
#         self.samples_since_last_fft += 1
#         
#         # Compute FFT when step_size new samples have arrived
#         if len(self.buffer) == self.window_size and self.samples_since_last_fft >= self.step_size:
#             result = self.compute_fft()
#             self.samples_since_last_fft = 0  # Reset counter for next window
#             return result
#         return None
#     
#     def compute_fft(self):
#         data_chunk = np.array(list(self.buffer))
#         return compute_fft_windowed(data_chunk, self.hanning, self.sample_rate)
    
