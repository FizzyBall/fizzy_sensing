import csv
import matplotlib.pyplot as plt

filename = "FrontBack1.csv"
orientation_cols = ['Roll', 'Pitch', 'Yaw']
acceleration_cols = ['Acc X', 'Acc Y', 'Acc Z', 'Magnitude']
gyro_cols = ['Gyro X', 'Gyro Y', 'Gyro Z']

# Read CSV and extract available columns
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

print(f"Orientation: {available_orientation}\nAcceleration: {available_acceleration}\nGyro: {available_gyro}")

# Create subplots
fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 10))

# Plot orientation data (Roll, Pitch, Yaw)
if available_orientation:
    for col in available_orientation:
        ax1.plot(time_data, data_dict[col], label=col)
    ax1.set_xlabel('Time (ms)')
    ax1.set_ylabel('Angle (degrees)')
    ax1.set_title('Orientation Over Time')
    ax1.legend()
    ax1.grid(True)
else:
    ax1.text(0.5, 0.5, 'No orientation data', ha='center', va='center')

# Plot acceleration data (Acc X, Y, Z, Magnitude)
if available_acceleration:
    for col in available_acceleration:
        ax2.plot(time_data, data_dict[col], label=col)
    ax2.set_xlabel('Time (ms)')
    ax2.set_ylabel('Acceleration (g or g)')
    ax2.set_title('Acceleration Over Time')
    ax2.legend()
    ax2.grid(True)
else:
    ax2.text(0.5, 0.5, 'No acceleration data', ha='center', va='center')

# Plot gyro data (Gyro X, Y, Z)
if available_gyro:
    for col in available_gyro:
        ax3.plot(time_data, data_dict[col], label=col)
    ax3.set_xlabel('Time (ms)')
    ax3.set_ylabel('Angular Velocity (rad/s or deg/s)')
    ax3.set_title('Gyro Over Time')
    ax3.legend()
    ax3.grid(True)
else:
    ax3.text(0.5, 0.5, 'No gyro data', ha='center', va='center')

plt.tight_layout()
plt.show()
