from fizzy_udp import Fizzy
import numpy as np
import matplotlib.pyplot as plt
import time
from collections import deque

## Plots the IMU angles. 
# for exit use the keys: control + C 

# The IMU gives the angular velocity in degrees per second, we convert it to radians per second and apply a threshold to filter out noise. The data is then plotted in real-time using Matplotlib.
def extract_ang_vel(packet):

    # Extract XYZ 
    gyro_x, gyro_y, gyro_z = packet[13:16]
    #Go to rad/s::
    gyro_x, gyro_y, gyro_z = np.radians([gyro_x, gyro_y, gyro_z] / 131)

    # Compute magnitude
    ang_vel_mag = np.sqrt(gyro_x**2 + gyro_y**2 + gyro_z**2)
    # In extract_ang_vel

    if ang_vel_mag < 0.05:  # threshold in rad/s
        gyro_x, gyro_y, gyro_z = 0, 0, 0
    #if ang_vel_mag > 1:
    #    print(ang_vel_mag)

    return gyro_x, gyro_y, gyro_z, ang_vel_mag

# Initialize fizzy
fizzy = Fizzy()

# Setup live plot
plt.ion()
fig, axs = plt.subplots(3, 1, figsize=(8, 6))
labels = ['Gyro X (rad/s)', 'Gyro Y (rad/s)', 'Gyro Z (rad/s)']
data_plotting = 100

x_data = [deque(maxlen=data_plotting) for _ in range(3)]
y_data = [deque(maxlen=data_plotting) for _ in range(3)]

# Main loop
try:
    start_time = time.time()
    while True:
        loop_start = time.time()

        data = fizzy.get_data()
        timestamp = data[0] / 1_000_000  # in seconds
        gyro_x, gyro_y, gyro_z, magnitude = extract_ang_vel(data)
        gyro_data = [gyro_x, gyro_y, gyro_z]
        #angles_deg = np.degrees([roll, pitch, yaw]) # transforms to degree
        # print(angles_deg)

        # Update data for plots
        for i, angle in enumerate(gyro_data):
            x_data[i].append(timestamp - start_time)
            y_data[i].append(angle)

            axs[i].clear()
            axs[i].plot(x_data[i], y_data[i])
            axs[i].set_ylabel(labels[i])
            axs[i].set_xlabel("Time (s)")
            axs[i].grid(True)

        plt.tight_layout()
        plt.pause(0.01)
        
        # # For printing Magnetometer calibration level
        # print(data[-1])

        # Control loop speed
        loop_time = time.time() - loop_start
        minimal_cycle_time = 0.01
        if loop_time < minimal_cycle_time:
            time.sleep(minimal_cycle_time - loop_time)

except KeyboardInterrupt:
    print("Measurement interrupted by user.")