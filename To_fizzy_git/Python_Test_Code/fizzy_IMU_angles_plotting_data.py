from fizzy_udp import Fizzy

import time
from typing import Any
import numpy as np
import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtGui
from collections import deque
import csv


def extract_euler_from_packet(packet):
    """
    Given a data packet with the specified format, extract Euler angles.
    Input packet: [timestamp, motor_speed, battery_voltage, qx, qy, qz, qw, mag_cal_level]
    Returns: (roll, pitch, yaw) in radians
    """
    # Extract quaternion components
    qx = packet[3]
    qy = packet[4]
    qz = packet[5]
    qw = packet[6]

    # Rearrange to [w, x, y, z]
    q = [qw, qx, qy, qz]

    # Convert to Euler angles
    w, x, y, z = q

    t0 = +2.0 * (w * x + y * z)
    t1 = +1.0 - 2.0 * (x * x + y * y)
    roll = np.arctan2(t0, t1)

    t2 = +2.0 * (w * y - z * x)
    t2 = np.clip(t2, -1.0, 1.0)
    pitch = np.arcsin(t2)

    t3 = +2.0 * (w * z + x * y)
    t4 = +1.0 - 2.0 * (y * y + z * z)
    yaw = np.arctan2(t3, t4)

    return roll, pitch, yaw


# Set up pyqtgraph plots
pg.setConfigOptions(antialias=True)
app = pg.mkQApp()

win = pg.GraphicsLayoutWidget(show=True, title="Fizzy IMU Angles")
win.resize(800, 600)

angle_names = ['Roll', 'Pitch', 'Yaw']
plots = []
curves = []
data_plotting = 200

# Lists to store all data for each signal
x_data_all = [[] for _ in range(3)]
y_data_all = [[] for _ in range(3)]

# Deques to store the last N data points for plotting for each signal
x_data_plot = [deque(maxlen=data_plotting) for _ in range(3)]
y_data_plot = [deque(maxlen=data_plotting) for _ in range(3)]

# Create 3 plots
for i in range(3):
    p = win.addPlot(row=i, col=0)
    p.setLabel('left', angle_names[i], units='rad')
    p.setLabel('bottom', 'Time', units='s')
    p.setTitle(angle_names[i])
    p.showGrid(x=True, y=True)
    
    curve = p.plot(pen='w')
    plots.append(p)
    curves.append(curve)

save_data = True                 # Flag to indicate if data should be saved
desired_minimal_cycle_time = 0.001  # this value is used to limit really fast cylcle times 
bais_negative = 0
bais_positive = 0
# Frequency monitoring
frame_count = 0
last_freq_check = time.time()


# Initialize fizzy
fizzy = Fizzy()
fizzy.start_downlink()  # Start streaming at ~104Hz

data = fizzy.get_data_downlink()

print(data)

start_time_on_PCB = data[0]/1_000_000 # in seconds

# Main loop
try:
    while True:
        start = time.time()

        # Import angles, temperature, battery voltage and time
        data = fizzy.get_data_downlink()
        roll, pitch, yaw = extract_euler_from_packet(data)
        
        time_on_PCB = data[0]/1_000_000 # in seconds
        
        angles = [roll, pitch, yaw]
        for signal_id in range(3):
            y = angles[signal_id]
            x = time_on_PCB
            
            x_data_all[signal_id].append(x)
            y_data_all[signal_id].append(y)

            x_data_plot[signal_id].append(x)
            y_data_plot[signal_id].append(y)

            # Update curve with new data
            curves[signal_id].setData(list(x_data_plot[signal_id]), list(y_data_plot[signal_id]))

        # Process Qt events to keep UI responsive
        app.processEvents()
        
        # Frequency monitoring
        frame_count += 1
        current_time = time.time()
        if current_time - last_freq_check >= 1.0:  # Check every second
            frequency = frame_count / (current_time - last_freq_check)
            cycle_time = time.time() - start
            print(f"Frequency: {frequency:.1f} Hz, Cycle time: {cycle_time*1000:.2f}ms")
            frame_count = 0
            last_freq_check = current_time

except KeyboardInterrupt:
    print("Measurement interrupted by user.")




# Save data if condition is met
if save_data:
    filename = "sensor_data_angles.csv"
    with open(filename, 'w', newline='') as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerow(['Time', 'Signal 0', 'Signal 1', 'Signal 2'])
        for i in range(len(x_data_all[0])):
            row = (x_data_all[0][i], y_data_all[0][i], y_data_all[1][i], y_data_all[2][i]) 
            csvwriter.writerow(row)
    print(f"Data saved to {filename}")

# plt.show()