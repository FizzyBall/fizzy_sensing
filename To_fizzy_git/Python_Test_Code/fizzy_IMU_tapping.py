from fizzy_udp import Fizzy
import numpy as np
import matplotlib.pyplot as plt
import time
from collections import deque

## Plots the IMU angles. 
# for exit use the keys: control + C 

# The IMU gives Quaternions which are transformed into eurler angles with the following function 
def extract_euler_from_packet(packet):
    """
    Given a data packet with the specified format, extract Euler angles.
    Input packet: [timestamp, motor-speed, v_bat,
                    q1, q2, q3, q4,
                    lin. acc x, lin. acc y, lin.acc z,
                    ACC_x_raw, ACC_y_raw, ACC_z_raw,
                    GYRO_x_raw, GYRO_y_raw, GYRO_z_raw,
                    MAG_x_raw, MAG_y_RAW, MAG_z_raw,
                    quality_mag]
    Returns: (roll, pitch, yaw) in radians
    """
    qx, qy, qz, qw = packet[3], packet[4], packet[5], packet[6]
    w, x, y, z = qw, qx, qy, qz

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


SIGNALS = {
    "lin_acc": {
        "indices": (7, 8, 9),
        "labels": ("Lin Acc X (g)", "Lin Acc Y (g)", "Lin Acc Z (g)"),
    },
    "raw_acc": {
        "indices": (10, 11, 12),
        "labels": ("Raw Acc X", "Raw Acc Y", "Raw Acc Z"),
    },
    "gyro_raw": {
        "indices": (13, 14, 15),
        "labels": ("Gyro X", "Gyro Y", "Gyro Z"),
    },
}

# Initialize fizzy
fizzy = Fizzy()


MODE = "lin_acc"   # "lin_acc", "raw_acc", "gyro_raw"

# Main loop
try:
    start_time = time.time()

    indices = SIGNALS[MODE]["indices"]
    axis_labels = SIGNALS[MODE]["labels"]

    while True:
        loop_start = time.time()
        data = fizzy.get_data()

        t = data[0] / 1_000_000 - start_time
        
        # Extract XYZ
        x, y, z = (data[i] for i in indices)

        # Compute magnitude
        mag = np.sqrt(x**2 + y**2 + z**2)
        print(mag)

        if mag > 1.5:
            fizzy.set_motor(0.5)
            time.sleep(0.5)
            fizzy.set_motor(-0.5)
            time.sleep(0.5)
            fizzy.set_motor(0)

        # Timing control
        loop_time = time.time() - loop_start
        if loop_time < 0.01:
            time.sleep(0.01 - loop_time)

        
    


except KeyboardInterrupt:
    print("Measurement interrupted by user.")
