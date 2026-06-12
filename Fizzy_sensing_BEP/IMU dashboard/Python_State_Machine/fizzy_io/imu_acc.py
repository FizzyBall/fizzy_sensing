"""
IMU processing accelerations.

"lin_acc"
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
"""

import numpy as np



def extract_acc(packet):

    # Extract XYZ 
    x, y, z = packet[7:10]

    # Compute magnitude
    acc_mag = np.sqrt(x**2 + y**2 + z**2)
    
    # if acc_mag > 1:
        # print(acc_mag)

    return acc_mag
