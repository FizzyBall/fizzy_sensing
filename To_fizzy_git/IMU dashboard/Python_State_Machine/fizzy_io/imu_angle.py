"""
IMU processing utilities.
"""

import numpy as np


def extract_euler(packet):

    qx, qy, qz, qw = packet[3:7]

    w, x, y, z = qw, qx, qy, qz

    t0 = 2*(w*x + y*z)
    t1 = 1 - 2*(x*x + y*y)

    roll = np.arctan2(t0, t1)

    t2 = 2*(w*y - z*x)
    t2 = np.clip(t2, -1, 1)

    pitch = np.arcsin(t2)

    t3 = 2*(w*z + x*y)
    t4 = 1 - 2*(y*y + z*z)

    yaw = np.arctan2(t3, t4)

    return roll, pitch, yaw
