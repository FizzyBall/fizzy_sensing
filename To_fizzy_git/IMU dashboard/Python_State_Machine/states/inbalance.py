"""
Balancing controller.
"""

import numpy as np

from states.base import State


class Inbalance(State):

    def __init__(self, Kp, duration=None):

        super().__init__(duration)

        self.Kp = Kp + 0.8

    def update(self, dt, sensors, joystick):

        roll = sensors["roll"]

        # Reference upright position
        ref = 0.0

        # Convert IMU frame
        error = ref - np.sin(roll)

        print("Inbalance")
        return -self.Kp * error
