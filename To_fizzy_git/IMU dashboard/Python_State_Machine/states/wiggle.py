"""
Oscillatory wiggle motion.
"""

import time
import numpy as np

from states.base import State


class Wiggle(State):

    def __init__(self, config, duration=5):

        super().__init__(duration)
        self.config = config
        

    def update(self, dt, sensors, joystick):

        self.T1 = self.config.T1
        self.T2 = self.config.T2
        
        self.A2 = self.config.A2
        self.A1 = -0.5 * self.A2
        # Total oscillation cycle time
        self.cycle = self.T1 + self.T2 + self.T1


        # Time inside oscillation cycle
        t = (time.time() - self.start_time) % self.cycle
        # print("Wiggle")
        # Piecewise cosine trajectory
        if t < self.T1:
            return self.A1 * np.cos(np.pi * t / self.T1)

        elif t < self.T1 + self.T2:
            return self.A2 * np.cos(np.pi * (t-self.T1) / self.T2)

        else:
            return self.A1 * np.cos(np.pi * (t-self.T1-self.T2) / self.T1)
