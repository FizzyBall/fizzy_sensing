"""
Random controller, chaotic movements
"""

import random
import numpy as np

from states.base import State


class Random(State):

    def __init__(self, config, duration=5):
        super().__init__(duration)

        self.config = config

        # Internal state
        self.output = 0.0
        self.target = 0.0

    def enter(self):
        super().enter()
        self.output = 0.0
        self.target = self._new_target()

    def _new_target(self):
        """
        Generate a new random target outside the deadzone.
        - deadzone is where the motor does not move [from -0.2 to 0.2]
        """

        # Runtime parameters from config
        gain = self.config.random_gain
        deadzone = self.config.random_deadzone

        target = random.uniform(-1.0, 1.0)

        # Avoid region where motor does not move
        if 0 <= target < deadzone:
            target = deadzone
        elif -deadzone < target < 0:
            target = -deadzone

        return gain * target

    def update(self, dt, sensors, joystick):

        # Read parameters every loop so Streamlit can change them live
        chaos = self.config.random_chaos            # how often target changes
        smoothness = self.config.random_smooth      # 0..1
        deadzone = self.config.random_deadzone

        # With higher chaos value, choose a new target more often
        if random.random() < chaos * dt:
            self.target = self._new_target()

        # Smoothly move output toward target
        alpha = np.clip(1.0 - smoothness, 0.0, 1.0)
        self.output += alpha * (self.target - self.output)

        # Prevent output from getting stuck inside motor deadzone
        if abs(self.output) < deadzone:
            if self.target >= 0:
                self.output = deadzone
            else:
                self.output = -deadzone

        return float(np.clip(self.output, -1.0, 1.0))
