"""
Downwards position controller.
"""

import numpy as np

from states.base import State


class Zero_stand(State):

    def __init__(self, config, duration=None):

        super().__init__(duration)

        self.config = config
        self.Kp = self.config.Kp


    def update(self, dt, sensors, joystick):

        self.Kp = self.config.Kp
        
        roll = sensors["roll"]

        # Reference upright position
        ref = 0.0

        # Convert IMU frame
        error = ref - np.sin(roll)      # 180 and -180 at the bottom and 0 when standing upright, calibrate the IMU data to the orientation of the drivetrain 
       
        
        return self.Kp * error