
"""
Forward motion.
"""

import time
import numpy as np

from states.base import State


class Forward(State):

    def __init__(self, Kp, time_backwards, time_forwards, cycle_duration_roll_forward, duration=None):

        super().__init__(duration)

        self.Kp = Kp

        self.time_backwards = time_backwards
        self.time_forwards = time_forwards
        self.cycle_duration_roll_forward = cycle_duration_roll_forward


    def update(self, dt, sensors, joystick):

        # Time inside oscillation cycle
        t = (time.time() - self.start_time) % self.cycle_duration_roll_forward

        roll = sensors["roll"]
       

        # Piecewise cosine trajectory
        if t < self.time_backwards:
             # Reference backwards position (first go backwards for a small duration and then forward to have the most momentum)
            ref = -1.0
        
            # Convert IMU frame
            error = ref - np.sin(roll)      # 180 and -180 at the bottom and 0 when standing upright, calibrate the IMU data to the orientation of the drivetrain 

            return self.Kp * error

        elif t < self.time_backwards + self.time_forwards:
             # Reference forward position
            ref = 1.0
        
            # Convert IMU frame
            error = ref - np.sin(roll)      # 180 and -180 at the bottom and 0 when standing upright, calibrate the IMU data to the orientation of the drivetrain 
            
            return self.Kp * error
        
        else:
            return 1.0    # full power forward
