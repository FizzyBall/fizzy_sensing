"""
Backward motion.
"""

import time
import numpy as np

from states.base import State


class Backward(State):

    def __init__(self, config, duration=None):

        super().__init__(duration)
       
        self.config = config

        # Internal timer for motion phase
        self.phase_start = None

    def enter(self):
        super().enter()

        # Reset cycle timing every time the state starts
        self.phase_start = time.time()

    def update(self, dt, sensors, joystick):


        self.Kp = self.config.Kp

        self.time_backwards = self.config.time_backwards
        self.time_forwards = self.config.time_forwards
        self.cycle_duration_roll_forward = self.config.cycle_duration_roll_forward # total ammount of time in this cycle

        
        # Time since entering this state
        elapsed = time.time() - self.phase_start

        # Optional repeat of the whole cycle
        t = elapsed % self.cycle_duration_roll_forward

        roll = sensors["roll"]
       

        # Piecewise cosine trajectory
        if t < self.time_backwards:
             # Reference backwards position (first go backwards for a small duration and then forward to have the most momentum)
            ref = 1.3
        
            # Convert IMU frame
            error = ref - np.sin(roll)      # 180 and -180 at the bottom and 0 when standing upright, calibrate the IMU data to the orientation of the drivetrain 

            return self.Kp * error

        elif t < self.time_backwards + self.time_forwards:
             # Reference forward position
            ref = -1.5
        
            # Convert IMU frame
            error = ref - np.sin(roll)      # 180 and -180 at the bottom and 0 when standing upright, calibrate the IMU data to the orientation of the drivetrain 
            
            return self.Kp * error
        
        else:
            return -1.0    # full power forward
