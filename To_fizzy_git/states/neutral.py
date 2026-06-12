"""
Neutral state: motor off.
"""

from states.base import State


class Neutral(State):

    def update(self, dt, sensors, joystick):
        # print("Neutral")
        return 0.0
