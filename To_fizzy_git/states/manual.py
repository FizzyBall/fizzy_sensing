"""
Manual joystick control.
"""

from states.base import State


class Manual(State):

    def update(self, dt, sensors, joystick):

        return joystick.get_y_axis_left()
