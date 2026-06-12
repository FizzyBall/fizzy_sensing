"""
Base class for all robot states.
All states must inherit from this.
"""

import time


class State:
    """
    Abstract state class.
    Defines common interface.
    """

    def __init__(self, duration=None):

        # How long this state runs (seconds)
        # None = infinite
        self.duration = duration

        self.start_time = None

    def enter(self):
        """
        Called once when state starts.
        """
        self.start_time = time.time()
        print(f"[FSM] Entering state: {self.__class__.__name__}")

    def exit(self):
        """
        Called when leaving state.
        """
        # print(f"[FSM] Exiting state: {self.__class__.__name__}")
        pass

    def update(self, dt, sensors, joystick):
        """
        Main control logic.

        Must return motor power [-1,1]
        """
        return 0.0

    def finished(self):
        """
        Check if state duration expired.
        """
        if self.duration is None:
            return False

        return time.time() - self.start_time >= self.duration
