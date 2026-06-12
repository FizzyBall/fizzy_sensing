"""
Interrupt Manager.

This module decides when external events
(joystick, IMU, safety) should affect the FSM.

Priority:
----------
1. Emergency (highest)
2. Sensor corrections
3. Manual commands
4. Normal operation
"""


class Interrupts:

    def check(self, joystick, sensors, config):
        """
        Analyze inputs and generate events.

        Returns:
        --------
        String event name, or None
        """

        roll = sensors["roll"]
        pitch = sensors["pitch"]
        yaw = sensors["yaw"]
        acc_mag = sensors["acc_mag"]

        self.config = config

        # --------------------------------------
        # Tapping sensor input
        # --------------------------------------

        # Hard tap
        if acc_mag > self.config.sensitivity:
            return "TAP"

        # --------------------------------------
        # Joystick inputs
        # --------------------------------------

        # Program exit
        if joystick.exit():
            return "EXIT"

        # Resume autonomous mode
        if joystick.case() == 7:
            return "RESUME"

        # Manual override
        if joystick.case() == 8:
            return "MANUAL"

        # Neutral mode
        if joystick.case() == 1:
            return "NEUTRAL"

        # --------------------------------------
        # No interrupt
        # --------------------------------------

        return None

