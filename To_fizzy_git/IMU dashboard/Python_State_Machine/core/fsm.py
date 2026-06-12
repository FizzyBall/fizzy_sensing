"""
Finite State Machine (FSM) with stack-based overrides.

This module manages which "behavior" is currently active.

It supports:

1. Autonomous sequences (playlist of states)
2. Temporary overrides (manual / sensor reactions)
3. Automatic resume after override

Main idea:

AUTO MODE:
    State1 -> State2 -> State3 -> ...

OVERRIDE MODE:
    State2 -> Recovery -> State2

This is implemented using a stack.
"""


class StateMachine:

    def __init__(self, states):
        """
        Initialize FSM.

        Parameters:
        -----------
        states : list
            List of autonomous states to execute in order.
        """

        # Main autonomous sequence
        self.states = states

        # Index of current state in sequence
        self.index = 0

        # Currently active state object
        self.current = None

        # Stack for temporary states
        # Each element: (previous_state, previous_index)
        self.stack = []

        # True  = following autonomous sequence
        # False = running override
        self.auto_mode = True

    # --------------------------------------------------
    # Autonomous mode management
    # --------------------------------------------------

    def start(self):
        """
        Start the FSM.
        Activates the first state in the sequence.
        """

        self.current = self.states[0]
        self.current.enter()

    def next(self):
        """
        Switch to next state in autonomous sequence.
        """

        # Cleanly exit current state
        self.current.exit()

        # Advance index (wraps around)
        self.index = (self.index + 1) % len(self.states)

        # Enter next state
        self.current = self.states[self.index]
        self.current.enter()

    def update_auto(self, dt, sensors, joystick):
        """
        Update while in autonomous mode.

        Checks if current state finished,
        and switches if needed.
        """

        if self.current.finished():
            self.next()

        return self.current.update(dt, sensors, joystick)

    # --------------------------------------------------
    # Override stack management
    # --------------------------------------------------

    def push(self, state):
        """
        Temporarily override current state.

        Used for:
        - Manual control
        - Recovery
        - Safety reflexes

        Saves current state on stack. 
        """

        if self.current:
            # Save current context
            self.stack.append((self.current, self.index))

            # Exit current state
            self.current.exit()

        # Activate new override state
        self.current = state
        self.current.enter()

        # Disable autonomous sequencing
        self.auto_mode = False

    def pop(self):
        """
        Restore previous state from stack or restart sequence from beginning.

        Called when temporary state finishes and returns to auto mode when stack is empty.
        """

        if self.stack:

            # Exit override
            self.current.exit()

            # # Restore previous state
            # self.current, self.index = self.stack.pop()
            # self.current.enter()

            # Remove previous context from stack
            self.stack.pop()

        # If stack empty → return to auto
        if not self.stack:
            self.auto_mode = True
            # restart sequence from beginning 
            self.index = 0
            self.current = self.states[self.index]
            self.current.enter()

    # --------------------------------------------------
    # Main update interface
    # --------------------------------------------------

    def update(self, dt, sensors, joystick):
        """
        Main FSM update function.

        Called every control cycle.
        """

        # If running temporary state
        if not self.auto_mode:

            # Check if override finished
            if self.current.finished():
                self.pop()

        # Run correct update mode
        if self.auto_mode:
            return self.update_auto(dt, sensors, joystick)

        return self.current.update(dt, sensors, joystick)
