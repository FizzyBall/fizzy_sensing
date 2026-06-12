"""
Global configuration parameters for Fizzy.
- Separate settings and runtime into two JSON files
- Atomic save with retry loop (important on Windows)
- Dict-like access for Streamlit
"""

import json
import os
import time

SETTINGS_FILE = "config_settings.json"
RUNTIME_FILE = "config_runtime.json"


class Config:
    _last_valid_settings = {}
    _last_valid_runtime = {}

    def __init__(self):
        # ------------------------
        # Loop timing
        # ------------------------
        self.MIN_CYCLE_TIME = 0.001

        # ------------------------
        # Motor limits
        # ------------------------
        self.POWER_LIMIT = 0.95

        # ------------------------
        # Control gains
        # ------------------------
        self.Kp = 0.7

        # ------------------------
        # Wiggle movement parameters
        # ------------------------
        self.T1 = 0.4
        self.T2 = 0.8
        self.A2 = 0.6
        self.A1 = -0.5 * self.A2

        # ------------------------
        # Vibrate movement parameters
        # ------------------------
        self.T1V = 0.2
        self.T2V = 0.2
        self.A2V = 0.7
        self.A1V = -0.5 * self.A2V

        # ------------------------
        # Forward/backward movement parameters
        # ------------------------
        self.time_backwards = 0.5
        self.time_forwards = 3
        self.cycle_duration_roll_forward = 5
        
        # ------------------------
        # Random state parameters
        # ------------------------
        self.random_gain = 0.8        # overall movement amplitude
        self.random_chaos = 1.0        # how often new random targets appear
        self.random_smooth = 0.92     # higher = smoother / slower changes
        self.random_deadzone = 0.2    # motor does not move below this

        # ------------------------
        # User-controlled settings
        # ------------------------
        self.difficulty = 1
        self.program = "standby"
        self.power = True
        self.sensitivity = 1


        # ------------------------
        # Runtime values
        # ------------------------
        self.taps = 0
        self.time_seconds = 0

        # ------------------------
        # Difficulty settings
        # ------------------------
        self.duration_walk = 5
        self.duration_table = 5

    # ------------------------------------------------------------------
    # Dict-like access
    # ------------------------------------------------------------------
    def __getitem__(self, key):
        return getattr(self, key)

    def __setitem__(self, key, value):
        setattr(self, key, value)

    def get(self, key, default=None):
        return getattr(self, key, default)

    # ------------------------------------------------------------------
    # Split between settings and runtime
    # ------------------------------------------------------------------
    def settings_dict(self):
        return {
            "difficulty": self.difficulty,
            "program": self.program,
            "power": self.power,
            "sensitivity": self.sensitivity,
        }

    def runtime_dict(self):
        return {
            "taps": self.taps,
            "time_seconds": self.time_seconds,
        }

    # ------------------------------------------------------------------
    # Robust atomic save with retry
    # ------------------------------------------------------------------
    def _atomic_save(self, filename, data):
        temp_file = filename + ".tmp"

        for _ in range(10):
            try:
                with open(temp_file, "w") as f:
                    json.dump(data, f, indent=2)

                os.replace(temp_file, filename)
                return

            except PermissionError:
                time.sleep(0.01)

        raise RuntimeError(f"Could not save {filename}")

    # ------------------------------------------------------------------
    # Load both files
    # ------------------------------------------------------------------
    def load(self):
        self._load_file(
            SETTINGS_FILE,
            Config._last_valid_settings,
            is_runtime=False
        )

        self._load_file(
            RUNTIME_FILE,
            Config._last_valid_runtime,
            is_runtime=True
        )

        self.recompute()

    def _load_file(self, filename, cache, is_runtime):
        if not os.path.exists(filename):
            with open(filename, "w") as f:
                json.dump({}, f)

        try:
            with open(filename, "r") as f:
                data = json.load(f)

            if not isinstance(data, dict):
                data = {}

            if is_runtime:
                Config._last_valid_runtime = data
            else:
                Config._last_valid_settings = data

        except (json.JSONDecodeError, OSError):
            data = (
                Config._last_valid_runtime
                if is_runtime
                else Config._last_valid_settings
            )

        for key, value in data.items():
            if hasattr(self, key):
                setattr(self, key, value)

    # ------------------------------------------------------------------
    # Save only settings or runtime
    # ------------------------------------------------------------------
    def save_settings(self):
        self.recompute()
        self._atomic_save(SETTINGS_FILE, self.settings_dict())

    def save_runtime(self):
        self._atomic_save(RUNTIME_FILE, self.runtime_dict())

    # ------------------------------------------------------------------
    # Convenient update helpers
    # ------------------------------------------------------------------
    def update_settings(self, **kwargs):
        self.load()

        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)

        self.save_settings()

    def update_runtime(self, **kwargs):
        self.load()

        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)

        self.save_runtime()

    # ------------------------------------------------------------------
    # Recompute dependent values
    # ------------------------------------------------------------------
    def recompute(self):
        if self.difficulty == 1:
            # walk
            self.duration_walk = 4
            self.duration_wait = 10
            self.Kp = 0.4
            
            # random
            self.random_gain = 0.8        # overall movement amplitude
            self.random_chaos = 1.0        # how often new random targets appear
            self.random_smooth = 0.92     # higher = smoother / slower changes
            self.random_deadzone = 0.4 

            # table
            self.duration_table = 5

        elif self.difficulty == 2:
            # walk
            self.duration_walk = 6
            self.duration_wait = 5
            self.Kp = 0.4
            
            # random
            self.random_gain = 0.8        # overall movement amplitude
            self.random_chaos = 4.0        # how often new random targets appear
            self.random_smooth = 0.92     # higher = smoother / slower changes
            self.random_deadzone = 0.6

            # table
            self.duration_table = 3

        else:
            # walk
            self.duration_walk = 7
            self.duration_wait = 3
            self.Kp = 0.4
            
            # random
            self.random_gain = 0.8        # overall movement amplitude
            self.random_chaos = 6.0        # how often new random targets appear
            self.random_smooth = 0.92     # higher = smoother / slower changes
            self.random_deadzone = 0.9 

            # table
            self.duration_table = 1

