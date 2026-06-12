
"""
fizzy_main_combined.py
======================

Single entry point that merges the two former scripts:

  * fizzy_main.py         -> "basic" mode: manual LT/RT motor control, Xbox A
                             starts/stops a recording, motor command is only
                             *logged* (visualisation + data recording tool).
  * fizzy_main_random.py  -> "random/actions" mode: same gizmo + recording, but
                             also drives the motor via the GUI Actions panel
                             (Random / Wiggle / Forward state classes) and an
                             Xbox-A "random wiggle" toggle. Motor commands are
                             actually sent to the hardware.

Difference between the two originals
------------------------------------
  fizzy_main.py:
    - Does NOT create a Fizzy() itself; the dashboard owns the connection.
    - dashboard.set_motor() only stores the value for the recording log; the
      motor is never actuated.
    - Xbox A = start recording, X = cancel, Y = stop.
    - No action states, no random mode.

  fizzy_main_random.py:
    - Creates Fizzy() explicitly and starts the downlink before the dashboard.
    - Sends fizzy.set_motor(satpower) every loop -> the motor is actuated.
    - Xbox A = toggle a random-wiggle generator (smoothed random power).
    - GUI Actions panel can run Random / Wiggle / Forward state classes.

This combined file is a superset: it owns the Fizzy connection and actuates the
motor (like the random version), supports the GUI Actions panel and the random
wiggle toggle, AND keeps the recording callbacks. The single behavioural choice
that the two originals disagreed on -- what the Xbox A button does -- is exposed
as the XBOX_A_BUTTON setting below.

Controls
--------
  B button            : exit
  LT / RT             : manual motor power (when no GUI action / random is active)
  A button            : configurable -> see XBOX_A_BUTTON
  X button            : cancel recording
  Y button            : stop recording
  D-pad up/right/down/left : recording markers 1..4
  GUI Actions panel   : Random / Wiggle / Forward (overrides manual + random)
  GUI Record button   : start/stop recording (always available)
"""

import sys
import os

# Import control logic dependencies first
from utilities.fizzy_udp import Fizzy
import time
import random
from typing import Any
from pyjoystick.sdl2 import Key, Joystick, run_event_loop
from threading import Thread
import numpy as np
import matplotlib.pyplot as plt
import utilities.fizzy_config as fizzy_config_module
from utilities.fizzy_config import *

# Action state classes (used by the GUI Actions panel dispatch)
from states.random import Random as RandomState
from states.wiggle import Wiggle as WiggleState
from states.forward import Forward as ForwardState

# Import PyQt6 for integrated gizmo visualization
from PyQt6 import QtWidgets, QtCore
from fizzy_imu_dashboard import FizzyIMUDashboard

# Import for signal handling
import signal

# ---------------------------------------------------------------------------
# Behaviour selection
# ---------------------------------------------------------------------------
# What should the Xbox "A" button do?
#   "toggle_random"  -> toggle the random-wiggle generator
#   "start_recording"-> start a recording
XBOX_A_BUTTON = "toggle_random"

print("Initializing Fizzy control system with IMU 3D visualization (COMBINED mode)...")


# ---------------------------------------------------------------------------
# Runtime parameters consumed by the action state classes.
# ---------------------------------------------------------------------------
def _ensure_config_defaults(cfg):
    # --- Random ---
    if not hasattr(cfg, "random_gain"):
        cfg.random_gain = 0.9
    if not hasattr(cfg, "random_chaos"):
        cfg.random_chaos = 1.5
    if not hasattr(cfg, "random_smooth"):
        cfg.random_smooth = 0.85
    if not hasattr(cfg, "random_deadzone"):
        cfg.random_deadzone = 0.2

    # --- Forward ---
    if not hasattr(cfg, "Kp"):
        cfg.Kp = 1.5
    if not hasattr(cfg, "time_backwards"):
        cfg.time_backwards = 0.25
    if not hasattr(cfg, "time_forwards"):
        cfg.time_forwards = 0.35
    if not hasattr(cfg, "cycle_duration_roll_forward"):
        cfg.cycle_duration_roll_forward = 1.2

_ensure_config_defaults(fizzy_config_module)


class XboxThread(Thread):   # receives xbox controller data on an event-loop basis and exposes those values
    def __init__(self) -> None:
        super().__init__()
        self.__value = 0
        self.x_axis_right = 0
        self.y_axis_right = 0
        self.x_axis_left = 0
        self.y_axis_left = 0
        self.state = 0
        self.terminate = 0
        self.shutdown_requested = False

        # Random-mode toggle (A button, when XBOX_A_BUTTON == "toggle_random")
        self.random_mode = False

        # Recording button callbacks (bound by the dashboard).
        self.on_record_start = None
        self.on_record_stop = None
        self.on_record_cancel = None
        self.on_marker_1 = None
        self.on_marker_2 = None
        self.on_marker_3 = None
        self.on_marker_4 = None

        # Joystick for rumble
        self.joystick = None

    def print_remove(self, key):
        print(f'Removed device: {key}')

    def print_add(self, key):
        print(f'Added device: {key}')
        if hasattr(key, 'joystick'):
            self.joystick = key.joystick

    def run(self):
        try:
            run_event_loop(self.print_add, self.print_remove, self.key_received)
        except Exception as e:
            print(f"Gamepad thread error: {e}")
        finally:
            print("Gamepad thread terminated")

    def request_shutdown(self):
        """Request the thread to shut down gracefully"""
        self.shutdown_requested = True
        self.terminate = 1

    def key_received(self, key: Key):
        ### DOCUMENTATION:
        # Key.AXIS:   2 = LT (0..1), 5 = RT (0..1)
        # Key.BUTTON: 0 = A, 1 = B (exit), 2 = X, 3 = Y

        # LT / RT analog triggers -> manual motor power
        if key.keytype == Key.AXIS and key.number == 2:   # LT
            self.y_axis_left = -key.value - 0.05           # offset for resistance difference
        elif key.keytype == Key.AXIS and key.number == 5: # RT
            self.y_axis_left = key.value

        # A button: configurable behaviour
        elif key.keytype == Key.BUTTON and key.number == 0 and key.value == 1:
            if XBOX_A_BUTTON == "toggle_random":
                self.random_mode = not self.random_mode
                print(f"[A] Random mode: {'ON' if self.random_mode else 'OFF'}")
            elif XBOX_A_BUTTON == "start_recording":
                if self.on_record_start:
                    self.on_record_start()

        # B button: exit
        elif key.keytype == Key.BUTTON and key.number == 1 and key.value == 1:
            self.terminate = 1

        # X button: cancel recording
        elif key.keytype == Key.BUTTON and key.number == 2 and key.value == 1:
            if self.on_record_cancel:
                self.on_record_cancel()

        # Y button: stop recording
        elif key.keytype == Key.BUTTON and key.number == 3 and key.value == 1:
            if self.on_record_stop:
                self.on_record_stop()

        # D-pad markers: up/right/down/left -> marker_1..marker_4
        elif key.keytype == Key.HAT and not getattr(key, 'is_repeat', False):
            if key.value == Key.HAT_UP and self.on_marker_1:
                self.on_marker_1()
            elif key.value == Key.HAT_RIGHT and self.on_marker_2:
                self.on_marker_2()
            elif key.value == Key.HAT_DOWN and self.on_marker_3:
                self.on_marker_3()
            elif key.value == Key.HAT_LEFT and self.on_marker_4:
                self.on_marker_4()

    def exit(self):
        return self.terminate

    def case(self):
        return self.state

    def get_value(self) -> float:
        return self.__value

    def get_y_axis_left(self) -> float:
        return self.y_axis_left

    def get_x_axis_right(self) -> float:
        return self.x_axis_right

    def get_y_axis_right(self) -> float:
        return self.y_axis_right

    def is_random_mode(self) -> bool:
        return self.random_mode

    def reset(self):
        self.__value = 0
        self.x_axis_right = 0
        self.y_axis_right = 0
        self.x_axis_left = 0
        self.y_axis_left = 0
        self.state = 0

thread = XboxThread()
thread.daemon = True
thread.start()

# # ----------------------------------------------------------------------------------------

def extract_euler_from_packet(packet):
    """
    Given a Fizzy data packet, extract Euler angles (roll, pitch, yaw) in radians.

    Input packet: [timestamp, motor-speed, v_bat,
                   q1, q2, q3, q4,
                   lin. acc x/y/z, ACC_raw x/y/z,
                   GYRO_raw x/y/z, MAG_raw x/y/z, quality_mag]
    """
    qx, qy, qz, qw = packet[3], packet[4], packet[5], packet[6]
    w, x, y, z = qw, qx, qy, qz

    t0 = +2.0 * (w * x + y * z)
    t1 = +1.0 - 2.0 * (x * x + y * y)
    roll = np.arctan2(t0, t1)

    t2 = +2.0 * (w * y - z * x)
    t2 = np.clip(t2, -1.0, 1.0)
    pitch = np.arcsin(t2)

    t3 = +2.0 * (w * z + x * y)
    t4 = +1.0 - 2.0 * (y * y + z * z)
    yaw = np.arctan2(t3, t4) + np.pi

    return roll, pitch, yaw

# # ----------------------------------------------------------------------------------------

# Initialize communication (this combined script owns the Fizzy connection so it
# can actuate the motor, like the old fizzy_main_random.py).
fizzy = Fizzy()
fizzy.start_downlink()  # continuous data streaming (~104 Hz) — always on for gizmo

# PyQt app + gizmo window (this gives us the GUI record button + actions panel)
app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(sys.argv)
dashboard = FizzyIMUDashboard(fizzy=fizzy, xbox_thread=thread)
dashboard.show()

# # ----------------------------------------------------------------------------------------

def cleanup_and_exit(exit_code=0):
    """Properly clean up all resources and exit the program"""
    print("\nInitiating shutdown sequence...")
    try:
        thread.request_shutdown()
        print("Stopping motor and communication...")
        fizzy.stop()
        print("Closing GUI window...")
        dashboard.close()
        app.quit()
        print("Cleanup complete. Exiting...")
        sys.exit(exit_code)
    except Exception as e:
        print(f"Error during cleanup: {e}")
        sys.exit(1)

def signal_handler(signum, frame):
    """Handle system signals (Ctrl+C, etc.)"""
    print(f"\nReceived signal {signum}. Shutting down gracefully...")
    cleanup_and_exit(0)

signal.signal(signal.SIGINT, signal_handler)   # Handle Ctrl+C
try:
    signal.signal(signal.SIGTERM, signal_handler)  # Windows may not support SIGTERM
except AttributeError:
    pass

# Wait for first valid data packet (with timeout)
start_time_on_PCB = None
timeout_start = time.time()
while start_time_on_PCB is None and time.time() - timeout_start < 5.0:
    try:
        acquired = dashboard.data_lock.acquire(blocking=False)
        if acquired:
            try:
                if dashboard.data_buffer:
                    data = dashboard.data_buffer[-1]
                    if data and data != -1 and len(data) > 0:
                        start_time_on_PCB = data[0] / 1_000_000
            finally:
                dashboard.data_lock.release()
    except:
        pass
    if start_time_on_PCB is None:
        time.sleep(0.1)
        app.processEvents()

if start_time_on_PCB is None:
    print("Warning: Could not get initial timestamp from Fizzy within 5 seconds")
    start_time_on_PCB = 0

# # ----------------------------------------------------------------------------------------

# General loop parameters
limitpower = 0.95
satpower = 0.0
rawpower = 0.0
desired_minimal_cycle_time = 0.01   # limit really fast cycle times

plotting_data = False

# ---------------------------------------------------------------------------
# Action dispatch helpers (GUI Actions panel)
# ---------------------------------------------------------------------------
def _build_action(name):
    """Instantiate a fresh action state for the given name."""
    if name == "random":
        return RandomState(fizzy_config_module)
    if name == "wiggle":
        return WiggleState(fizzy_config_module)
    if name == "forward":
        return ForwardState(
            Kp=fizzy_config_module.Kp,
            time_backwards=fizzy_config_module.time_backwards,
            time_forwards=fizzy_config_module.time_forwards,
            cycle_duration_roll_forward=fizzy_config_module.cycle_duration_roll_forward,
        )
    return None

current_action_name = None
current_action = None
last_loop_time = time.time()

# --- Random wiggle parameters ---------------------------------------------
# Each "wiggle" is a random target motor power held for a random duration.
# Power is sampled uniformly in [-RANDOM_POWER_MAX, +RANDOM_POWER_MAX].
# Duration is sampled uniformly in [RANDOM_DUR_MIN, RANDOM_DUR_MAX] seconds.
# The output is smoothed by a first-order filter to avoid jerky motor jumps.
RANDOM_POWER_MAX = 0.6     # cap on |power| during random mode (<= limitpower)
RANDOM_DUR_MIN   = 0.4     # seconds, min hold time
RANDOM_DUR_MAX   = 1.5     # seconds, max hold time
RANDOM_SMOOTH    = 0.15    # 0..1, lower = smoother / slower transitions

# Random-mode runtime state
_rand_target = 0.0
_rand_segment_end = 0.0       # wall-clock time when current segment expires
_rand_was_active = False

def _new_random_segment(now):
    """Pick a new random target power + segment end time."""
    global _rand_target, _rand_segment_end
    _rand_target = random.uniform(-RANDOM_POWER_MAX, RANDOM_POWER_MAX)
    _rand_segment_end = now + random.uniform(RANDOM_DUR_MIN, RANDOM_DUR_MAX)

# --------------------------------------------------------------------------

if plotting_data:
    plt.ion()
    fig, axs = plt.subplots()
    x_data_plot = []
    y_data_plot = []
    line, = axs.plot(x_data_plot, y_data_plot)

    def update_plot():
        line.set_xdata(x_data_plot)
        line.set_ydata(y_data_plot)
        axs.relim()
        axs.autoscale_view()
        fig.canvas.draw()
        fig.canvas.flush_events()

# # ----------------------------------------------------------------------------------------

## Main loop ##
# NOTE: Data collection happens in the dashboard's background thread. We read
# packets from dashboard.data_buffer rather than calling fizzy directly, to
# avoid blocking the Qt main thread.
#
# rawpower priority:  GUI Actions panel  >  Xbox-A random wiggle  >  manual LT/RT
try:
    while True:
        start = time.time()
        dt = max(start - last_loop_time, 1e-4)
        last_loop_time = start

        # Send the most recently computed motor command (actuate + log)
        if fizzy.is_connected:
            fizzy.set_motor(satpower)
        dashboard.set_motor(satpower)

        fizzy.check_and_restart_downlink()

        # Pull most recent data packet from the dashboard's shared buffer
        data = None
        try:
            acquired = dashboard.data_lock.acquire(blocking=False)
            if acquired:
                try:
                    if dashboard.data_buffer:
                        data = dashboard.data_buffer[-1]
                finally:
                    dashboard.data_lock.release()
        except Exception as e:
            print(f"Error getting data from dashboard buffer: {e}")

        if data is None or data == -1 or not isinstance(data, (list, tuple)) or len(data) < 7:
            app.processEvents()
            time.sleep(0.001)
            continue

        roll, pitch, yaw = extract_euler_from_packet(data)
        angles_deg = np.degrees([roll, pitch, yaw])

        time_on_PCB = data[0] / 1_000_000
        timer_precise_on_PCB = time_on_PCB - start_time_on_PCB

        # Keep GUI responsive
        app.processEvents()

        # Exit
        if thread.exit() == 1:
            cleanup_and_exit(0)

        # --- Decide rawpower: GUI action panel > Xbox A random > manual ----
        requested_action = dashboard.actions_panel.active_action

        if requested_action != current_action_name:
            if current_action is not None:
                try:
                    current_action.exit()
                except Exception as e:
                    print(f"Error exiting action {current_action_name}: {e}")
            current_action_name = requested_action
            current_action = _build_action(requested_action)
            if current_action is not None:
                try:
                    current_action.enter()
                except Exception as e:
                    print(f"Error entering action {current_action_name}: {e}")
                    current_action = None
                    current_action_name = None

        if current_action is not None:
            sensors = {"roll": roll, "pitch": pitch, "yaw": yaw}
            try:
                rawpower = float(current_action.update(dt, sensors, joystick=None))
            except Exception as e:
                print(f"Error in action {current_action_name}.update(): {e}")
                rawpower = 0.0
        elif thread.is_random_mode():
            now = time.time()

            # On the first iteration after enabling random mode, seed a segment
            if not _rand_was_active:
                _new_random_segment(now)
                _rand_was_active = True

            # If the current segment is over, pick a new target
            if now >= _rand_segment_end:
                _new_random_segment(now)

            # Smooth toward the target so the motor doesn't jolt
            rawpower = rawpower + RANDOM_SMOOTH * (_rand_target - rawpower)
        else:
            # Random mode just got turned off -> reset smoothing state
            if _rand_was_active:
                _rand_was_active = False
                rawpower = 0.0

            # Manual control via LT / RT
            rawpower = thread.get_y_axis_left()

        # Saturate the motor
        if rawpower > limitpower:
            satpower = limitpower
        elif rawpower < -limitpower:
            satpower = -limitpower
        else:
            satpower = rawpower

        # Optional live plot of the commanded power
        if plotting_data:
            y_data_plot.append(satpower)
            x_data_plot.append(timer_precise_on_PCB)
            update_plot()
            axs.set_xlabel('time (s)')
            axs.set_ylabel('motor power')
            axs.set_title('Signal')
            plt.pause(0.01)

        # Enforce a minimum cycle time
        endtime = time.time()
        if endtime - start < desired_minimal_cycle_time:
            app.processEvents()
            time.sleep(desired_minimal_cycle_time - (endtime - start))

except KeyboardInterrupt:
    print("\nKeyboard interrupt received. Shutting down...")
    cleanup_and_exit(0)
except Exception as e:
    print(f"\nUnexpected error occurred: {e}")
    import traceback
    traceback.print_exc()
    cleanup_and_exit(1)
