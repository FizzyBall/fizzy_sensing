
import sys
import os

import time
from typing import Any
from pyjoystick.sdl2 import Key, Joystick, run_event_loop
from threading import Thread
import numpy as np
import matplotlib.pyplot as plt
from fizzy_config import *

from PyQt6 import QtWidgets, QtCore
from fizzy_imu_dashboard import FizzyIMUDashboard

import signal

print("Initializing Fizzy control system with IMU 3D visualization...")

## State Machine states: ##
# B = exit
# A = neutral (motor torque 0)
# X = big wiggle
# Y = vibrating
# Left joystick press = speed control (controlled with the left joystick back and forth)
# Right joystick press = directional control (controlled with the right joystick gives direction and magnitude)
# Left button (for index finger) = Position control on downward position
# Right button (for index finger) = Position control (controlled with the left joystick back and forth)


class XboxThread(Thread):   # creates a class that recieves xboc controller data on an eventloop bases and returns those values
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
        
        # Recording button callbacks
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
        # Store joystick reference for rumble control
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
        # Key.AXIS:         # 2 = LT, values between 0 and 1
                            # 5 = RT, values between 0 and 1 

        # Key.BUTTON:       # 0 = A
                            # 1 = B (exit button)
                            # 2 = X
                            # 3 = Y

        # levers left and right (LT and RT)
        if key.keytype == Key.AXIS and key.number == 2: # key number 2 LT, values between 0 and 1
            self.y_axis_left = -key.value - 0.05        # off set due to difference in resistance in direction
        elif key.keytype == Key.AXIS and key.number == 5: # key number 5 RT, values between 0 and 1
            self.y_axis_left = key.value 
        
        # A button: Start recording
        elif key.keytype == Key.BUTTON and key.number == 0 and key.value == 1:
            if self.on_record_start:
                self.on_record_start()
            
        # B button: Exit
        elif key.keytype == Key.BUTTON and key.number == 1 and key.value == 1: # key number 1 is 'B BUTTON'
            self.terminate = 1
        
        # X button: Cancel recording
        elif key.keytype == Key.BUTTON and key.number == 2 and key.value == 1:
            if self.on_record_cancel:
                self.on_record_cancel()
        
        # Y button: Stop recording
        elif key.keytype == Key.BUTTON and key.number == 3 and key.value == 1:
            if self.on_record_stop:
                self.on_record_stop()  

        # D-pad markers: up/right/down/left map to marker_1..marker_4
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
    
    def reset(self):
        self.__value = 0
        self.x_axis_right = 0
        self.y_axis_right = 0
        self.x_axis_left = 0
        self.y_axis_left = 0
        self.state = 0

thread = XboxThread()
thread.daemon = True # cleaning the receive buffer of the socket
thread.start()

# # ----------------------------------------------------------------------------------------

##Quaternions to earler angels function

def extract_euler_from_packet(packet):
    """
    Given a data packet with the specified format, extract Euler angles.
    Input packet: [timestamp, motor-speed, v_bat,
                    q1, q2, q3, q4,
                    lin. acc x, lin. acc y, lin.acc z,
                    ACC_x_raw, ACC_y_raw, ACC_z_raw,
                    GYRO_x_raw, GYRO_y_raw, GYRO_z_raw,
                    MAG_x_raw, MAG_y_RAW, MAG_z_raw,
                    quality_mag]
    Returns: (roll, pitch, yaw) in radians
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

# Initialize PyQt Application and Gizmo window
# NOTE: Fizzy instance is created and initialized by the dashboard, not here
# This prevents socket contention between multiple threads
app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(sys.argv)
dashboard = FizzyIMUDashboard(xbox_thread=thread)
dashboard.show()

# # ----------------------------------------------------------------------------------------

def cleanup_and_exit(exit_code=0):
    """Properly clean up all resources and exit the program"""
    print("\nInitiating shutdown sequence...")
    try:
        # Request gamepad thread to shut down
        thread.request_shutdown()
        
        # Close the gizmo window (this will stop the dashboard and its background threads)
        print("Closing GUI window...")
        dashboard.close()
        
        # Close the application
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

# Register signal handlers
signal.signal(signal.SIGINT, signal_handler)   # Handle Ctrl+C
try:
    signal.signal(signal.SIGTERM, signal_handler)  # Handle termination signal (Windows may not support this)
except AttributeError:
    pass  # SIGTERM might not be available on all platforms

# Get starting time (wait for first valid data with timeout)
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

# Parameters


'''
# to get out of singularities, the ball will do two things:

# 1. modify the reference direction, to get a sort of zigzag motion:
amplitude_zigzag= np.pi/10     # amplitude in rad, zigzagging the reference yaw direction to get out of singularities
freq_zigzag=2               # frequency in Hz, zigzagging reference direction to get the ball out of singularities

# 2. flap the block back and forth around the "down" configuration, to generate torques due to the dynamic imbalance:
cycle_duration=4.5  # cycle duration, in seconds, of back-and-forth flapping the motor to generate some yaw. Must be larger than duration_fast
duration_fast=2     # duration of the fast part of the cycle, in seconds (the rest remains for the slow movement)
K_p_high=0.5        # motor feedback gain to enforce fast motion in first part of cycle, in percent/rad
K_p_low=0.5          # motor feedback gain for slow motion in second part of cycle, percent/rad
'''


desired_minimal_cycle_time = 0.01  # this value is used to limit really fast cylcle times 


if plotting_data == True:
    # Set up plots
    plt.ion()  # Turn on interactive mode
    fig, axs = plt.subplots()
    data_plotting = 10
    # Deques to store the last 100 data points for plotting for each signal
    x_data_plot = []
    y_data_plot = []
    line, = axs.plot(x_data_plot, y_data_plot)

    # Function to update plot
    def update_plot():
        line.set_xdata(x_data_plot)
        line.set_ydata(y_data_plot)
        axs.relim()
        axs.autoscale_view()
        fig.canvas.draw()
        fig.canvas.flush_events()





# # ----------------------------------------------------------------------------------------

## Main loop with Joystick input as changing factor for the different states ##
# NOTE: Data collection happens in dashboard's background thread to avoid blocking the main Qt thread
# We read from the dashboard's buffer instead of directly from the socket
try:
    motor_command_timer = 0  # Timer to throttle motor commands
    motor_command_interval = 0.1  # Send motor command every 0.1 seconds (10 Hz)
    
    while True:
        start = time.time()
        
        # IMPORTANT: Don't call fizzy.set_motor() directly - let dashboard's background thread handle socket I/O
        # Instead, use the dashboard's set_motor() method which properly integrates with the background thread
        # Only update motor command periodically to avoid socket contention
        motor_command_timer += desired_minimal_cycle_time
        if motor_command_timer >= motor_command_interval:
            dashboard.set_motor(satpower)  # Store motor command for dashboard to handle
            motor_command_timer = 0
        
        # Get data from dashboard's buffer (populated by background thread)
        # This avoids blocking the main thread on socket I/O
        data = None
        try:
            # Use non-blocking lock to avoid contention
            acquired = dashboard.data_lock.acquire(blocking=False)
            if acquired:
                try:
                    if dashboard.data_buffer:
                        data = dashboard.data_buffer[-1]  # Get most recent packet
                finally:
                    dashboard.data_lock.release()
        except Exception as e:
            print(f"Error getting data from dashboard buffer: {e}")

        # Validate that data is a proper packet
        if data is None or data == -1 or not isinstance(data, (list, tuple)) or len(data) < 7:
            # No valid data - skip this iteration and process events
            app.processEvents()
            time.sleep(0.001)  # Small sleep to prevent busy-waiting
            continue

        roll, pitch, yaw = extract_euler_from_packet(data)
        angles_deg = np.degrees([roll, pitch, yaw]) # transforms to degree
        
        # yaw 0deg to 360deg 
        # pitch pedulum along -/+ z axis (hanging down and up) measures 0deg, horizontal one side 90deg and the other -90deg
        # roll pedulum along - z axis (hanging down) 0deg and along + z axis 180deg or -180deg
     
        time_on_PCB = data[0]/1_000_000
        timer_precise_on_PCB = time_on_PCB-start_time_on_PCB
        
        # Process PyQt events to keep GUI responsive and updated
        app.processEvents()
       

        ## Different states: ##
        # ----------------------------------------------------------------------------------
        # exit
        if thread.exit() == 1:
            cleanup_and_exit(0)

        # -----------------------------------------------------------------------------------
        # trigger-based speed control (LT and RT)
        else:
            rawpower = thread.get_y_axis_left()




        # saturate the motor:
        if rawpower>limitpower:
            satpower=limitpower
        elif rawpower<-limitpower:
            satpower=-limitpower
        else:
            satpower=rawpower
        # print(satpower)

         # plotting desired data 
        if plotting_data == True:
       
            y_data_plot.append(satpower)

            x_data_plot.append(timer_precise_on_PCB)
            update_plot()
            axs.set_xlabel('X Axis')
            axs.set_ylabel('Y Axis')
            axs.set_title('Signal ')
            plt.pause(0.01)


        # Making sure that the cycle time is running with a minimum cycle time.
        endtime = time.time()
        cycle_time = endtime-start
        
        if endtime-start < desired_minimal_cycle_time:
            # print('processing time', endtime-start)
            # Process PyQt events to keep GUI responsive
            app.processEvents()
            time.sleep(desired_minimal_cycle_time-(endtime-start))

except KeyboardInterrupt:
    print("\nKeyboard interrupt received. Shutting down...")
    cleanup_and_exit(0)
except Exception as e:
    print(f"\nUnexpected error occurred: {e}")
    import traceback
    traceback.print_exc()
    cleanup_and_exit(1)