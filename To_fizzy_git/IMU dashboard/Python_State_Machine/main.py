"""
Main robot control program.

Responsibilities:
-----------------
1. Initialize hardware
2. Read sensors
3. Run FSM
4. Handle interrupts
5. Send motor commands
6. loads and writes to JSON file to communicate with dashboard.py
"""


import time
import numpy as np

from fizzy_udp import Fizzy


from core.fsm import StateMachine
from core.interrupts import Interrupts

from states.neutral import Neutral
from states.wiggle import Wiggle
from states.vibrate import Vibrate
from states.random import Random
from states.manual import Manual
from states.zero_stand import Zero_stand
from states.forward import Forward 
from states.backward import Backward 

from fizzy_io.imu_angle import extract_euler
from fizzy_io.imu_acc import extract_acc
from fizzy_io.joystick import XboxThread

from config import Config



def program_sequence(program, config):
    if program == "walk":
        return [
            Backward(config, duration=config.duration_walk),
            Zero_stand(config, duration=1),
            Neutral(duration=config.duration_wait),
        ]

    elif program == "balance":
        return [
            Random(config, duration=2),
        ]

    elif program == "table":
        return [
            Backward(config, duration=6),
            Wiggle(config, duration=config.duration_table)
        ]
    elif program == "standby":
        return [
            Vibrate(config, duration=1),
            Neutral(duration=5),
        ]


def main():
    # Time at which the current program started
    program_start_time = time.time()
    taps = 0

    # Start joystick thread
    joystick = XboxThread()
    joystick.daemon = True
    joystick.start()

    # Hardware interface
    fizzy = Fizzy()

    # Parameters
    config = Config()
    config.load()
    
    # Make sure the main.py can run by itself
    config.update_settings (power = True)

    # Reset the dashboard values when fizzy starts
    config.save_runtime()

    # save last data storage
    _last_valid_data = {}

    # FSM
    current_program = config.program

    fsm = StateMachine(program_sequence(current_program, config))
    fsm.start()

    interrupts = Interrupts()

    last = time.time()

    # Main loop
    while True:

        now = time.time()
        dt = now - last
        last = now

        config.load()

        if config.program != current_program:
            print(f"Switching to {config.program}")
            current_program = config.program

            # Reset counters for the newly selected program
            taps = 0
            program_start_time = time.time()

            # Update dashboard immediately
            config.update_runtime(taps=taps, time_seconds=0)

            fsm = StateMachine(program_sequence(current_program, config))
            fsm.start()

        # Read IMU
        try:
            data = fizzy.get_data()
            
        except Exception as e:
            print("IMU read failed:", e) # Communication dropout try again and print error
            continue

        try:
            roll, pitch, yaw = extract_euler(data)    # place in the try loop to make the connection loss not vital
        except:
            data = _last_valid_data
            roll, pitch, yaw = extract_euler(data)

        acc_mag = extract_acc(data)

        sensors = {
            "roll": roll,
            "pitch": pitch,
            "yaw": yaw,
            "acc_mag": acc_mag
        }

       
        # Handle interrupts
        event = interrupts.check(joystick, sensors, config)

        # Exit program
        if event == "EXIT":
            fizzy.stop()
            print('Bye ...')
            break

        # Shutdown check via JSON
        if getattr(config, "power", True) is False:
            fizzy.stop()
            print("Power OFF detected in JSON. Bye...")
            break

        # IMU input correction
        elif event == "TAP":
            if fsm.auto_mode:
                fsm.push(Zero_stand(config, duration=1))
                # Increase tap counter and write it to the JSON file
                taps += 1
                print(f"Taps: {taps}")
                config.update_runtime(taps=taps)

        # Manual override
        elif event == "MANUAL":
            if not isinstance(fsm.current, Manual):
                fsm.push(Manual())

        # Motor off
        elif event == "NEUTRAL":
            config.program = "balance"
            fsm.push(Neutral(duration=1))

        # Resume autonomy
        elif event == "RESUME":
            while not fsm.auto_mode:
                fsm.pop()

        # Run FSM
        power = fsm.update(dt, sensors, joystick)

        # Saturate
        power = np.clip(power, -config.POWER_LIMIT, config.POWER_LIMIT)

        # Send command
        fizzy.set_motor(power)

        # Update elapsed program time in the JSON file
        elapsed_time = time.time() - program_start_time
        config.update_runtime(time_seconds=elapsed_time)

        # save old data
        _last_valid_data = data

        # Making sure that the cycle time is running with a minimum cycle time.
        endtime = time.time()
        cycle_duration = endtime-last
        if cycle_duration < config.MIN_CYCLE_TIME:
            time.sleep(config.MIN_CYCLE_TIME-(cycle_duration))

if __name__ == "__main__":
    main()
