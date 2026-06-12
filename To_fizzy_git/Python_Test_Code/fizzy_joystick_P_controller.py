from collections.abc import Callable, Iterable, Mapping
import struct
import socket
import sys
import time
from typing import Any
from pyjoystick.sdl2 import Key, Joystick, run_event_loop
from threading import Thread
import numpy as np


class XboxThread(Thread):   # creates a class that recieves xboc controller data on an eventloop bases and returns those values
    def __init__(self) -> None:
        super().__init__()
        self.__value = 0
        self.terminate1 = 0

    def print_remove(self, key):
        print(f'Removed device: {key}')
        
    def print_add(self, key):
        print(f'Added device: {key}')

    def run(self):
        run_event_loop(self.print_add, self.print_remove, self.key_received)

    def key_received(self, key: Key):
        if key.number == 4:
            self.__value = key.value
        elif key.keytype == Key.BUTTON and key.number == 1 and key.value == 1:
            self.terminate1 = 1
            #print(self.__value)
    
    def terminate(self):
        return self.terminate1

    def get_value(self) -> float:
        return self.__value
    


thread = XboxThread()
thread.daemon = True # cleaning the receive buffer of the socket
thread.start()

# setup UDP
s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
s.bind(("0.0.0.0", 4711))
s.settimeout(2)
s.sendto(struct.pack('Bf', 0xff, 0), ('192.168.4.1', 4711))
initial_time = time.time()
s.recv(1000)

roll_measurement = 0                  # roll angle is shifted 90 degrees to pendulum roll angle, due to upright IMU
desired_minimal_cycle_time = 0.01  # this value is used to limit really fast cylcle times 

while True:
    start = time.time()

    # exit the loop by pressing B on the xbox controller
    if thread.terminate() == 1:
        s.sendto(struct.pack('Bf', 1, 0), ('192.168.4.1', 4711))
        print('Bye ...')
        sys.exit()


    ### Control part is copied from Heikes code in the Lego Mindstorm version ###
    K_P = 0.5           #gain
    limitpower = 0.8    #limits the speed, don't know if it is needed but maybe good to start with

    refrollangle = thread.get_value()*1.5 #degrees, now it controls half the operating area reference roll angle to which it should move (change to joystick variable wenn it works)
    # print('refrollangle:', round(refrollangle,2))
    

    roll = np.sin(roll_measurement*np.pi/180) # 180 and -180 at the bottom and 0 when standing upright, calibrate the IMU data to the orientation of the drivetrain 
   

    rollerror = refrollangle-roll

    rawpower = float(K_P*rollerror) 


    #saturate the motor:
    if rawpower>limitpower:
        satpower=limitpower
    elif rawpower<-limitpower:
        satpower=-limitpower
    else:
        satpower=rawpower

   

    try:
        s.sendto(struct.pack('Bf', 1, satpower), ('192.168.4.1', 4711))
        # s.sendto(struct.pack('Bf', 1, 0), ('192.168.4.1', 4711)) # idel motor
    except:
        print('Communication error (you can terminate the app with <CTRL>-C)')
    
    try:
        rec_list = list(struct.unpack(16*'f'+'q', s.recv(200)))
        
    except:
        print("dropout")

    
    yaw_measurement = rec_list[0]
    pitch_measurment = rec_list[1]
    roll_measurement = rec_list[2]
    
    time_on_PCB = rec_list[-1]/1_000_000
       

        
    ## making sure there is a minimal cycle time ##
    endtime = time.time()
    cycle_time = endtime-start
    # print('time in python',start-initial_time)
    
    if endtime-start < desired_minimal_cycle_time:
        # print('processing time', endtime-start)
        # print("sleep")
        time.sleep(desired_minimal_cycle_time-(endtime-start))

    
    