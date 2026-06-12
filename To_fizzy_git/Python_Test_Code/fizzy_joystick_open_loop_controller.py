import struct
import socket
import sys
import time
from pyjoystick.sdl2 import Key, Joystick, run_event_loop

def key_received(key: Key):
    global s
    if key.number == 2:
        val = -key.value
        print(val)
        try:
            s.sendto(struct.pack('Bf', 1, val), ('192.168.4.1', 4711))
        except:
            print('Communication error (you can terminate the app with <CTRL>-C)')
    
    if key.number == 5:
        val = key.value
        print(val)
        try:
            s.sendto(struct.pack('Bf', 1, val), ('192.168.4.1', 4711))
        except:
            print('Communication error (you can terminate the app with <CTRL>-C)')
        
    elif key.keytype == Key.BUTTON and key.number == 1 and key.value == 1:
        s.sendto(struct.pack('Bf', 1, 0), ('192.168.4.1', 4711))
        print('Bye ...')
        sys.exit()

def print_remove(key):
    print(f'Removed device: {key}')
    
def print_add(key):
    print(f'Added device: {key}')

if __name__ == '__main__':
    # setup UDP
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    s.bind(("0.0.0.0", 4711))
    s.settimeout(2)
    run_event_loop(print_add, print_remove, key_received)

