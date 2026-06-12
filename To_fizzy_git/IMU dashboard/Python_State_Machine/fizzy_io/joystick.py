from pyjoystick.sdl2 import Key, Joystick, run_event_loop
from threading import Thread

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

    def print_remove(self, key):
        print(f'Removed device: {key}')
        
    def print_add(self, key):
        print(f'Added device: {key}')

    def run(self):
        run_event_loop(self.print_add, self.print_remove, self.key_received)

    def key_received(self, key: Key):
        ### DOCUMENTATION:
        # Key.AXIS:         # 0 = x-axis left joystick, Left (-1) and Right (1)
                            # 1 = y-axis left joystick, up (-1) and down (1)
                            # 2 = LT, values between 0 and 1
                            # 3 = x-axis right joystick, 
                            # 4 = y-axis right joystick 
                            # 5 = RT, values between 0 and 1 

        # Key.BUTTON:       # 0 = A, 
                            # 1 = B, 
                            # 2 = X,
                            # 3 = Y,
                            # 4 = LB,
                            # 5 = RB,
                            # 6 = from the 3 small buttons in the middle the left one
                            # 7 = ?from the 3 small buttons in the middle the right one
                            # 8 = pressing Left joystick,
                            # 9 = pressing Right joystick,

        # Key.HAT:          # key.number = 0
                            # key.value: 
                                # 0 = CENTER
                                # 1 = UP
                                # 4 = DOWN
                                # 8 = LEFT
                                # 2 = RIGHT


        # right joystick
        if key.keytype == Key.AXIS and key.number == 3: # key number 3 is Right Joystick Left (-1) and Right (1)
            self.x_axis_right = key.value
        elif key.keytype == Key.AXIS and key.number == 4: # key number 4 is Right Joystick up (-1) and down (1)
            self.y_axis_right = key.value

        # left joystick
        elif key.keytype == Key.AXIS and key.number == 0: # key number 0 is Right Joystick left (-1) and right (1)
            self.x_axis_left = key.value
        elif key.keytype == Key.AXIS and key.number == 1: # key number 1 is Right Joystick up (-1) and down (1)
            self.y_axis_left = key.value  
        
        # levers left and right (LT and RT)
        elif key.keytype == Key.AXIS and key.number == 2: # key number 2 LT, values between 0 and 1
            self.y_axis_left = -key.value - 0.05        # off set due to difference in resistance in direction
        elif key.keytype == Key.AXIS and key.number == 5: # key number 5 RT, values between 0 and 1
            self.y_axis_left = key.value 
            

        # termination key B
        elif key.keytype == Key.BUTTON and key.number == 1 and key.value == 1: # key number 1 is 'B BUTTON'
            self.terminate = 1

        # other keys
        elif key.keytype == Key.BUTTON and key.number == 0 and key.value ==1: # key number 0 is 'A BUTTON'
            self.state = 1
        elif key.keytype == Key.BUTTON and key.number == 2 and key.value ==1: # key number 2 is 'X BUTTON'
            self.state = 2
        elif key.keytype == Key.BUTTON and key.number == 3 and key.value ==1: # key number 3 is 'Y BUTTON'
            self.state = 3
        elif key.keytype == Key.BUTTON and key.number == 4 and key.value ==1: # key number 4 is 'LB BUTTON'
            self.state = 4
        elif key.keytype == Key.BUTTON and key.number == 5 and key.value ==1: # key number 5 is 'RB BUTTON'
            self.state = 5
        
        elif key.keytype == Key.BUTTON and key.number == 7 and key.value ==1: # key number 8 is 'pressing Left joystick'
            self.state = 7 

        elif key.keytype == Key.BUTTON and key.number == 8 and key.value ==1: # key number 8 is 'pressing Left joystick'
            self.state = 8 
        elif key.keytype == Key.BUTTON and key.number == 9 and key.value ==1: # key number 9 is 'pressing Right joystick'
            self.state = 9 
        
        elif key.keytype == Key.HAT and key.number == 0 and key.value ==1: # key number 10 is 'pressing HAT UP key'
            self.state = 10
        elif key.keytype == Key.HAT and key.number == 0 and key.value ==4: # key number 10 is 'pressing HAT DOWN key'
            self.state = 11
            
            # To map out joystick keys uncomment lines below
        # else:
        #     print(key.typ,key.number,key.value)  

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