import socket
import struct
import time

class Fizzy:
    """
    This is a class to for communication with Fizzy.
    It handles the actual set of functions available on Fizzy.
    In general there are to way to communicate:
    1. polling data on request
    2. downlink, which can be triggered to initiate Fizzy to send data-frames as soon as they are available. The Downlink frequency is about 104Hz.

    :param ip: IP address of Fizzy, defaults to '192.168.4.1'
    :type ip: str, optional
    :param port: UDP port of Fizzy, defaults to 4711
    :type port: int, optional
    :param timeout: timeout for UDP requests in seconds, defaults to 2
    :type timeout: int, optional
    """
    
    def __init__(self, ip = '192.168.4.1', port = 4711, timeout = 0.1):
        """
        Constructor method. Ultra-short timeout (0.1s) prevents UI freezing on disconnect.
        """
        self.port = port
        self.ip = ip
        self.timeout = timeout
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.sock.bind(("0.0.0.0", port))
        self.sock.settimeout(timeout)
        self.is_connected = True
        self.consecutive_timeouts = 0
        self.timeout_threshold = 5  # Detect disconnect after 5 consecutive timeouts (~0.5s) instead of 1
                                     # This provides better tolerance for UDP packet loss and network jitter
                                     # while still being responsive (0.5s is still < typical user perception of lag)
        self.downlink_active = False  # Track if downlink mode is currently active
        self.downlink_restart_needed = False  # Flag to trigger downlink restart on reconnect
        self.downlink_restart_last_attempt_time = 0  # Track when last restart attempt was made
        self.downlink_restart_attempts = 0  # Count of restart attempts
        self.downlink_restart_max_attempts = 10  # Maximum restart attempts before giving up

    def decode(self, data):
        """
        Decode raw udp dataframe to list.

        :param data: UDP dataframe 
        :type data: byte[64]
        :return: [timestamp, motor-speed, v_bat,
                    q1, q2, q3, q4,
                    lin. acc x, lin. acc y, lin.acc z,
                    ACC_x_raw, ACC_y_raw, ACC_z_raw,
                    GYRO_x_raw, GYRO_y_raw, GYRO_z_raw,
                    MAG_x_raw, MAG_y_RAW, MAG_z_raw,
                    quality_mag]
        :rtype: [int64_t, float, float,
                    float, float, float, float,
                    float, float, float,
                    int16_t, int16_t, int16_t,
                    int16_t, int16_t, int16_t,
                    int16_t, int16_t, int16_t,
                    uint8_t]
        """
        return list(struct.unpack('<qfffffffffhhhhhhhhhB', data))
    
    def flush_buffer(self):
        """
        Flush all stale data from the UDP receive buffer.
        Useful after reconnection to clear out old packets.
        """
        self.sock.setblocking(False)
        try:
            while True:
                self.sock.recv(200)
        except BlockingIOError:
            pass
        finally:
            self.sock.setblocking(True)
            self.sock.settimeout(self.timeout)
    
    def stop(self):
        """
        1. stops the motor.
        2. stops data downlink (if running)
        """
        self.sock.sendto(struct.pack('Bf', 0, 0), (self.ip, self.port))

    def set_motor(self, value):
        """
        Set new motor speed (Range -1.0 to 1.0).
        Returns actual data-frame or -1 on timeout/error.
        Non-blocking to prevent UI freeze during disconnect.

        :param value: speed [-1.0 to 1.0]
        :type value: float
        :return: [timestamp, motor-speed, v_bat, ...] or -1
        """
        try:
            self.sock.sendto(struct.pack('Bf', 1, value), (self.ip, self.port))
            try:
                data = self.sock.recv(200)
                # Update connection state on successful receive
                if self.consecutive_timeouts > 0:
                    self.consecutive_timeouts = 0
                    if not self.is_connected:
                        print("Fizzy reconnected")
                        self.is_connected = True
                        self.downlink_restart_needed = False
                        self.downlink_restart_attempts = 0  # Reset restart attempts on successful reconnection
                        self.flush_buffer()
                # Clear downlink restart flag on any successful data reception
                if self.downlink_restart_needed:
                    print("Downlink restart successful")
                    self.downlink_restart_needed = False
                    self.downlink_restart_attempts = 0
                return self.decode(data)
            except socket.timeout:
                self.consecutive_timeouts += 1
                if self.consecutive_timeouts >= self.timeout_threshold and self.is_connected:
                    print(f"Fizzy connection lost (timeout #{self.consecutive_timeouts})")
                    self.is_connected = False
                    self.downlink_restart_needed = True  # Mark that we need to restart downlink
                return -1
        except Exception as e:
            self.consecutive_timeouts += 1
            if self.consecutive_timeouts >= self.timeout_threshold and self.is_connected:
                print(f"Fizzy set_motor error: {e}")
                self.is_connected = False
                self.downlink_restart_needed = True
            return -1
    
    def start_downlink(self):
        """
        Starts data-downlink.
        Forces Fizzy to fire data-frames at ~104Hz.
        To stop data-downlink use the "stop()" command.

        Note: Data-frames will pile up in the UDP-receive buffer (FIFO) on the local computer.
        Take care to pull them out or flush the buffer frequently. 
        """
        try:
            self.sock.sendto(struct.pack('Bf', 2, 0), (self.ip, self.port))
            self.downlink_active = True
            # Don't clear downlink_restart_needed here - only clear it when we actually receive data
        except Exception as e:
            print(f"Error starting downlink: {e}")

    def get_data(self):
        """
        Get data-frame from Fizzy in non-downlink mode.

        :return: [timestamp, motor-speed, v_bat,
                    q1, q2, q3, q4,
                    lin. acc x, lin. acc y, lin.acc z,
                    ACC_x_raw, ACC_y_raw, ACC_z_raw,
                    GYRO_x_raw, GYRO_y_raw, GYRO_z_raw,
                    MAG_x_raw, MAG_y_RAW, MAG_z_raw,
                    quality_mag]
        :rtype: [int64_t, float, float,
                    float, float, float, float,
                    float, float, float,
                    int16_t, int16_t, int16_t,
                    int16_t, int16_t, int16_t,
                    int16_t, int16_t, int16_t,
                    uint8_t]
        """
        self.sock.sendto(struct.pack('Bf', 66, 0), (self.ip, self.port))
        try:
            data = self.sock.recv(200)
        except:
            return -1
        return self.decode(data)
    
    def get_data_downlink(self):
        """
        Get data-frame from Fizzy in downlink mode.

        Note: v_bat will be 0 in downlink mode.

        :return: [timestamp, motor-speed, v_bat,
                    q1, q2, q3, q4,
                    lin. acc x, lin. acc y, lin.acc z,
                    ACC_x_raw, ACC_y_raw, ACC_z_raw,
                    GYRO_x_raw, GYRO_y_raw, GYRO_z_raw,
                    MAG_x_raw, MAG_y_RAW, MAG_z_raw,
                    quality_mag]
        :rtype: [int64_t, float, float,
                    float, float, float, float,
                    float, float, float,
                    int16_t, int16_t, int16_t,
                    int16_t, int16_t, int16_t,
                    int16_t, int16_t, int16_t,
                    uint8_t]
        """
        try:
            data = self.sock.recv(200)
            # Successfully received data - reset timeout counter
            if self.consecutive_timeouts > 0:
                self.consecutive_timeouts = 0
                if not self.is_connected:
                    print("Fizzy reconnected")
                    self.is_connected = True
                    self.downlink_restart_needed = False
                    self.downlink_restart_attempts = 0  # Reset restart attempts on successful reconnection
                    self.flush_buffer()
            # Clear downlink restart flag on any successful data reception
            if self.downlink_restart_needed:
                print("Downlink restart successful")
                self.downlink_restart_needed = False
                self.downlink_restart_attempts = 0
            return self.decode(data)
        except socket.timeout:
            self.consecutive_timeouts += 1
            if self.consecutive_timeouts >= self.timeout_threshold and self.is_connected:
                print(f"Fizzy connection lost (timeout #{self.consecutive_timeouts})")
                self.is_connected = False
                self.downlink_restart_needed = True  # Mark that we need to restart downlink
            return -1
        except Exception as e:
            self.consecutive_timeouts += 1
            if self.consecutive_timeouts >= self.timeout_threshold and self.is_connected:
                print(f"Fizzy connection error: {e}")
                self.is_connected = False
                self.downlink_restart_needed = True
            return -1
    
    def check_and_restart_downlink(self):
        """
        Check if downlink needs to be restarted (e.g., after reconnection).
        This should be called periodically from the main thread/timer.
        Attempts restart once per second for up to 10 tries (10 second window).
        Proactively attempts to restart downlink if needed, even if connection state is uncertain.
        """
        if self.downlink_restart_needed:
            current_time = time.time()
            # Only attempt restart every 1 second
            if current_time - self.downlink_restart_last_attempt_time >= 1.0:
                if self.downlink_restart_attempts < self.downlink_restart_max_attempts:
                    print(f"Attempting to restart downlink... (attempt {self.downlink_restart_attempts + 1}/{self.downlink_restart_max_attempts})")
                    self.start_downlink()
                    self.downlink_restart_attempts += 1
                    self.downlink_restart_last_attempt_time = current_time
                else:
                    # Give up after max attempts
                    print(f"Failed to restart downlink after {self.downlink_restart_max_attempts} attempts. Giving up.")
                    self.downlink_restart_needed = False
                    self.downlink_restart_attempts = 0
    
    def get_firmware_version(self):
        """
        Requests the actual firmware version of the connected Fizzy.
        
        :return: git-version (FW)
        :rtype: string 
        """
        self.sock.sendto(struct.pack('Bf', 0xff, 0), (self.ip, self.port))
        try:
            data = self.sock.recv(200)
        except:
            return -1
        return data.decode('utf-8')

    def flush_receive_buffer(self):
        """
        Flushes the UDP-receive buffer on the local computer.

        Note: this will take at least the time of UDP timeout (>2s)

        :return: number of data-frames that has been deleted
        :rtype: int
        """
        cnt = 0
        while(isinstance(self.get_data_downlink(), list)):
            cnt +=1
        return cnt