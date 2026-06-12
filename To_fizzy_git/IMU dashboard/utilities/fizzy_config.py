DOWNLINK_MODE = True  # True: ~104 Hz (continuous streaming), False: ~33 Hz (polling)
plotting_data = False

CERTAINTY_THRESHOLD = 50  # Threshold for accepting a prediction based on its certainty (confidence) 0 - 100
CERTAINTY_DIFFERENCE_THRESHOLD = 20  # Threshold for the difference between top two prediction certainties 0 - 100

# ============ CLASSIFICATION LABELS ============
# Centralized configuration for classification labels
# Add new labels here to automatically update everywhere in the codebase
CLASSIFICATION_LABELS = ['Idle', 'Tap', 'Lift', 'Drop', 'Kick', 'Shake', 'Roll', 'Spin', 'Bounce', 'Unknown']

# Special label for unclassified data (gets separate folder in interactive_labeler)
UNKNOWN_LABEL = 'Unknown'

# Color configuration for labels (RGBA format with alpha for transparency)
LABEL_COLORS = {
    'Idle': (255, 255, 255, 30),    # White with alpha
    'Tap': (0, 255, 0, 30),        # Green with alpha
    'Lift': (255, 165, 0, 30),     # Orange with alpha
    'Drop': (0, 0, 255, 30),          # Blue with alpha
    'Kick': (128,0,128,30),
    'Shake': (0, 128, 128,30),
    'Roll': (128, 0, 128, 30),
    'Spin': (255, 0, 0, 30),
    'Bounce': (255, 255, 0, 30),    # Yellow with alpha
    'Unknown': (128, 128, 128, 30), # Gray with alpha
}

limitpower = 0.95
satpower = 0
rawpower = 0

Backward_motion = True
Forward_motion = False
Forward_full_speed = False

# Ossilating movement parameters
T1 = 0.4  # Period for the first and third segments, in seconds, limit: both period on 0.03 with 0.4 amplitude
T2 = 0.8  # Period for the second segment, in seconds

A2 = 0.9         # Amplitude of second segment   choose value between 0 and 0.5, Limit 0.8 with period 0.1
A1 = -0.5*A2         # Amplitude of first segment, (needs to be in relation with the second one)


cycle_duration= T1 + T2 + T1 # cycle duration, in seconds, of back-and-forth flapping the motor to generate some yaw. Sum of all the different flapping frequencies

offset1 = -A1   # Offset to shift the range
offset2 = 0   # Offset to shift the range
offset3 = A1  # Offset to shift the range

# vibrating
T1v = 0.1  # Period for the first and third segments, in seconds, limit: both period on 0.03 with 0.4 amplitude
T2v = 0.1  # Period for the second segment, in seconds

A2v = 0.7         # Amplitude of second segment   choose value between 0 and 0.5, Limit 0.8 with period 0.1
A1v = -0.5*A2v         # Amplitude of first segment, (needs to be in relation with the second one)

cycle_duration_vibration= T1v + T2v + T1v # cycle duration, in seconds, of back-and-forth flapping the motor to generate some yaw. Sum of all the different flapping frequencies

offset1v = -A1v   # Offset to shift the range
offset2v = 0   # Offset to shift the range
offset3v = A1v  # Offset to shift the range


# zero stand
ref_roll = 0
K_P = 0.5           #gain


## for Virtual Model Control (VMC) ##
# set constant parameters:
g = 9.81  # acceleration of gravity in m/s^2
    
# geometrical parameters of the ball (needed for virtual model control):
r = 0.025  # distance from axle to center of mass in m, should be equal to distance from sphere center to axle.
Delta = 0.01366 # distance from axle center to block center of mass in x-direction
R = .1    # radius of ball in m

# virtual model control parameters:
k_virtual = 30 # 30 # rigid version   # virtual spring constant, in percent per m, not the same as in simulation where it is in N/m
dist_error = 1      # m, just a carrot always 1 m ahead
dir_ref = 0         # rad, reference direction (track or heading) in absolute coordinates. Note the hub re-initializes yaw each time to zero!
# So this direction is not really absolute. Alternative setting could e.g. be dir_ref= -pi/2
