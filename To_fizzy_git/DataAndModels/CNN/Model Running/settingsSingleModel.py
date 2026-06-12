

# ===== LiveClassifier.py (single-model live UI) =====
# Uses the CNN structure from CNN_fizzy.py together with MODEL_NUMBER,
# data_version, class_names, INPUT_AMOUNT and OUTPUT_AMOUNT can be tuned here
# of this file. Per-class tuning below mirrors the two-model dashboard but for
# the single live model. Classes not listed fall back to CONFIDENCE_THRESHOLD.


MODEL_NUMBER = '039'                # Number of the model
data_version = '21'                 # Recording version for the std and mean


INPUT_AMOUNT = 6
OUTPUT_AMOUNT = 6


#CNN_movement and CNN_pytorch, uses fizzy_ground model
MODEL_NUMBER = '039'                # Number of the model
data_version = '21'                  # Recording version for the std and meanp

# class_names = ['idle', 'tap', 'shake', 'drop']
class_names = ['idle', 'shake', 'tap', 'drop', 'lift', 'kick']


# Default confidence threshold. Classes not listed in CONFIDENCE_THRESHOLD_LIVE
# fall back to this value.
CONFIDENCE_THRESHOLD = 0.70

# Per-class confidence thresholds (override CONFIDENCE_THRESHOLD above).
CONFIDENCE_THRESHOLD_LIVE = {
    'idle':  0.50,
    'shake': 0.90,
    'tap':   0.60,
    'drop':  0.85,
    'lift':  0.90,
    'kick':  0.95,
}
# Consecutive confident detections required before a class is "confirmed".
CONSECUTIVE_COUNT_LIVE = {
    'idle':  1,
    'shake': 3,
    'tap':   1,
    'drop':  1,
    'lift':  1,
    'kick':  1,
}
# Cooldown (seconds) after a class is confirmed before classification resumes.
COOLDOWN_AFTER_CLASS_LIVE = {
    'idle':  0.0,
    'shake': 1.0,
    'tap':   1.0,
    'drop':  1.0,
    'lift':  1.0,
    'kick':  1.0,
}
#================================================
