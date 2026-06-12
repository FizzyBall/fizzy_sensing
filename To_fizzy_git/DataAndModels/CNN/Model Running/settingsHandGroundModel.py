
# ===== LiveClassifyDashboard.py====
#live classify UI
MODEL_NUMBER_GROUND  = '038'
MODEL_NUMBER_HAND    = '037'
DATA_VERSION_GROUND  = '16'
DATA_VERSION_HAND    = '18'

#CNN INPUT OUTPUT AMOUNT GROUND
INPUT_AMOUNT_GROUND = 7
OUTPUT_AMOUNT_GROUND = 3

#CNN INPUT OUTPUT AMOUNT HAND
INPUT_AMOUNT_HAND = 6
OUTPUT_AMOUNT_HAND = 4


CLASS_NAMES_GROUND = ['idle', 'lift', 'kick']
# CLASS_NAMES_GROUND = ['idle', 'shake', 'tap', 'drop', 'lift', 'kick']
CLASS_NAMES_HAND   = ['idle', 'tap', 'shake', 'drop']


#Amount of times drop or lift needs to be detected to change modes.
LIFT_COUNT_TO_SWITCH = 2   # consecutive 'lift' detections to switch to hand mode
DROP_COUNT_TO_SWITCH = 2   # consecutive 'drop' detections to switch to ground mode

# Per-class number of consecutive confident classifications required before a
# class is "confirmed" (i.e. before the dashboard says "okay, this is that class").
# Defined separately for the ground model and the in-hand model.
# 'lift' / 'drop' double as the mode-switch triggers, so they reuse the counts above.
CONSECUTIVE_COUNT_GROUND = {
    'idle': 1,
    'lift': LIFT_COUNT_TO_SWITCH,
    'kick': 1,
}
CONSECUTIVE_COUNT_HAND = {
    'idle': 1,
    'tap':  1,
    'shake': 3,
    'drop': DROP_COUNT_TO_SWITCH,
}

# Per-class cooldown (seconds) to wait after a class is confirmed before
# classification starts again. Defined separately for ground and hand models.
# For 'lift' and 'drop' this cooldown also doubles as the pause after the
# mode switch they trigger (ground → hand / hand → ground).

COOLDOWN_AFTER_CLASS_GROUND = {
    'idle': 0.0,
    'lift': 2.0,   # also the ground → hand switch cooldown
    'kick': 1.0,
}
COOLDOWN_AFTER_CLASS_HAND = {
    'idle': 0.0,
    'tap':  1.0,
    'shake': 1.0,
    'drop': 2.0,   # also the hand → ground switch cooldown
}

#CONFIDENCE THRESHOLD
CONFIDENCE_THRESHOLD = 0.7          # global fallback → below this a class is "unsure"

# Per-class confidence threshold. The winning (argmax) class must reach its own
# threshold before it counts as a confident detection; classes not listed fall
# back to CONFIDENCE_THRESHOLD above. Defined separately for ground and hand.
CONFIDENCE_THRESHOLD_GROUND = {
    'idle': 0.50,
    'lift': 0.90,
    'kick': 0.95,
}
CONFIDENCE_THRESHOLD_HAND = {
    'idle': 0.50,
    'tap':  0.60,
    'shake': 0.90,
    'drop': 0.85,
}
