"""
LiveClassifyDashboard.py

PyQt6 UI wrapper around CNN_movement_pytorch.py logic.
Supports two modes: ground (ball on floor) and hand (ball held).
"""

import sys
import os
import re
import time
import types
import threading
from collections import deque

import numpy as np
import torch
import torch.nn.functional as F
from PyQt6 import QtWidgets, QtCore, QtGui

# The 'states' package lives in the 'To_fizzy_git' root, three folders above
# this 'Model Running' folder ('Model Running' -> 'CNN_final' -> 'BEP code' ->
# 'To_fizzy_git'), so add that folder to sys.path.
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', '..'))
from states.random import Random as RandomState

from Utilities.Fizzy_CNNudp import Fizzy
from Utilities.imu_CNN_data_extractor import (
    extract_linear_acceleration_from_packet,
    extract_angular_velocity_from_packet,
)
from Utilities.CNN_fizzy_ground import IMUNet
from Utilities.CNN_fizzy_hand import IMUNet as IMUNetHand

from settingsHandGroundModel import CONFIDENCE_THRESHOLD, MODEL_NUMBER_GROUND, MODEL_NUMBER_HAND, DATA_VERSION_GROUND, DATA_VERSION_HAND, CLASS_NAMES_HAND
from settingsHandGroundModel import INPUT_AMOUNT_GROUND, CLASS_NAMES_GROUND
from settingsHandGroundModel import CONSECUTIVE_COUNT_GROUND, CONSECUTIVE_COUNT_HAND, COOLDOWN_AFTER_CLASS_GROUND, COOLDOWN_AFTER_CLASS_HAND
from settingsHandGroundModel import CONFIDENCE_THRESHOLD_GROUND, CONFIDENCE_THRESHOLD_HAND

# ===== SETTINGS =====
# Fixed windowing parameters. Intentionally hardcoded here (not in
# settingsHandGroundModel.py) so they are not user-tunable.
WINDOW_SIZE = 64                    # Window size for classifying
STRIDE = 32                         # Stride for classifying
# CONFIDENCE_THRESHOLD = 0.8
# MODEL_NUMBER_GROUND  = '017'
# MODEL_NUMBER_HAND    = '019'
# DATA_VERSION_GROUND  = '7'
# DATA_VERSION_HAND    = '8'

# LIFT_COUNT_TO_SWITCH = 2   # consecutive 'lift' detections to switch to hand mode
# DROP_COUNT_TO_SWITCH = 1   # consecutive 'drop' detections to switch to ground mode

# Fixed random-motor behaviour. These are intentionally hardcoded here (not in
# settings.py) so they are not user-tunable.
_random_config = types.SimpleNamespace(
    random_gain=0.9,
    random_chaos=1.5,
    random_smooth=0.85,
    random_deadzone=0.2,
)

# CLASS_NAMES_GROUND = ['idle', 'tap', 'lift', 'kick']
# CLASS_NAMES_HAND   = ['idle', 'tap', 'shake', 'drop']

CLASS_COLORS = {
    'idle':  '#607D8B',
    'tap':   '#2196F3',
    'lift':  '#4CAF50',
    'kick':  '#F44336',
    'shake': '#9C27B0',
    'drop':  '#FF5722',
}

FILE_PATH         = os.path.dirname(os.path.abspath(__file__))
# The 'Models' and 'Mean, std' folders stayed in the CNN_final root, which is one
# level up from this 'Model Running' folder. settingsHandGroundModel.py moved here
# alongside this script, so it still lives next to it (FILE_PATH).
ROOT_PATH         = os.path.dirname(FILE_PATH)
MODEL_PATH_GROUND = os.path.join(ROOT_PATH, 'Models', f'Trained_{MODEL_NUMBER_GROUND}.pth')
MODEL_PATH_HAND   = os.path.join(ROOT_PATH, 'Models', f'Trained_{MODEL_NUMBER_HAND}.pth')
SETTINGS_PATH     = os.path.join(FILE_PATH, 'settingsHandGroundModel.py')


# ===== SETTINGS PERSISTENCE =====

# The dicts below are the same objects imported from settingsHandGroundModel.py.
# Editing them in place (from the UI) takes effect live; save_settings_to_file()
# writes the current values back into settingsHandGroundModel.py, rewriting only the
# values and keeping all comments.
_MANAGED_DICTS = [
    ('CONFIDENCE_THRESHOLD_GROUND', CONFIDENCE_THRESHOLD_GROUND, lambda v: f"{float(v):.2f}"),
    ('CONFIDENCE_THRESHOLD_HAND',   CONFIDENCE_THRESHOLD_HAND,   lambda v: f"{float(v):.2f}"),
    ('CONSECUTIVE_COUNT_GROUND',    CONSECUTIVE_COUNT_GROUND,    lambda v: f"{int(v)}"),
    ('CONSECUTIVE_COUNT_HAND',      CONSECUTIVE_COUNT_HAND,      lambda v: f"{int(v)}"),
    ('COOLDOWN_AFTER_CLASS_GROUND', COOLDOWN_AFTER_CLASS_GROUND, lambda v: f"{float(v):.1f}"),
    ('COOLDOWN_AFTER_CLASS_HAND',   COOLDOWN_AFTER_CLASS_HAND,   lambda v: f"{float(v):.1f}"),
]

# Dict entries whose value in settings.py is a named constant rather than a
# literal. These are NOT inlined on save; instead the named scalar constant is
# updated, so 'lift'/'drop' keep doubling as the mode-switch thresholds.
_LINKED_TO_SCALAR = {
    ('CONSECUTIVE_COUNT_GROUND', 'lift'): 'LIFT_COUNT_TO_SWITCH',
    ('CONSECUTIVE_COUNT_HAND',   'drop'): 'DROP_COUNT_TO_SWITCH',
}


def save_settings_to_file():
    """Write the current in-memory dict values back into settingsHandGroundModel.py.

    Only the numeric value of each existing key is replaced; comments, ordering
    and any unmanaged content are left untouched. Keys listed in
    _LINKED_TO_SCALAR (the 'lift'/'drop' switch counts) keep their named-constant
    reference in the dict; the constant's own assignment is updated instead."""
    with open(SETTINGS_PATH, 'r', encoding='utf-8') as f:
        text = f.read()

    for name, d, fmt in _MANAGED_DICTS:
        block_re = re.compile(r'(' + re.escape(name) + r'\s*=\s*\{)(.*?)(\n\})', re.DOTALL)
        m = block_re.search(text)
        if not m:
            continue
        body = m.group(2)
        for key, val in d.items():
            if (name, key) in _LINKED_TO_SCALAR:
                continue  # leave the named-constant reference intact
            key_re = re.compile(r"(['\"]" + re.escape(key) + r"['\"]\s*:\s*)([^,\n#]+)")
            body = key_re.sub(lambda mm, v=val, f=fmt: mm.group(1) + f(v), body, count=1)
        text = text[:m.start(2)] + body + text[m.end(2):]

    # Update the named scalar constants that the linked dict keys point at.
    scalar_updates = {
        'LIFT_COUNT_TO_SWITCH': int(CONSECUTIVE_COUNT_GROUND.get('lift', 1)),
        'DROP_COUNT_TO_SWITCH': int(CONSECUTIVE_COUNT_HAND.get('drop', 1)),
    }
    for sname, sval in scalar_updates.items():
        scalar_re = re.compile(r'^(' + re.escape(sname) + r'\s*=\s*)(\d+)', re.M)
        text = scalar_re.sub(lambda mm, v=sval: mm.group(1) + str(v), text, count=1)

    with open(SETTINGS_PATH, 'w', encoding='utf-8') as f:
        f.write(text)


# ===== HELPERS =====

if INPUT_AMOUNT_GROUND == 7:
    def packet_to_features_ground(packet, motor_input=0.0):
        ax, ay, az = extract_linear_acceleration_from_packet(packet)
        wx, wy, wz = extract_angular_velocity_from_packet(packet)
        return [ax, ay, az, wx, wy, wz, motor_input]
elif INPUT_AMOUNT_GROUND == 9:
    def packet_to_features_ground(packet, motor_input=0.0):
        ax, ay, az = extract_linear_acceleration_from_packet(packet)
        wx, wy, wz = extract_angular_velocity_from_packet(packet)
        acc_mag = np.sqrt((ax**2) + (ay**2) + (az**2))
        gyro_mag = np.sqrt((wx**2) + (wy**2) + (wz**2))
        return [ax, ay, az, wx, wy, wz, motor_input, acc_mag, gyro_mag]


def packet_to_features_hand(packet):
    ax, ay, az = extract_linear_acceleration_from_packet(packet)
    wx, wy, wz = extract_angular_velocity_from_packet(packet)
    return [ax, ay, az, wx, wy, wz]

# ===== WORKER THREAD =====

class WorkerThread(threading.Thread):
    def __init__(self, fizzy, net_ground, mean_ground, std_ground, net_hand, mean_hand, std_hand):
        super().__init__(daemon=True)
        self.fizzy        = fizzy
        self._net_ground  = net_ground
        self._mean_ground = mean_ground
        self._std_ground  = std_ground
        self._net_hand    = net_hand
        self._mean_hand   = mean_hand
        self._std_hand    = std_hand

        self._lock    = threading.Lock()
        self._mode    = 'random'   # motor mode: 'stop' or 'random'
        self._running = True

        # Ball state
        self.ball_mode           = 'ground'   # 'ground' or 'hand'
        self.current_class_names = list(CLASS_NAMES_GROUND)

        # Per-class confirmation tracking (consecutive confident detections)
        self._streak_label = None
        self._streak_count = 0

        # Exposed state for UI
        self.last_label           = None
        self.last_raw_label       = None
        self.last_conf            = 0.0
        self.last_probs           = [0.0] * 4
        self.motor_power          = 0.0
        self.is_connected         = False
        # Confirmed class (met its consecutive-count requirement) + live streak
        self.last_confirmed_label = None
        self.streak_label         = None
        self.streak_count         = 0

        # Random motor state
        self._random_state = RandomState(_random_config, duration=None)
        self._random_state.enter()

        self._pending_drop_cooldown = False
        self._pending_lift_cooldown = False

    @property
    def motor_mode(self):
        with self._lock:
            return self._mode

    def set_mode(self, mode: str):
        with self._lock:
            self._mode = mode

    def set_ball_mode(self, mode: str):
        self.ball_mode = mode
        self.current_class_names = list(CLASS_NAMES_HAND if mode == 'hand' else CLASS_NAMES_GROUND)
        self._streak_label = None
        self._streak_count = 0
        if mode == 'hand':
            self._pending_lift_cooldown = True
            self._pending_drop_cooldown = False
        else:
            self._pending_drop_cooldown = True
            self._pending_lift_cooldown = False

    def stop(self):
        self._running = False

    def _classify(self, window_np, ball_mode):
        if ball_mode == 'ground':
            net, mean, std = self._net_ground, self._mean_ground, self._std_ground
            class_names = CLASS_NAMES_GROUND
            thresholds  = CONFIDENCE_THRESHOLD_GROUND
        else:
            net, mean, std = self._net_hand, self._mean_hand, self._std_hand
            class_names = CLASS_NAMES_HAND
            thresholds  = CONFIDENCE_THRESHOLD_HAND

        x = window_np.T.astype(np.float32)
        n = mean.shape[0]
        x = x[:n]
        x = (x - mean) / (std + 1e-8)
        x = torch.from_numpy(x).unsqueeze(0)
        with torch.no_grad():
            probs = F.softmax(net(x), dim=1)[0]
        conf, idx = torch.max(probs, dim=0)
        raw_label = class_names[idx.item()]
        threshold = thresholds.get(raw_label, CONFIDENCE_THRESHOLD)
        label = raw_label if conf.item() >= threshold else None
        return label, raw_label, conf.item(), probs.tolist()

    def run(self):
        buffer                 = deque(maxlen=WINDOW_SIZE)
        samples_since_pred     = 0
        cooldown_until         = 0.0
        active_ball_mode       = 'ground'   # local copy to detect switches
        _prev_time             = time.time()
        _prev_mode             = 'random'
        _prev_active_ball_mode = 'ground'

        while self._running:
            with self._lock:
                mode = self._mode

            # Clear buffer and reset state when ball mode changes
            if self.ball_mode != active_ball_mode:
                active_ball_mode    = self.ball_mode
                buffer.clear()
                samples_since_pred  = 0
                if self._pending_drop_cooldown:
                    cooldown_until              = time.time() + COOLDOWN_AFTER_CLASS_HAND.get('drop', 0.0)
                    self._pending_drop_cooldown = False
                elif self._pending_lift_cooldown:
                    cooldown_until              = time.time() + COOLDOWN_AFTER_CLASS_GROUND.get('lift', 0.0)
                    self._pending_lift_cooldown = False
                else:
                    cooldown_until = 0.0
                self.last_label     = None
                self.last_raw_label = None
                self.last_conf      = 0.0
                self.last_probs     = [0.0] * 4
                self._streak_label        = None
                self._streak_count        = 0
                self.streak_label         = None
                self.streak_count         = 0
                self.last_confirmed_label = None

            now = time.time()
            dt  = max(now - _prev_time, 1e-6)
            _prev_time = now

            # Re-enter random state on mode or ball-mode transitions
            entering_random  = (mode == 'random' and _prev_mode != 'random')
            hand_to_ground   = (active_ball_mode == 'ground' and _prev_active_ball_mode == 'hand')
            if entering_random or hand_to_ground:
                self._random_state.enter()
            _prev_mode             = mode
            _prev_active_ball_mode = active_ball_mode

            # Motor control — forced 0 in hand mode
            if active_ball_mode == 'hand':
                motor = 0.0
            elif mode == 'random':
                motor = self._random_state.update(dt, None, None)
            else:
                motor = 0.0

            self.motor_power = motor
            self.fizzy.set_motor(motor)

            data = self.fizzy.get_data()
            if data == -1 or not isinstance(data, (list, tuple)) or len(data) < 16:
                self.is_connected = False
                time.sleep(0.001)
                continue

            self.is_connected = True

            if active_ball_mode == 'ground':
                buffer.append(packet_to_features_ground(data, motor))
            else:
                buffer.append(packet_to_features_hand(data))

            samples_since_pred += 1

            if len(buffer) < WINDOW_SIZE or samples_since_pred < STRIDE:
                continue

            samples_since_pred = 0

            if now < cooldown_until:
                continue

            label, raw_label, conf, probs = self._classify(np.array(buffer), active_ball_mode)

            self.last_label     = label
            self.last_raw_label = raw_label
            self.last_conf      = conf
            self.last_probs     = probs

            # Per-class requirement settings for the active model
            if active_ball_mode == 'ground':
                required_counts = CONSECUTIVE_COUNT_GROUND
                cooldown_map    = COOLDOWN_AFTER_CLASS_GROUND
            else:
                required_counts = CONSECUTIVE_COUNT_HAND
                cooldown_map    = COOLDOWN_AFTER_CLASS_HAND

            # Update the consecutive-confident-detection streak and decide whether
            # the current class is "confirmed" (met its required count in a row).
            confirmed_label = None
            if label is not None:
                if label == self._streak_label:
                    self._streak_count += 1
                else:
                    self._streak_label = label
                    self._streak_count = 1
                required = required_counts.get(label, 1)
                if self._streak_count >= required:
                    confirmed_label = label
                    self._streak_count = 0
            else:
                self._streak_label = None
                self._streak_count = 0

            # Expose streak / confirmation to the UI
            self.streak_label = self._streak_label
            self.streak_count = self._streak_count
            if confirmed_label is not None:
                self.last_confirmed_label = confirmed_label

            class_names = CLASS_NAMES_GROUND if active_ball_mode == 'ground' else CLASS_NAMES_HAND
            prob_str = '  '.join(f"{n}:{p*100:.1f}%" for n, p in zip(class_names, probs))
            accepted = "OK" if label is not None else "low-conf"
            confirm_str = f"  CONFIRMED={confirmed_label.upper()}" if confirmed_label else ""
            print(f"[{active_ball_mode.upper()}] {raw_label.upper():6s} {conf*100:5.1f}%  ({accepted})  |  {prob_str}{confirm_str}")

            # Ball mode switching is driven by a confirmed class
            if active_ball_mode == 'ground':
                if confirmed_label == 'lift':
                    print(f"  [SWITCH] -> hand mode")
                    self.ball_mode              = 'hand'
                    self.current_class_names    = list(CLASS_NAMES_HAND)
                    self._pending_lift_cooldown = True
            else:  # hand mode
                if confirmed_label == 'drop':
                    print(f"  [SWITCH] -> ground mode")
                    self.ball_mode              = 'ground'
                    self.current_class_names    = list(CLASS_NAMES_GROUND)
                    self._pending_drop_cooldown = True
                    self.set_mode('random')

            # Per-class cooldown: wait before classifying again after a confirmation
            if confirmed_label is not None:
                cooldown_until = now + cooldown_map.get(confirmed_label, 0.0)


# ===== UI =====

class ClassificationWidget(QtWidgets.QGroupBox):
    def __init__(self, parent=None):
        super().__init__("Gesture Classification", parent)
        layout = QtWidgets.QVBoxLayout(self)
        layout.setSpacing(8)

        self.gesture_label = QtWidgets.QLabel("---")
        gf = QtGui.QFont()
        gf.setPointSize(14)
        gf.setBold(True)
        self.gesture_label.setFont(gf)
        self.gesture_label.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.gesture_label.setMinimumHeight(30)
        layout.addWidget(self.gesture_label)

        self.conf_label = QtWidgets.QLabel("---")
        cf = QtGui.QFont()
        cf.setPointSize(10)
        self.conf_label.setFont(cf)
        self.conf_label.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self.conf_label)

        grid = QtWidgets.QGridLayout()
        grid.setHorizontalSpacing(8)
        grid.setVerticalSpacing(4)
        self._bar_labels: list[QtWidgets.QLabel] = []
        self._bars: list[QtWidgets.QProgressBar] = []

        _all_names = max(CLASS_NAMES_GROUND, CLASS_NAMES_HAND, key=len)
        for i, name in enumerate(_all_names):
            lbl = QtWidgets.QLabel(f"{name}:")
            lbl.setAlignment(
                QtCore.Qt.AlignmentFlag.AlignRight | QtCore.Qt.AlignmentFlag.AlignVCenter
            )
            lbl.setMinimumWidth(48)
            bar = QtWidgets.QProgressBar()
            bar.setRange(0, 100)
            bar.setValue(0)
            bar.setFormat("%p%")
            color = CLASS_COLORS.get(name, '#2196F3')
            bar.setStyleSheet(f"QProgressBar::chunk {{ background-color: {color}; }}")
            self._bar_labels.append(lbl)
            self._bars.append(bar)
            grid.addWidget(lbl, i, 0)
            grid.addWidget(bar, i, 1)
        layout.addLayout(grid)

        self._current_names: list[str] = []
        self.update_class_names(CLASS_NAMES_GROUND)

    def update_class_names(self, names):
        if names == self._current_names:
            return
        for i, name in enumerate(names):
            self._bar_labels[i].setText(f"{name}:")
            color = CLASS_COLORS.get(name, '#2196F3')
            self._bars[i].setStyleSheet(f"QProgressBar::chunk {{ background-color: {color}; }}")
            self._bar_labels[i].setVisible(True)
            self._bars[i].setVisible(True)
        for i in range(len(names), len(self._bar_labels)):
            self._bar_labels[i].setVisible(False)
            self._bars[i].setVisible(False)
        self._current_names = list(names)

    def refresh(self, label, raw_label, conf, probs):
        text  = raw_label.upper() if raw_label else "---"
        color = '#4CAF50' if label is not None else '#FF9800'
        self.gesture_label.setText(text)
        self.gesture_label.setStyleSheet(f"color: {color};")
        self.conf_label.setText(f"{conf * 100:.1f}%" if raw_label else "---")
        for i in range(len(self._current_names)):
            pct = int(probs[i] * 100) if i < len(probs) else 0
            self._bars[i].setValue(pct)


class RequirementsWidget(QtWidgets.QGroupBox):
    """Shows the confirmed class and, per class, how many confident detections in
    a row are required plus the cooldown before classifying resumes. The settings
    differ per model, so the table rebuilds when the ball mode changes."""

    def __init__(self, parent=None):
        super().__init__("Confirmed Class & Requirements", parent)
        layout = QtWidgets.QVBoxLayout(self)
        layout.setSpacing(6)

        self.confirmed_label = QtWidgets.QLabel("Confirmed: ---")
        cf = QtGui.QFont()
        cf.setPointSize(36)
        cf.setBold(True)
        self.confirmed_label.setFont(cf)
        self.confirmed_label.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.confirmed_label.setMinimumHeight(80)
        layout.addWidget(self.confirmed_label)

        self._grid = QtWidgets.QGridLayout()
        self._grid.setHorizontalSpacing(12)
        self._grid.setVerticalSpacing(5)
        for col, title in enumerate(("Class", "In a row", "Cooldown")):
            h = QtWidgets.QLabel(title)
            h.setStyleSheet("font-weight: bold; font-size: 13px; color: #888;")
            self._grid.addWidget(h, 0, col)
        layout.addLayout(self._grid)

        self._rows: dict[str, tuple] = {}   # name -> (name_lbl, streak_lbl, cd_lbl)
        self._current_mode = None

    def _build_rows(self, names, required_counts, cooldown_map):
        for widgets in self._rows.values():
            for w in widgets:
                self._grid.removeWidget(w)
                w.deleteLater()
        self._rows = {}
        for i, name in enumerate(names, start=1):
            color    = CLASS_COLORS.get(name, '#2196F3')
            name_lbl = QtWidgets.QLabel(name)
            name_lbl.setStyleSheet(f"font-weight: bold; color: {color};")
            streak_lbl = QtWidgets.QLabel(f"0/{required_counts.get(name, 1)}")
            cd_lbl     = QtWidgets.QLabel(f"{cooldown_map.get(name, 0.0):.1f}s")
            self._grid.addWidget(name_lbl,   i, 0)
            self._grid.addWidget(streak_lbl, i, 1)
            self._grid.addWidget(cd_lbl,     i, 2)
            self._rows[name] = (name_lbl, streak_lbl, cd_lbl)

    def refresh(self, ball_mode, confirmed_label, streak_label, streak_count):
        if ball_mode == 'ground':
            names           = CLASS_NAMES_GROUND
            required_counts = CONSECUTIVE_COUNT_GROUND
            cooldown_map    = COOLDOWN_AFTER_CLASS_GROUND
        else:
            names           = CLASS_NAMES_HAND
            required_counts = CONSECUTIVE_COUNT_HAND
            cooldown_map    = COOLDOWN_AFTER_CLASS_HAND

        if ball_mode != self._current_mode:
            self._build_rows(names, required_counts, cooldown_map)
            self._current_mode = ball_mode

        if confirmed_label:
            color = CLASS_COLORS.get(confirmed_label, '#4CAF50')
            self.confirmed_label.setText(f"Confirmed: {confirmed_label.upper()}")
            self.confirmed_label.setStyleSheet(f"color: {color};")
        else:
            self.confirmed_label.setText("Confirmed: ---")
            self.confirmed_label.setStyleSheet("")

        for name, (_name_lbl, streak_lbl, _cd_lbl) in self._rows.items():
            required = required_counts.get(name, 1)
            count    = streak_count if name == streak_label else 0
            streak_lbl.setText(f"{count}/{required}")
            streak_lbl.setStyleSheet("font-weight: bold; color: #4CAF50;" if count > 0 else "")


class TuningWidget(QtWidgets.QGroupBox):
    """Editable per-class tuning for the active ball mode: confidence threshold,
    number of confident detections required in a row, and the cooldown after a
    confirmation. Edits mutate the shared settings dicts in place so they take
    effect immediately; 'Save to settings.py' persists them to disk."""

    def __init__(self, parent=None):
        super().__init__("Tuning (active mode)", parent)
        layout = QtWidgets.QVBoxLayout(self)
        layout.setSpacing(6)

        self._grid = QtWidgets.QGridLayout()
        self._grid.setHorizontalSpacing(10)
        self._grid.setVerticalSpacing(3)
        for col, title in enumerate(("Class", "Conf ≥", "In a row", "Cooldown s")):
            h = QtWidgets.QLabel(title)
            h.setStyleSheet("font-weight: bold; font-size: 11px; color: #888;")
            self._grid.addWidget(h, 0, col)
        layout.addLayout(self._grid)

        save_row = QtWidgets.QHBoxLayout()
        self.status_lbl = QtWidgets.QLabel("")
        self.status_lbl.setStyleSheet("font-size: 11px;")
        self.btn_save = QtWidgets.QPushButton("Save to settings")
        self.btn_save.setStyleSheet("QPushButton { padding: 6px 16px; font-size: 12px; }")
        save_row.addWidget(self.status_lbl)
        save_row.addStretch()
        save_row.addWidget(self.btn_save)
        layout.addLayout(save_row)
        self.btn_save.clicked.connect(self._on_save)

        self._rows: dict[str, tuple] = {}
        self._current_mode = None

    def _build_rows(self, mode):
        for widgets in self._rows.values():
            for w in widgets:
                self._grid.removeWidget(w)
                w.deleteLater()
        self._rows = {}

        if mode == 'ground':
            names    = CLASS_NAMES_GROUND
            thr_map  = CONFIDENCE_THRESHOLD_GROUND
            cnt_map  = CONSECUTIVE_COUNT_GROUND
            cd_map   = COOLDOWN_AFTER_CLASS_GROUND
        else:
            names    = CLASS_NAMES_HAND
            thr_map  = CONFIDENCE_THRESHOLD_HAND
            cnt_map  = CONSECUTIVE_COUNT_HAND
            cd_map   = COOLDOWN_AFTER_CLASS_HAND

        for i, name in enumerate(names, start=1):
            color    = CLASS_COLORS.get(name, '#2196F3')
            name_lbl = QtWidgets.QLabel(name)
            name_lbl.setStyleSheet(f"font-weight: bold; color: {color};")

            thr_spin = QtWidgets.QDoubleSpinBox()
            thr_spin.setRange(0.0, 1.0)
            thr_spin.setSingleStep(0.05)
            thr_spin.setDecimals(2)
            thr_spin.setValue(float(thr_map.get(name, CONFIDENCE_THRESHOLD)))
            thr_spin.valueChanged.connect(
                lambda v, d=thr_map, k=name: d.__setitem__(k, float(v))
            )

            cnt_spin = QtWidgets.QSpinBox()
            cnt_spin.setRange(1, 99)
            cnt_spin.setValue(int(cnt_map.get(name, 1)))
            cnt_spin.valueChanged.connect(
                lambda v, d=cnt_map, k=name: d.__setitem__(k, int(v))
            )

            cd_spin = QtWidgets.QDoubleSpinBox()
            cd_spin.setRange(0.0, 60.0)
            cd_spin.setSingleStep(0.5)
            cd_spin.setDecimals(1)
            cd_spin.setValue(float(cd_map.get(name, 0.0)))
            cd_spin.valueChanged.connect(
                lambda v, d=cd_map, k=name: d.__setitem__(k, float(v))
            )

            self._grid.addWidget(name_lbl, i, 0)
            self._grid.addWidget(thr_spin, i, 1)
            self._grid.addWidget(cnt_spin, i, 2)
            self._grid.addWidget(cd_spin,  i, 3)
            self._rows[name] = (name_lbl, thr_spin, cnt_spin, cd_spin)

    def refresh(self, ball_mode):
        if ball_mode != self._current_mode:
            self._build_rows(ball_mode)
            self._current_mode = ball_mode

    def _on_save(self):
        try:
            save_settings_to_file()
            self.status_lbl.setText("Saved ✓")
            self.status_lbl.setStyleSheet("font-size: 11px; color: #4CAF50;")
        except Exception as e:
            self.status_lbl.setText(f"Save failed: {e}")
            self.status_lbl.setStyleSheet("font-size: 11px; color: #F44336;")


class MotorPanel(QtWidgets.QGroupBox):
    mode_changed = QtCore.pyqtSignal(str)

    def __init__(self, parent=None):
        super().__init__("Motor Control", parent)
        layout = QtWidgets.QHBoxLayout(self)
        layout.setSpacing(8)

        self.btn_random = QtWidgets.QPushButton("Random (P)")
        self.btn_stop   = QtWidgets.QPushButton("Stop (S)")

        btn_style_active = (
            "QPushButton { padding: 8px 20px; font-size: 13px; }"
            "QPushButton:checked { background-color: #4CAF50; color: white; font-weight: bold; }"
        )
        for btn in (self.btn_random, self.btn_stop):
            btn.setCheckable(True)
            btn.setStyleSheet(btn_style_active)
            layout.addWidget(btn)

        self.btn_random.setChecked(True)

        self.motor_label = QtWidgets.QLabel("Motor: 0.000")
        self.motor_label.setStyleSheet("font-size: 12px; font-weight: bold;")
        layout.addStretch()
        layout.addWidget(self.motor_label)

        self.btn_random.clicked.connect(lambda: self._select('random'))
        self.btn_stop.clicked.connect(lambda: self._select('stop'))

    def _select(self, mode: str):
        self.btn_random.setChecked(mode == 'random')
        self.btn_stop.setChecked(mode == 'stop')
        self.mode_changed.emit(mode)

    def sync_mode(self, mode: str):
        self.btn_random.setChecked(mode == 'random')
        self.btn_stop.setChecked(mode == 'stop')

    def refresh(self, motor_power: float):
        self.motor_label.setText(f"Motor: {motor_power:+.3f}")


class BallModePanel(QtWidgets.QGroupBox):
    ball_mode_changed = QtCore.pyqtSignal(str)

    def __init__(self, parent=None):
        super().__init__("Ball Mode Override", parent)
        layout = QtWidgets.QHBoxLayout(self)
        layout.setSpacing(8)

        self.btn_ground = QtWidgets.QPushButton("Ground (G)")
        self.btn_hand   = QtWidgets.QPushButton("Hand (H)")

        btn_style = (
            "QPushButton { padding: 8px 20px; font-size: 13px; }"
            "QPushButton:checked { background-color: #4CAF50; color: white; font-weight: bold; }"
        )
        for btn in (self.btn_ground, self.btn_hand):
            btn.setCheckable(True)
            btn.setStyleSheet(btn_style)
            layout.addWidget(btn)

        self.btn_ground.setChecked(True)

        self.btn_ground.clicked.connect(lambda: self._select('ground'))
        self.btn_hand.clicked.connect(lambda: self._select('hand'))

    def _select(self, mode: str):
        self.btn_ground.setChecked(mode == 'ground')
        self.btn_hand.setChecked(mode == 'hand')
        self.ball_mode_changed.emit(mode)

    def sync_mode(self, mode: str):
        self.btn_ground.setChecked(mode == 'ground')
        self.btn_hand.setChecked(mode == 'hand')


class LiveClassifyDashboard(QtWidgets.QMainWindow):
    def __init__(self, worker: WorkerThread):
        super().__init__()
        self.worker = worker
        self.setWindowTitle(
            f"Live Classify Dashboard  |  Ground: {MODEL_NUMBER_GROUND}  |  Hand: {MODEL_NUMBER_HAND}"
        )
        self.resize(500, 980)

        cw = QtWidgets.QWidget()
        self.setCentralWidget(cw)
        root = QtWidgets.QVBoxLayout(cw)
        root.setSpacing(10)
        root.setContentsMargins(12, 12, 12, 12)

        # Status row
        status_row = QtWidgets.QHBoxLayout()
        self.conn_label = QtWidgets.QLabel("Fizzy: --")
        self.conn_label.setStyleSheet("font-weight: bold; font-size: 12px;")
        self.mode_label = QtWidgets.QLabel("Ball: GROUND")
        self.mode_label.setStyleSheet("font-weight: bold; font-size: 12px;")
        status_row.addWidget(self.conn_label)
        status_row.addStretch()
        status_row.addWidget(self.mode_label)
        root.addLayout(status_row)

        self.classify_widget     = ClassificationWidget()
        self.requirements_widget  = RequirementsWidget()
        self.tuning_widget        = TuningWidget()
        self.motor_panel          = MotorPanel()
        self.ball_mode_panel      = BallModePanel()

        root.addWidget(self.requirements_widget, stretch=2)
        root.addWidget(self.classify_widget)
        root.addWidget(self.tuning_widget, stretch=1)
        root.addWidget(self.ball_mode_panel)
        root.addWidget(self.motor_panel)

        self.motor_panel.mode_changed.connect(self.worker.set_mode)
        self.ball_mode_panel.ball_mode_changed.connect(self.worker.set_ball_mode)

        self._timer = QtCore.QTimer()
        self._timer.timeout.connect(self._refresh)
        self._timer.start(50)  # 20 Hz UI refresh

        QtGui.QShortcut(QtGui.QKeySequence('P'), self).activated.connect(
            lambda: self.motor_panel._select('random')
        )
        QtGui.QShortcut(QtGui.QKeySequence('S'), self).activated.connect(
            lambda: self.motor_panel._select('stop')
        )
        QtGui.QShortcut(QtGui.QKeySequence('G'), self).activated.connect(
            lambda: self.ball_mode_panel._select('ground')
        )
        QtGui.QShortcut(QtGui.QKeySequence('H'), self).activated.connect(
            lambda: self.ball_mode_panel._select('hand')
        )

    def _refresh(self):
        w = self.worker
        connected = w.is_connected
        self.conn_label.setText("Fizzy: Connected" if connected else "Fizzy: Disconnected")
        self.conn_label.setStyleSheet(
            f"font-weight: bold; font-size: 12px; color: {'#4CAF50' if connected else '#F44336'};"
        )

        ball_mode = w.ball_mode
        self.mode_label.setText(f"Ball: {ball_mode.upper()}")
        self.mode_label.setStyleSheet(
            f"font-weight: bold; font-size: 12px; "
            f"color: {'#4CAF50' if ball_mode == 'ground' else '#2196F3'};"
        )

        self.classify_widget.update_class_names(w.current_class_names)
        self.classify_widget.refresh(w.last_label, w.last_raw_label, w.last_conf, w.last_probs)
        self.requirements_widget.refresh(
            ball_mode, w.last_confirmed_label, w.streak_label, w.streak_count
        )
        self.tuning_widget.refresh(ball_mode)
        self.ball_mode_panel.sync_mode(w.ball_mode)
        self.motor_panel.sync_mode(w.motor_mode)
        self.motor_panel.refresh(w.motor_power)

    def closeEvent(self, event):
        self._timer.stop()
        self.worker.stop()
        super().closeEvent(event)


# ===== ENTRY POINT =====

def main():
    print(f"Loading ground model: {MODEL_PATH_GROUND}")
    net_ground = IMUNet()
    net_ground.load_state_dict(torch.load(MODEL_PATH_GROUND, map_location='cpu'))
    net_ground.eval()
    mean_ground = np.load(os.path.join(ROOT_PATH, "Mean, std", f"imu_mean_v{DATA_VERSION_GROUND}.npy")).astype(np.float32)
    std_ground  = np.load(os.path.join(ROOT_PATH, "Mean, std", f"imu_std_v{DATA_VERSION_GROUND}.npy")).astype(np.float32)
    if mean_ground.ndim == 3:
        mean_ground = mean_ground.squeeze(0)
        std_ground  = std_ground.squeeze(0)
    print(f"Ground model loaded. Classes: {CLASS_NAMES_GROUND}  |  norm shape: {mean_ground.shape}")

    print(f"Loading hand model: {MODEL_PATH_HAND}")
    net_hand = IMUNetHand()
    net_hand.load_state_dict(torch.load(MODEL_PATH_HAND, map_location='cpu'))
    net_hand.eval()
    mean_hand = np.load(os.path.join(ROOT_PATH, "Mean, std", f"imu_mean_v{DATA_VERSION_HAND}.npy")).astype(np.float32)
    std_hand  = np.load(os.path.join(ROOT_PATH, "Mean, std", f"imu_std_v{DATA_VERSION_HAND}.npy")).astype(np.float32)
    if mean_hand.ndim == 3:
        mean_hand = mean_hand.squeeze(0)
        std_hand  = std_hand.squeeze(0)
    print(f"Hand model loaded.   Classes: {CLASS_NAMES_HAND}  |  norm shape: {mean_hand.shape}")

    fizzy  = Fizzy()
    worker = WorkerThread(fizzy, net_ground, mean_ground, std_ground, net_hand, mean_hand, std_hand)
    worker.start()
    print("Worker started. P = random motor  |  S = stop  |  Close window to quit")

    app = QtWidgets.QApplication(sys.argv)
    win = LiveClassifyDashboard(worker)
    win.show()
    code = app.exec()

    worker.stop()
    worker.join(timeout=2.0)
    try:
        fizzy.stop()
    except Exception:
        pass
    sys.exit(code)


if __name__ == '__main__':
    main()
