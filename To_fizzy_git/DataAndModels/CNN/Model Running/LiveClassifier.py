"""
LiveClassifier.py

PyQt6 UI for live gesture classification with a SINGLE CNN model.
Same dashboard layout as LiveClassifyDashboard.py, but without ground/hand
model switching. The CNN structure is imported from CNN_fizzy.py and the model
is selected with MODEL_NUMBER / data_version / class_names in
settingsSingleModel.py.
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
from Utilities.CNN_fizzy import IMUNet

from settingsSingleModel import (
    CONFIDENCE_THRESHOLD,
    MODEL_NUMBER, data_version, class_names, INPUT_AMOUNT,
    CONFIDENCE_THRESHOLD_LIVE, CONSECUTIVE_COUNT_LIVE, COOLDOWN_AFTER_CLASS_LIVE,
)

# ===== SETTINGS =====

CLASS_NAMES = list(class_names)

# Fixed windowing/calibration parameters. These are intentionally hardcoded here
# (not in settingsSingleModel.py) so they are not user-tunable.
WINDOW_SIZE = 64                    # Window size for classifying
STRIDE = 32                         # Stride for classifying
CALIBRATION_SAMPLES = 500           # ~5 seconds at 100 Hz, used to compute mean/std

# Fixed random-motor behaviour. These are intentionally hardcoded here (not in
# settings.py) so they are not user-tunable.
_random_config = types.SimpleNamespace(
    random_gain=0.9,
    random_chaos=1.5,
    random_smooth=0.85,
    random_deadzone=0.2,
)

CLASS_COLORS = {
    'idle':  '#607D8B',
    'tap':   '#2196F3',
    'lift':  '#4CAF50',
    'kick':  '#F44336',
    'shake': '#9C27B0',
    'drop':  '#FF5722',
}

FILE_PATH     = os.path.dirname(os.path.abspath(__file__))
# 'Models' and 'Mean, std' live in the CNN_final root, one level up from here.
ROOT_PATH     = os.path.dirname(FILE_PATH)
MODEL_PATH    = os.path.join(ROOT_PATH, 'Models', f'Trained_{MODEL_NUMBER}.pth')
SETTINGS_PATH = os.path.join(FILE_PATH, 'settingsSingleModel.py')


# ===== SETTINGS PERSISTENCE =====

# The dicts below are the same objects imported from settingsSingleModel.py.
# Editing them in place (from the UI) takes effect live; save_settings_to_file()
# writes the current values back into settingsSingleModel.py, rewriting only the
# values and keeping all comments.
_MANAGED_DICTS = [
    ('CONFIDENCE_THRESHOLD_LIVE',  CONFIDENCE_THRESHOLD_LIVE,  lambda v: f"{float(v):.2f}"),
    ('CONSECUTIVE_COUNT_LIVE',     CONSECUTIVE_COUNT_LIVE,     lambda v: f"{int(v)}"),
    ('COOLDOWN_AFTER_CLASS_LIVE',  COOLDOWN_AFTER_CLASS_LIVE,  lambda v: f"{float(v):.1f}"),
]


def save_settings_to_file():
    """Write the current in-memory dict values back into settingsSingleModel.py.

    Only the numeric value of each existing key is replaced; comments, ordering
    and any unmanaged content are left untouched."""
    with open(SETTINGS_PATH, 'r', encoding='utf-8') as f:
        text = f.read()

    for name, d, fmt in _MANAGED_DICTS:
        block_re = re.compile(r'(' + re.escape(name) + r'\s*=\s*\{)(.*?)(\n\})', re.DOTALL)
        m = block_re.search(text)
        if not m:
            continue
        body = m.group(2)
        for key, val in d.items():
            key_re = re.compile(r"(['\"]" + re.escape(key) + r"['\"]\s*:\s*)([^,\n#]+)")
            body = key_re.sub(lambda mm, v=val, f=fmt: mm.group(1) + f(v), body, count=1)
        text = text[:m.start(2)] + body + text[m.end(2):]

    with open(SETTINGS_PATH, 'w', encoding='utf-8') as f:
        f.write(text)


# ===== HELPERS =====

def packet_to_features(packet, motor_input=0.0):
    ax, ay, az = extract_linear_acceleration_from_packet(packet)
    wx, wy, wz = extract_angular_velocity_from_packet(packet)
    if INPUT_AMOUNT == 7:
        return [ax, ay, az, wx, wy, wz, motor_input]
    elif INPUT_AMOUNT == 9:
        acc_mag  = np.sqrt((ax**2) + (ay**2) + (az**2))
        gyro_mag = np.sqrt((wx**2) + (wy**2) + (wz**2))
        return [ax, ay, az, wx, wy, wz, motor_input, acc_mag, gyro_mag]
    else:  # 6
        return [ax, ay, az, wx, wy, wz]


# ===== WORKER THREAD =====

class WorkerThread(threading.Thread):
    def __init__(self, fizzy, net, mean, std):
        super().__init__(daemon=True)
        self.fizzy = fizzy
        self._net  = net
        self._mean = mean
        self._std  = std

        self._lock    = threading.Lock()
        self._mode    = 'random'   # motor mode: 'stop' or 'random'
        self._running = True

        # Per-class confirmation tracking (consecutive confident detections)
        self._streak_label = None
        self._streak_count = 0

        # Exposed state for UI
        self.last_label           = None
        self.last_raw_label       = None
        self.last_conf            = 0.0
        self.last_probs           = [0.0] * len(CLASS_NAMES)
        self.motor_power          = 0.0
        self.is_connected         = False
        # Confirmed class (met its consecutive-count requirement) + live streak
        self.last_confirmed_label = None
        self.streak_label         = None
        self.streak_count         = 0

        # Random motor state
        self._random_state = RandomState(_random_config, duration=None)
        self._random_state.enter()

    @property
    def motor_mode(self):
        with self._lock:
            return self._mode

    def set_mode(self, mode: str):
        with self._lock:
            self._mode = mode

    def stop(self):
        self._running = False

    def _classify(self, window_np):
        x = window_np.T.astype(np.float32)
        n = self._mean.shape[0]
        x = x[:n]
        x = (x - self._mean) / (self._std + 1e-8)
        x = torch.from_numpy(x).unsqueeze(0)
        with torch.no_grad():
            probs = F.softmax(self._net(x), dim=1)[0]
        conf, idx = torch.max(probs, dim=0)
        raw_label = CLASS_NAMES[idx.item()]
        threshold = CONFIDENCE_THRESHOLD_LIVE.get(raw_label, CONFIDENCE_THRESHOLD)
        label = raw_label if conf.item() >= threshold else None
        return label, raw_label, conf.item(), probs.tolist()

    def run(self):
        buffer             = deque(maxlen=WINDOW_SIZE)
        samples_since_pred = 0
        cooldown_until     = 0.0
        _prev_time         = time.time()
        _prev_mode         = 'random'

        while self._running:
            with self._lock:
                mode = self._mode

            now = time.time()
            dt  = max(now - _prev_time, 1e-6)
            _prev_time = now

            # Re-enter random state on motor-mode transition
            if mode == 'random' and _prev_mode != 'random':
                self._random_state.enter()
            _prev_mode = mode

            # Motor control
            if mode == 'random':
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

            buffer.append(packet_to_features(data, motor))
            samples_since_pred += 1

            if len(buffer) < WINDOW_SIZE or samples_since_pred < STRIDE:
                continue

            samples_since_pred = 0

            if now < cooldown_until:
                continue

            label, raw_label, conf, probs = self._classify(np.array(buffer))

            self.last_label     = label
            self.last_raw_label = raw_label
            self.last_conf      = conf
            self.last_probs     = probs

            # Update the consecutive-confident-detection streak and decide whether
            # the current class is "confirmed" (met its required count in a row).
            confirmed_label = None
            if label is not None:
                if label == self._streak_label:
                    self._streak_count += 1
                else:
                    self._streak_label = label
                    self._streak_count = 1
                required = CONSECUTIVE_COUNT_LIVE.get(label, 1)
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

            prob_str = '  '.join(f"{n}:{p*100:.1f}%" for n, p in zip(CLASS_NAMES, probs))
            accepted = "OK" if label is not None else "low-conf"
            confirm_str = f"  CONFIRMED={confirmed_label.upper()}" if confirmed_label else ""
            print(f"{raw_label.upper():6s} {conf*100:5.1f}%  ({accepted})  |  {prob_str}{confirm_str}")

            # Per-class cooldown: wait before classifying again after a confirmation
            if confirmed_label is not None:
                cooldown_until = now + COOLDOWN_AFTER_CLASS_LIVE.get(confirmed_label, 0.0)


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
        self._bars: list[QtWidgets.QProgressBar] = []

        for i, name in enumerate(CLASS_NAMES):
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
            self._bars.append(bar)
            grid.addWidget(lbl, i, 0)
            grid.addWidget(bar, i, 1)
        layout.addLayout(grid)

    def refresh(self, label, raw_label, conf, probs):
        text  = raw_label.upper() if raw_label else "---"
        color = '#4CAF50' if label is not None else '#FF9800'
        self.gesture_label.setText(text)
        self.gesture_label.setStyleSheet(f"color: {color};")
        self.conf_label.setText(f"{conf * 100:.1f}%" if raw_label else "---")
        for i in range(len(self._bars)):
            pct = int(probs[i] * 100) if i < len(probs) else 0
            self._bars[i].setValue(pct)


class RequirementsWidget(QtWidgets.QGroupBox):
    """Shows the confirmed class and, per class, how many confident detections in
    a row are required plus the cooldown before classifying resumes."""

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

        self._rows: dict[str, tuple] = {}
        for i, name in enumerate(CLASS_NAMES, start=1):
            color    = CLASS_COLORS.get(name, '#2196F3')
            name_lbl = QtWidgets.QLabel(name)
            name_lbl.setStyleSheet(f"font-weight: bold; color: {color};")
            streak_lbl = QtWidgets.QLabel(f"0/{CONSECUTIVE_COUNT_LIVE.get(name, 1)}")
            cd_lbl     = QtWidgets.QLabel(f"{COOLDOWN_AFTER_CLASS_LIVE.get(name, 0.0):.1f}s")
            self._grid.addWidget(name_lbl,   i, 0)
            self._grid.addWidget(streak_lbl, i, 1)
            self._grid.addWidget(cd_lbl,     i, 2)
            self._rows[name] = (name_lbl, streak_lbl, cd_lbl)
        layout.addLayout(self._grid)

    def refresh(self, confirmed_label, streak_label, streak_count):
        if confirmed_label:
            color = CLASS_COLORS.get(confirmed_label, '#4CAF50')
            self.confirmed_label.setText(f"Confirmed: {confirmed_label.upper()}")
            self.confirmed_label.setStyleSheet(f"color: {color};")
        else:
            self.confirmed_label.setText("Confirmed: ---")
            self.confirmed_label.setStyleSheet("")

        for name, (_name_lbl, streak_lbl, cd_lbl) in self._rows.items():
            required = CONSECUTIVE_COUNT_LIVE.get(name, 1)
            count    = streak_count if name == streak_label else 0
            streak_lbl.setText(f"{count}/{required}")
            streak_lbl.setStyleSheet("font-weight: bold; color: #4CAF50;" if count > 0 else "")
            cd_lbl.setText(f"{COOLDOWN_AFTER_CLASS_LIVE.get(name, 0.0):.1f}s")


class TuningWidget(QtWidgets.QGroupBox):
    """Editable per-class tuning: confidence threshold, number of confident
    detections required in a row, and the cooldown after a confirmation. Edits
    mutate the shared settings dicts in place so they take effect immediately;
    'Save to settings.py' persists them to disk."""

    def __init__(self, parent=None):
        super().__init__("Tuning", parent)
        layout = QtWidgets.QVBoxLayout(self)
        layout.setSpacing(6)

        grid = QtWidgets.QGridLayout()
        grid.setHorizontalSpacing(10)
        grid.setVerticalSpacing(3)
        for col, title in enumerate(("Class", "Conf ≥", "In a row", "Cooldown s")):
            h = QtWidgets.QLabel(title)
            h.setStyleSheet("font-weight: bold; font-size: 11px; color: #888;")
            grid.addWidget(h, 0, col)

        for i, name in enumerate(CLASS_NAMES, start=1):
            color    = CLASS_COLORS.get(name, '#2196F3')
            name_lbl = QtWidgets.QLabel(name)
            name_lbl.setStyleSheet(f"font-weight: bold; color: {color};")

            thr_spin = QtWidgets.QDoubleSpinBox()
            thr_spin.setRange(0.0, 1.0)
            thr_spin.setSingleStep(0.05)
            thr_spin.setDecimals(2)
            thr_spin.setValue(float(CONFIDENCE_THRESHOLD_LIVE.get(name, CONFIDENCE_THRESHOLD)))
            thr_spin.valueChanged.connect(
                lambda v, k=name: CONFIDENCE_THRESHOLD_LIVE.__setitem__(k, float(v))
            )

            cnt_spin = QtWidgets.QSpinBox()
            cnt_spin.setRange(1, 99)
            cnt_spin.setValue(int(CONSECUTIVE_COUNT_LIVE.get(name, 1)))
            cnt_spin.valueChanged.connect(
                lambda v, k=name: CONSECUTIVE_COUNT_LIVE.__setitem__(k, int(v))
            )

            cd_spin = QtWidgets.QDoubleSpinBox()
            cd_spin.setRange(0.0, 60.0)
            cd_spin.setSingleStep(0.5)
            cd_spin.setDecimals(1)
            cd_spin.setValue(float(COOLDOWN_AFTER_CLASS_LIVE.get(name, 0.0)))
            cd_spin.valueChanged.connect(
                lambda v, k=name: COOLDOWN_AFTER_CLASS_LIVE.__setitem__(k, float(v))
            )

            grid.addWidget(name_lbl, i, 0)
            grid.addWidget(thr_spin, i, 1)
            grid.addWidget(cnt_spin, i, 2)
            grid.addWidget(cd_spin,  i, 3)
        layout.addLayout(grid)

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


class LiveClassifier(QtWidgets.QMainWindow):
    def __init__(self, worker: WorkerThread):
        super().__init__()
        self.worker = worker
        self.setWindowTitle(f"Live Classifier  |  Model: {MODEL_NUMBER}")
        self.resize(500, 900)

        cw = QtWidgets.QWidget()
        self.setCentralWidget(cw)
        root = QtWidgets.QVBoxLayout(cw)
        root.setSpacing(10)
        root.setContentsMargins(12, 12, 12, 12)

        # Status row
        status_row = QtWidgets.QHBoxLayout()
        self.conn_label = QtWidgets.QLabel("Fizzy: --")
        self.conn_label.setStyleSheet("font-weight: bold; font-size: 12px;")
        self.model_label = QtWidgets.QLabel(f"Model: {MODEL_NUMBER}")
        self.model_label.setStyleSheet("font-weight: bold; font-size: 12px;")
        status_row.addWidget(self.conn_label)
        status_row.addStretch()
        status_row.addWidget(self.model_label)
        root.addLayout(status_row)

        self.classify_widget     = ClassificationWidget()
        self.requirements_widget = RequirementsWidget()
        self.tuning_widget       = TuningWidget()
        self.motor_panel         = MotorPanel()

        root.addWidget(self.requirements_widget, stretch=2)
        root.addWidget(self.classify_widget)
        root.addWidget(self.tuning_widget, stretch=1)
        root.addWidget(self.motor_panel)

        self.motor_panel.mode_changed.connect(self.worker.set_mode)

        self._timer = QtCore.QTimer()
        self._timer.timeout.connect(self._refresh)
        self._timer.start(50)  # 20 Hz UI refresh

        QtGui.QShortcut(QtGui.QKeySequence('P'), self).activated.connect(
            lambda: self.motor_panel._select('random')
        )
        QtGui.QShortcut(QtGui.QKeySequence('S'), self).activated.connect(
            lambda: self.motor_panel._select('stop')
        )

    def _refresh(self):
        w = self.worker
        connected = w.is_connected
        self.conn_label.setText("Fizzy: Connected" if connected else "Fizzy: Disconnected")
        self.conn_label.setStyleSheet(
            f"font-weight: bold; font-size: 12px; color: {'#4CAF50' if connected else '#F44336'};"
        )

        self.classify_widget.refresh(w.last_label, w.last_raw_label, w.last_conf, w.last_probs)
        self.requirements_widget.refresh(w.last_confirmed_label, w.streak_label, w.streak_count)
        self.motor_panel.sync_mode(w.motor_mode)
        self.motor_panel.refresh(w.motor_power)

    def closeEvent(self, event):
        self._timer.stop()
        self.worker.stop()
        super().closeEvent(event)


# ===== ENTRY POINT =====

def main():
    print(f"Loading model: {MODEL_PATH}")
    net = IMUNet()
    net.load_state_dict(torch.load(MODEL_PATH, map_location='cpu'))
    net.eval()
    mean = np.load(os.path.join(ROOT_PATH, "Mean, std", f"imu_mean_v{data_version}.npy")).astype(np.float32)
    std  = np.load(os.path.join(ROOT_PATH, "Mean, std", f"imu_std_v{data_version}.npy")).astype(np.float32)
    if mean.ndim == 3:
        mean = mean.squeeze(0)
        std  = std.squeeze(0)
    print(f"Model loaded. Classes: {CLASS_NAMES}  |  norm shape: {mean.shape}")

    fizzy  = Fizzy()
    worker = WorkerThread(fizzy, net, mean, std)
    worker.start()
    print("Worker started. P = random motor  |  S = stop  |  Close window to quit")

    app = QtWidgets.QApplication(sys.argv)
    win = LiveClassifier(worker)
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
