"""Live classification workflow manager for the Live Classification tab."""

from __future__ import annotations

import sys
from pathlib import Path
import threading
import time

import numpy as np
from PyQt6 import QtCore, QtWidgets

# Ensure BEP code dir is on path for sibling imports
_parent_dir = str(Path(__file__).parent.parent)
if _parent_dir not in sys.path:
    sys.path.insert(0, _parent_dir)

# Put the CNN code + settings dirs (DataAndModels/CNN/Model Running) on sys.path
# so `import cnn_wrapper` below resolves.
from utilities.cnn_paths import ensure_on_path
ensure_on_path()

from utilities.imu_data_extractor import (
    extract_angular_velocity_from_packet,
    extract_euler_from_packet,
    extract_linear_acceleration_from_packet,
)
from utilities.live_data_windower import extract_features
import utilities.live_random_forest as live_random_forest
try:
    from cnn_wrapper import CNNModelWrapper
    _CNN_AVAILABLE = True
except ImportError:
    _CNN_AVAILABLE = False


class LiveClassificationManager:
    """Owns live-classification state and behaviour.

    Supports two model types:
      - 'rf'  : scikit-learn Random Forest (via live_random_forest)
      - 'cnn' : PyTorch CNN (via CNNModelWrapper)

    The active type is switched automatically when a model is loaded via
    on_model_loaded() or on_cnn_model_loaded().
    """

    def __init__(self, main_window):
        self.main = main_window

        # Active model
        self.live_classification_model = None   # RF model object
        self.cnn_model = None                   # CNNModelWrapper object
        self.active_model_type: str | None = None  # 'rf' | 'cnn'

        # Window / timing settings (driven by control panel)
        self.live_classification_samples_per_window = 128
        self.live_classification_seconds_per_window = None
        self.samples_in_seconds_window = 0
        self.live_classification_checks_per_second = 1.6
        self.live_classification_last_prediction_time = 0.0

        # Prediction results (shared between worker thread and UI timer)
        self.live_classification_latest_prediction: list = []
        self.live_classification_latest_message = "Waiting for model and data…"
        self.live_classification_latest_error = ""

        # Thread / timer handles
        self.live_classification_ui_update_timer: QtCore.QTimer | None = None
        self.live_classification_prediction_thread: threading.Thread | None = None
        self.live_classification_running = False

    # ------------------------------------------------------------------
    # Model loading callbacks (called from panel signals)
    # ------------------------------------------------------------------

    def on_model_loaded(self, model):
        """Called when an RF model is loaded from the panel."""
        self.live_classification_model = model
        self.cnn_model = None
        self.active_model_type = 'rf'
        self.live_classification_latest_message = "RF model loaded, waiting for data…"
        self.start()

    def on_cnn_model_loaded(self, cnn_wrapper):
        """Called when a CNN model is loaded from the panel."""
        self.cnn_model = cnn_wrapper
        self.live_classification_model = None
        self.active_model_type = 'cnn'
        self.live_classification_latest_message = "CNN model loaded, waiting for data…"
        self.start()

    # ------------------------------------------------------------------
    # Window / timing settings (synced from control panel)
    # ------------------------------------------------------------------

    def on_settings_changed(self, *args):
        window_size = self.main.control_panel.get_window_size()
        window_unit = self.main.control_panel.get_window_unit().lower()
        overlap_percent = self.main.control_panel.get_overlap_percent()
        sample_rate = 104.0

        if window_unit == "seconds":
            self.live_classification_samples_per_window = None
            self.live_classification_seconds_per_window = float(window_size)
            stride_sec = max(0.01, float(window_size) * (1.0 - overlap_percent / 100.0))
            self.live_classification_checks_per_second = 1.0 / stride_sec
        else:
            self.live_classification_samples_per_window = max(1, int(round(window_size)))
            self.live_classification_seconds_per_window = None
            stride_smp = max(1, int(round(
                self.live_classification_samples_per_window * (1.0 - overlap_percent / 100.0)
            )))
            self.live_classification_checks_per_second = sample_rate / stride_smp

        self.live_classification_checks_per_second = max(
            0.1, float(self.live_classification_checks_per_second)
        )
        self.live_classification_last_prediction_time = time.time()

    def on_browse_model(self):
        from utilities.dialog_utils import get_open_file_name

        model_path, _ = get_open_file_name(
            self.main,
            "Select Model",
            "",
            "Model files (*.joblib *.pkl);;All files (*)",
        )
        if model_path:
            self.main.live_classification_panel.add_model_path(model_path)

    # ------------------------------------------------------------------
    # Start / stop
    # ------------------------------------------------------------------

    def start(self):
        has_model = (
            (self.active_model_type == 'rf' and self.live_classification_model is not None)
            or (self.active_model_type == 'cnn' and self.cnn_model is not None)
        )
        if not has_model:
            self.live_classification_latest_message = "No model loaded"
            return

        # Stop any running worker cleanly before restarting
        self.live_classification_running = False
        if (self.live_classification_prediction_thread
                and self.live_classification_prediction_thread.is_alive()):
            self.live_classification_prediction_thread.join(timeout=0.3)

        self.live_classification_running = True
        self.live_classification_prediction_thread = threading.Thread(
            target=self._prediction_worker, daemon=True
        )
        self.live_classification_prediction_thread.start()

        if self.live_classification_ui_update_timer is None:
            self.live_classification_ui_update_timer = QtCore.QTimer()
            self.live_classification_ui_update_timer.timeout.connect(self.update_ui)
        self.live_classification_ui_update_timer.start(250)
        self.live_classification_latest_message = "Classification running…"

    def stop(self):
        self.live_classification_running = False
        if self.live_classification_ui_update_timer:
            self.live_classification_ui_update_timer.stop()

    # ------------------------------------------------------------------
    # Data collection helper
    # ------------------------------------------------------------------

    def _collect_data_list(self):
        """Return (data_list, wait_message). wait_message is None on success."""
        with self.main.data_lock:
            buffer_len = len(self.main.data_buffer)

            # CNN uses the model's fixed window size
            if self.active_model_type == 'cnn' and self.cnn_model is not None:
                cnn_win = self.cnn_model.window_size
                if buffer_len < cnn_win:
                    return None, f"Waiting for CNN data: {buffer_len}/{cnn_win} samples"
                return list(self.main.data_buffer)[-cnn_win:], None

            # RF: use control panel window settings
            if self.live_classification_samples_per_window is not None:
                needed = self.live_classification_samples_per_window
                if buffer_len < needed:
                    return None, f"Waiting for data: {buffer_len}/{needed} samples"
                return list(self.main.data_buffer)[-needed:], None

            if self.live_classification_seconds_per_window is not None:
                needed_approx = self.live_classification_seconds_per_window * 128
                if buffer_len < needed_approx:
                    return None, (
                        f"Waiting for data: {buffer_len} samples "
                        f"(need ~{needed_approx:.0f} for "
                        f"{self.live_classification_seconds_per_window}s window)"
                    )
                latest_ts = self.main.data_buffer[-1][0]
                cutoff_ts = latest_ts - self.live_classification_seconds_per_window * 1_000_000
                data_list = [e for e in self.main.data_buffer if e[0] >= cutoff_ts]
                self.samples_in_seconds_window = len(data_list)
                return data_list, None

            return None, "Invalid window configuration"

    # ------------------------------------------------------------------
    # Prediction worker thread
    # ------------------------------------------------------------------

    def _prediction_worker(self):
        while self.live_classification_running:
            try:
                now = time.time()
                interval = 1.0 / self.live_classification_checks_per_second
                if (now - self.live_classification_last_prediction_time) < interval:
                    time.sleep(0.01)
                    continue

                self.live_classification_last_prediction_time = now

                data_list, wait_msg = self._collect_data_list()
                if data_list is None:
                    self.live_classification_latest_message = wait_msg
                    continue

                t0 = time.time()

                if self.active_model_type == 'cnn' and self.cnn_model is not None:
                    try:
                        result = self._predict_with_cnn(data_list)
                        self.live_classification_latest_prediction = result
                        pt = f"({time.time()-t0:.3f}s)"
                        if result:
                            top_name, top_cert = result[0]
                            threshold_pct = self.cnn_model.confidence_threshold * 100.0
                            if top_cert < threshold_pct:
                                self.live_classification_latest_message = (
                                    f"Unsure ({top_name}: {top_cert:.1f}%) {pt}"
                                )
                            else:
                                self.live_classification_latest_message = (
                                    f"{top_name}: {top_cert:.1f}% {pt}"
                                )
                        else:
                            self.live_classification_latest_message = f"No CNN prediction {pt}"
                    except Exception as exc:
                        self.live_classification_latest_error = f"CNN error: {str(exc)[:100]}"
                        self.live_classification_latest_message = "Error during CNN prediction"
                else:
                    # RF path
                    data_dict = self._extract_imu_features(data_list)
                    if data_dict is None:
                        continue
                    try:
                        features_dict = extract_features(
                            data_dict
                        )
                        if features_dict is None:
                            continue
                        result = live_random_forest.predict_and_sort(
                            self.live_classification_model, features_dict
                        )
                        self.live_classification_latest_prediction = result
                        pt = f"({time.time()-t0:.3f}s)"
                        if result:
                            top_name, top_cert = result[0]
                            self.live_classification_latest_message = (
                                f"{top_name}: {top_cert:.1f}% {pt}"
                            )
                        else:
                            self.live_classification_latest_message = f"No prediction {pt}"
                    except Exception as exc:
                        self.live_classification_latest_error = f"RF error: {str(exc)[:100]}"
                        self.live_classification_latest_message = "Error during prediction"

            except Exception as exc:
                self.live_classification_latest_error = f"Thread error: {str(exc)[:100]}"
                time.sleep(0.1)

    # ------------------------------------------------------------------
    # CNN inference
    # ------------------------------------------------------------------

    def _predict_with_cnn(self, data_list: list) -> list:
        """Extract raw IMU channels from UDP packets and run CNN inference.

        Returns a list of (class_name, certainty_pct) sorted by certainty desc.
        """
        cnn_model = self.cnn_model
        input_channels = cnn_model.input_channels

        acc_xs, acc_ys, acc_zs = [], [], []
        gyro_xs, gyro_ys, gyro_zs = [], [], []
        motor_inputs = []

        for packet in data_list:
            try:
                ax, ay, az = extract_linear_acceleration_from_packet(packet, True)
                gx, gy, gz = extract_angular_velocity_from_packet(packet, True)
            except Exception:
                ax, ay, az = 0.0, 0.0, 0.0
                gx, gy, gz = 0.0, 0.0, 0.0
            acc_xs.append(ax)
            acc_ys.append(ay)
            acc_zs.append(az)
            gyro_xs.append(gx)
            gyro_ys.append(gy)
            gyro_zs.append(gz)
            motor_inputs.append(float(packet[1]) if len(packet) > 1 else 0.0)

        channels = [acc_xs, acc_ys, acc_zs, gyro_xs, gyro_ys, gyro_zs]
        if input_channels >= 7:
            channels.append(motor_inputs)
        channels = channels[:input_channels]

        # Shape: (1, input_channels, window_size)
        features = np.array(channels, dtype=np.float32)[np.newaxis, :, :]
        probs = cnn_model.predict_proba(features)[0]  # (n_classes,)
        class_names = cnn_model.get_class_names()

        result = [(class_names[i], float(probs[i]) * 100.0) for i in range(len(class_names))]
        result.sort(key=lambda x: x[1], reverse=True)
        return result

    # ------------------------------------------------------------------
    # RF feature extraction
    # ------------------------------------------------------------------

    def _extract_imu_features(self, data_list: list) -> dict | None:
        try:
            rolls, pitches, yaws = [], [], []
            acc_xs, acc_ys, acc_zs = [], [], []
            gyro_xs, gyro_ys, gyro_zs = [], [], []

            for entry in data_list:
                roll, pitch, yaw = extract_euler_from_packet(entry)
                ax, ay, az = extract_linear_acceleration_from_packet(entry, True)
                gx, gy, gz = extract_angular_velocity_from_packet(entry, True)
                rolls.append(np.degrees(roll))
                pitches.append(np.degrees(pitch))
                yaws.append(np.degrees(yaw))
                acc_xs.append(ax)
                acc_ys.append(ay)
                acc_zs.append(az)
                gyro_xs.append(gx)
                gyro_ys.append(gy)
                gyro_zs.append(gz)

            return {
                "roll":   np.array(rolls),
                "pitch":  np.array(pitches),
                "yaw":    np.array(yaws),
                "acc_x":  np.array(acc_xs),
                "acc_y":  np.array(acc_ys),
                "acc_z":  np.array(acc_zs),
                "gyro_x": np.array(gyro_xs),
                "gyro_y": np.array(gyro_ys),
                "gyro_z": np.array(gyro_zs),
            }
        except Exception as exc:
            self.live_classification_latest_error = f"Feature extraction error: {str(exc)[:100]}"
            return None

    # ------------------------------------------------------------------
    # UI update (called by QTimer every 250 ms from the main thread)
    # ------------------------------------------------------------------

    def update_ui(self):
        try:
            with self.main.data_lock:
                buffer_len = len(self.main.data_buffer)

            # Build model info line
            if self.active_model_type == 'cnn' and self.cnn_model is not None:
                cn = self.cnn_model.get_class_names()
                model_info = (
                    f"CNN (win={self.cnn_model.window_size}) "
                    f"→ [{', '.join(cn)}]"
                )
                window_info = f"CNN window: {self.cnn_model.window_size} samples (fixed)"
            elif self.active_model_type == 'rf' and self.live_classification_model is not None:
                model_info = (
                    getattr(self.live_classification_model, "__class__", type(None)).__name__
                )
                if self.live_classification_samples_per_window is not None:
                    window_info = f"Window: {self.live_classification_samples_per_window} samples"
                else:
                    window_info = (
                        f"Window: {self.live_classification_seconds_per_window}s "
                        f"/ {self.samples_in_seconds_window} samples"
                    )
            else:
                model_info = "None"
                window_info = ""

            summary_lines = [
                f"Model: {model_info}",
                window_info,
                f"Rate: {self.live_classification_checks_per_second:.2f} checks/s  |  "
                f"Buffer: {buffer_len} samples",
                "",
                f"→  {self.live_classification_latest_message}",
            ]

            if self.live_classification_latest_prediction:
                summary_lines.append("")
                summary_lines.append("Classes:")
                for class_name, cert in self.live_classification_latest_prediction[:6]:
                    bar = "█" * int(cert / 5)
                    summary_lines.append(f"  {class_name:<14} {bar:<20} {cert:5.1f}%")

            if self.live_classification_latest_error:
                summary_lines.extend(["", f"⚠ {self.live_classification_latest_error}"])

            self.main.live_classification_panel.set_summary_text("\n".join(summary_lines))
        except Exception as exc:
            print(f"LiveClassificationManager.update_ui error: {exc}")
