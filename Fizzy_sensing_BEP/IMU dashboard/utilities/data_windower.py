import os
import json
import numpy as np


def apply_hanning_to_window(data):
    """
    Applies a Hanning window to a single window of data.
    
    Args:
        data: list of dictionaries where each dict is one sample/row (e.g., [{'acc_x': 0.034, 'gyro_z': 0.0}, {...}])
        
    Returns:
        list of dictionaries: contains one dict with windowed data, where each key maps to a windowed array
    """
    window_size = len(data)
    
    # Create Hanning window
    hanning_window = np.hanning(window_size)
    
    # Get all keys from the first sample
    keys = data[0].keys()
    numeric_exclude_keys = {'timestamp', 'label', 'markers'}
    
    windowed_dict = {}
    # For each key, extract values and multiply by Hanning window
    for key in keys:
        if key in numeric_exclude_keys:
            continue
        values = [sample[key] for sample in data]
        try:
            numeric_values = np.asarray(values, dtype=float)
        except (TypeError, ValueError):
            continue
        windowed_dict[key] = numeric_values * hanning_window
    
    return [windowed_dict]


def compute_fft(windowed_data):
    """
    Computes the FFT of windowed data for each key.
    
    Args:
        windowed_data: list of dictionaries where each dict contains windowed data for one window
        sample_rate: sampling rate in Hz (default 106 Hz)
        
    Returns: 
        list of dictionaries: each dict contains FFT magnitude spectrum for each key in that window
    """
    fft_data = []
    
    for window in windowed_data:
        fft_dict = {}
        for key, signal in window.items():
            # Compute FFT
            fft_values = np.fft.fft(signal)
            # Get magnitude spectrum (only positive frequencies)
            magnitude = np.abs(fft_values)
            fft_dict[key] = magnitude
        fft_data.append(fft_dict)
    
    return fft_data


def _extract_time_domain_features_from_signal_dict(signal_dict, exclude_keys=None):
    """
    Helper function to extract time-domain features from a signal dictionary.
    This is the single source of truth for time-domain feature definitions.
    Replaces acc_x, acc_y with acc_xy magnitude and gyro_x, gyro_y with gyro_xy magnitude.
    
    Args:
        signal_dict: dictionary mapping signal names to numpy arrays (one window of data)
        exclude_keys: set or list of key names to exclude from feature extraction
        
    Returns:
        Dictionary of time-domain features
    """
    if exclude_keys is None:
        exclude_keys = set()
    else:
        exclude_keys = set(exclude_keys)
    
    standard_exclude_keys = {'timestamp', 'label', 'markers', 'roll', 'pitch', 'yaw', 'acc_x', 'acc_y', 'gyro_x', 'gyro_y'}
    exclude_keys.update(standard_exclude_keys)
    
    # Create a modified signal dict with xy magnitudes
    modified_signal_dict = dict(signal_dict)
    
    # Compute acc_xy magnitude if both acc_x and acc_y are available
    if 'acc_x' in signal_dict and 'acc_y' in signal_dict:
        modified_signal_dict['acc_xy'] = np.sqrt(signal_dict['acc_x']**2 + signal_dict['acc_y']**2)
        if 'acc_z' in signal_dict:
            modified_signal_dict['acc_xyz'] = np.sqrt(signal_dict['acc_x']**2 + signal_dict['acc_y']**2 + signal_dict['acc_z']**2)
    
    # Compute gyro_xy magnitude if both gyro_x and gyro_y are available
    if 'gyro_x' in signal_dict and 'gyro_y' in signal_dict:
        modified_signal_dict['gyro_xy'] = np.sqrt(signal_dict['gyro_x']**2 + signal_dict['gyro_y']**2)
        if 'gyro_z' in signal_dict:
            modified_signal_dict['gyro_xyz'] = np.sqrt(signal_dict['gyro_x']**2 + signal_dict['gyro_y']**2 + signal_dict['gyro_z']**2)
    
    feature_dict = {}
    
    for key, signal in modified_signal_dict.items():
        if key in exclude_keys:
            continue
        
        feature_dict[f'{key}_mean'] = np.mean(signal)
        feature_dict[f'{key}_var'] = np.var(signal)
        feature_dict[f'{key}_max'] = np.max(signal)
        feature_dict[f'{key}_min'] = np.min(signal)
        feature_dict[f'{key}_ZCR'] = np.sum((signal[:-1] < 0) & (signal[1:] >= 0)) + np.sum((signal[:-1] >= 0) & (signal[1:] < 0))  # Zero Crossing Rate
        feature_dict[f'{key}_MCR'] = np.sum(np.abs(signal[1:] - signal[:-1])) / len(signal)  # Mean Crossing Rate
        feature_dict[f'{key}_IQR'] = np.percentile(signal, 75) - np.percentile(signal, 25)
        feature_dict[f'{key}_MAD'] = np.mean(np.abs(signal - np.mean(signal)))  # Mean Absolute Deviation
        feature_dict[f'{key}_RMS'] = np.sqrt(np.mean(signal**2))  # Root Mean Square
        feature_dict[f'{key}_SMA'] = np.sum(np.abs(signal)) / len(signal)  # Signal Magnitude Area
        feature_dict[f'{key}_range'] = np.max(signal) - np.min(signal)
        feature_dict[f'{key}_skew'] = np.mean((signal - np.mean(signal))**3) / (np.std(signal)**3 + 1e-6)  # Skewness
        feature_dict[f'{key}_energy'] = np.sum(signal**2)  # Energy of the signal
        feature_dict[f'{key}_entropy'] = -np.sum((signal**2) * np.log(signal**2 + 1e-6))  # Shannon entropy
    
    return feature_dict


def _extract_fft_features_from_signal_dict(fft_dict, exclude_keys=None):
    """
    Helper function to extract FFT-based features from a spectrum dictionary.
    This is the single source of truth for FFT feature definitions.
    Replaces acc_x, acc_y with acc_xy magnitude and gyro_x, gyro_y with gyro_xy magnitude.
    
    Args:
        fft_dict: dictionary mapping signal names to magnitude spectra (numpy arrays)
        exclude_keys: set or list of key names to exclude from feature extraction
        
    Returns:
        Dictionary of FFT-based features
    """
    if exclude_keys is None:
        exclude_keys = set()
    else:
        exclude_keys = set(exclude_keys)
    
    standard_exclude_keys = {'timestamp', 'label', 'roll', 'pitch', 'yaw', 'markers', 'acc_x', 'acc_y', 'gyro_x', 'gyro_y'}
    exclude_keys.update(standard_exclude_keys)
    
    # Create a modified FFT dict with xy magnitudes
    modified_fft_dict = dict(fft_dict)
    
    # Compute acc_xy magnitude if both acc_x and acc_y are available
    if 'acc_x' in fft_dict and 'acc_y' in fft_dict:
        modified_fft_dict['acc_xy'] = np.sqrt(fft_dict['acc_x']**2 + fft_dict['acc_y']**2)
    
    # Compute gyro_xy magnitude if both gyro_x and gyro_y are available
    if 'gyro_x' in fft_dict and 'gyro_y' in fft_dict:
        modified_fft_dict['gyro_xy'] = np.sqrt(fft_dict['gyro_x']**2 + fft_dict['gyro_y']**2)

    # Compute acc_xyz magnitude if all three components are available
    if 'acc_x' in fft_dict and 'acc_y' in fft_dict and 'acc_z' in fft_dict:
        modified_fft_dict['acc_xyz'] = np.sqrt(fft_dict['acc_x']**2 + fft_dict['acc_y']**2 + fft_dict['acc_z']**2)

    # Compute gyro_xyz magnitude if all three components are available
    if 'gyro_x' in fft_dict and 'gyro_y' in fft_dict and 'gyro_z' in fft_dict:
        modified_fft_dict['gyro_xyz'] = np.sqrt(fft_dict['gyro_x']**2 + fft_dict['gyro_y']**2 + fft_dict['gyro_z']**2)

    feature_dict = {}
    
    for key, magnitude in modified_fft_dict.items():
        if key in exclude_keys:
            continue
        
        # Spectral centroid
        feature_dict[f'{key}_spectral_centroid'] = np.sum(np.arange(len(magnitude)) * magnitude) / (np.sum(magnitude) + 1e-6)
        # Spectral variance
        feature_dict[f'{key}_spectral_var'] = np.sum(((np.arange(len(magnitude)) - feature_dict[f'{key}_spectral_centroid'])**2) * magnitude) / (np.sum(magnitude) + 1e-6)
        # Dominant frequency
        dominant_freq_idx = np.argmax(magnitude)
        feature_dict[f'{key}_dominant_freq'] = dominant_freq_idx
        # Spectral entropy
        feature_dict[f'{key}_spectral_entropy'] = -np.sum((magnitude / (np.sum(magnitude) + 1e-6)) * np.log(magnitude / (np.sum(magnitude) + 1e-6) + 1e-6))
    
    return feature_dict


def extract_fft_features(fft_data, exclude_keys=None):
    """
    Extracts features from FFT data for each window.
    
    Args:
        fft_data: list of dictionaries where each dict contains FFT magnitude spectrum for each key in that window
        exclude_keys: set or list of key names to exclude from feature extraction (default None for no exclusion)
    """
    features = []
    
    for fft_dict in fft_data:
        feature_dict = _extract_fft_features_from_signal_dict(fft_dict, exclude_keys=exclude_keys)
        features.append(feature_dict)
    
    return features

def extract_time_domain_features(windowed_data: list, exclude_keys=None):
    """
    Extracts time-domain features from windowed data for each window.

    Args:
        windowed_data: list of dictionaries where each dict contains windowed data for one window
        exclude_keys: set or list of key names to exclude from feature extraction (default None for no exclusion)
    """
    features = []
    
    for window in windowed_data:
        window = window.copy()
        feature_dict = _extract_time_domain_features_from_signal_dict(window, exclude_keys=exclude_keys)
        features.append(feature_dict)
    
    return features

def load_labels_from_json(json_filepath):
    """
    Load window labels from a JSON file created by interactive_labeler.
    
    Args:
        json_filepath: path to the JSON labels file
        
    Returns:
        Dictionary mapping window indices to labels, or empty dict if file not found
    """
    if not os.path.exists(json_filepath):
        print(f"Labels file not found: {json_filepath}")
        return {}
    
    try:
        with open(json_filepath, 'r') as f:
            data = json.load(f)
        labels = {}
        for window_idx_str, info in data.get('labeled_windows', {}).items():
            if isinstance(info, dict):
                label = info.get('label', info.get('class'))
            else:
                label = info
            if label is None:
                continue
            labels[int(window_idx_str)] = label
        return labels
    except Exception as e:
        print(f"Error loading labels from {json_filepath}: {e}")
        return {}


def compute_shake_like_score(fft_dict, prominence_threshold=3.0):
    """
    Detects if acceleration has a single prominent frequency (shake-like motion) on any axis.
    A shake produces a sharp, isolated peak in the FFT of acceleration on one or more axes.
    
    Properties of a shake-like motion:
    - Dominant frequency is typically between 0 and 6 (FFT bins, covers 0-5.7 Hz)
    - Peak magnitude is larger than 0.15 (lowered threshold for better detection)
    - Magnitudes at ±1 frequencies from the peak should be substantially lower (flexible criterion)
    
    Args:
        fft_dict: FFT magnitude spectrum dictionary (from one window)
        prominence_threshold: deprecated, kept for backwards compatibility
        
    Returns:
        float between 0-1: confidence score that motion is shake-like (max score across axes)
    """
    axes = ['acc_x', 'acc_y', 'acc_z']
    max_score = 0.0
    
    for axis in axes:
        if axis not in fft_dict:
            continue
        
        magnitude = fft_dict[axis]
        
        # Skip DC component (index 0)
        magnitude_no_dc = magnitude[1:]
        
        if len(magnitude_no_dc) < 3:  # Need at least 3 points to calculate slopes
            continue
        
        # Find dominant peak
        dominant_idx = np.argmax(magnitude_no_dc) + 1  # +1 because we skipped DC
        dominant_magnitude = magnitude[dominant_idx]
        
        # Criterion 1: Dominant frequency should be in low range (0-6 bins, covers typical shake frequencies)
        # More lenient range to capture various shake frequencies
        if not (0 <= dominant_idx <= 6):
            continue
        
        # Criterion 2: Peak magnitude threshold (lowered from 0.3 to 0.15 for better detection)
        if dominant_magnitude <= 0.15:
            continue
        
        # Criterion 2.5: DC component (0 Hz) should be much smaller than dominant peak
        # Idle motions have large DC offset (steady state); shakes have oscillatory motion
        dc_magnitude = magnitude[0]
        max_dc = dominant_magnitude / 4.0  # DC must be <25% of peak
        if dc_magnitude >= max_dc:
            continue  # Too much DC component - likely idle or steady motion
        
        # Criterion 3: Neighbors should be lower than peak (but more flexible than before)
        # Allow partial isolation: neighbors can be up to 70% of peak
        max_neighbor_ratio = 0.4
        max_neighbor_magnitude = dominant_magnitude * max_neighbor_ratio
        
        left_neighbor = magnitude[dominant_idx - 1] if dominant_idx > 0 else 0
        right_neighbor = magnitude[dominant_idx + 1] if dominant_idx < len(magnitude) - 1 else 0
        
        # Score based on how well isolation criterion is met
        left_isolation = max(0.0, 1.0 - (left_neighbor / (max_neighbor_magnitude + 1e-6)))
        right_isolation = max(0.0, 1.0 - (right_neighbor / (max_neighbor_magnitude + 1e-6)))
        isolation_score = (left_isolation + right_isolation) / 2.0
        
        # Normalize magnitude to 0-1 range (above 0.15 is okay, above 0.5 is good)
        magnitude_score = min(1.0, (dominant_magnitude - 0.15) / 0.35)
        
        # Frequency centering score: prefer low frequencies but not too strict
        freq_center = abs(dominant_idx - 3) / 6.0  # Prefer bins around 3
        freq_score = 1.0 - freq_center
        
        # Combined score: magnitude is most important, isolation and frequency are secondary
        axis_score = (magnitude_score * 0.5 + isolation_score * 0.35 + freq_score * 0.15)
        max_score = max(max_score, axis_score)
    
    return max(0.0, min(1.0, max_score))


def compute_spin_like_score(signal_dict):
    """
    Detects if motion is spinning around Z-axis (yaw rotation).
    Key indicators: large gyro_z (consistent direction), yaw wrapping/changing, 
    and stable roll/pitch angles.
    
    Properties of a spin-like motion:
    - gyro_z is large and does not frequently cross zero (consistent direction)
    - yaw angle changes significantly (wrapping or large angle changes)
    - roll and pitch angles remain relatively stable
    
    Args:
        signal_dict: signal dictionary with raw samples (from one window)
        
    Returns:
        float between 0-1: confidence score that motion is spin-like
    """
    required_keys = {'gyro_z', 'yaw', 'roll', 'pitch'}
    if not all(k in signal_dict for k in required_keys):
        return 0.0
    
    gyro_z = signal_dict['gyro_z']
    yaw = signal_dict['yaw']
    roll = signal_dict['roll']
    pitch = signal_dict['pitch']
    
    # Criterion 1: gyro_z should be consistently large and not cross zero much
    gyro_z_magnitude = np.mean(np.abs(gyro_z))
    zero_crossings = np.sum((gyro_z[:-1] * gyro_z[1:]) < 0)
    gyro_consistency = 1.0 - min(1.0, zero_crossings / (len(gyro_z) / 8))
    
    # Normalize gyro magnitude to 0-1 (assuming max ~500 deg/s for strong spin)
    gyro_z_norm = min(1.0, gyro_z_magnitude / 500.0)
    
    # Criterion 2: yaw should be changing significantly (angle wrapping or large changes)
    yaw_diff = np.abs(np.diff(yaw))
    yaw_wrapping_events = np.sum(yaw_diff > 180)  # Crossing from 180 to -180 or vice versa
    yaw_total_change = np.sum(np.minimum(yaw_diff, 360 - yaw_diff))  # Account for wrapping
    
    yaw_changing = min(1.0, (yaw_wrapping_events * 2 + yaw_total_change) / (len(yaw) * 2))
    
    # Criterion 3: roll and pitch should be relatively stable
    roll_stability = 1.0 - min(1.0, (np.max(roll) - np.min(roll)) / 180)
    pitch_stability = 1.0 - min(1.0, (np.max(pitch) - np.min(pitch)) / 180)
    
    # Weighted combination
    spin_score = (
        gyro_consistency * 0.25 + 
        gyro_z_norm * 0.25 + 
        yaw_changing * 0.35 + 
        (roll_stability + pitch_stability) / 2 * 0.15
    )
    
    return min(1.0, spin_score)


def compute_roll_pitch_like_scores(signal_dict):
    """
    Detects if motion is rolling around X-axis or pitching around Y-axis.
    Key distinction from spin: the dominant gyro axis changes the corresponding angle significantly,
    while other angles remain stable.
    
    Properties of roll-like motion:
    - gyro_x is large and does not frequently cross zero (consistent direction)
    - roll angle changes significantly (within ±90°)
    - yaw and pitch angles remain relatively stable
    
    Properties of pitch-like motion:
    - gyro_y is large and does not frequently cross zero (consistent direction)
    - pitch angle changes significantly (within ±90°)
    - yaw and roll angles remain relatively stable
    
    Args:
        signal_dict: signal dictionary with raw samples (from one window)
        
    Returns:
        tuple: (roll_score, pitch_score) both between 0-1
    """
    required_keys = {'gyro_x', 'gyro_y', 'roll', 'pitch', 'yaw'}
    if not all(k in signal_dict for k in required_keys):
        return 0.0, 0.0
    
    gyro_x = signal_dict['gyro_x']
    gyro_y = signal_dict['gyro_y']
    roll = signal_dict['roll']
    pitch = signal_dict['pitch']
    yaw = signal_dict['yaw']
    
    # === ROLL SCORE (rotation around X-axis) ===
    gyro_x_magnitude = np.mean(np.abs(gyro_x))
    gyro_x_zero_crossings = np.sum((gyro_x[:-1] * gyro_x[1:]) < 0)
    gyro_x_consistency = 1.0 - min(1.0, gyro_x_zero_crossings / (len(gyro_x) / 8))
    
    gyro_x_norm = min(1.0, gyro_x_magnitude / 500.0)
    
    roll_range = np.max(roll) - np.min(roll)
    roll_changing = min(1.0, roll_range / 90)  # Good roll: changes significantly within ±90°
    
    # For roll motion, yaw and pitch should be stable
    yaw_stability = 1.0 - min(1.0, (np.max(yaw) - np.min(yaw)) / 180)
    pitch_stability = 1.0 - min(1.0, (np.max(pitch) - np.min(pitch)) / 90)
    
    roll_score = (
        gyro_x_consistency * 0.25 + 
        gyro_x_norm * 0.20 + 
        roll_changing * 0.35 + 
        (yaw_stability + pitch_stability) / 2 * 0.20
    )
    
    # === PITCH SCORE (rotation around Y-axis) ===
    gyro_y_magnitude = np.mean(np.abs(gyro_y))
    gyro_y_zero_crossings = np.sum((gyro_y[:-1] * gyro_y[1:]) < 0)
    gyro_y_consistency = 1.0 - min(1.0, gyro_y_zero_crossings / (len(gyro_y) / 8))
    
    gyro_y_norm = min(1.0, gyro_y_magnitude / 500.0)
    
    pitch_range = np.max(pitch) - np.min(pitch)
    pitch_changing = min(1.0, pitch_range / 90)  # Good pitch: changes significantly within ±90°
    
    # For pitch motion, yaw and roll should be stable
    yaw_stability_p = 1.0 - min(1.0, (np.max(yaw) - np.min(yaw)) / 180)
    roll_stability_p = 1.0 - min(1.0, (np.max(roll) - np.min(roll)) / 90)
    
    pitch_score = (
        gyro_y_consistency * 0.25 + 
        gyro_y_norm * 0.20 + 
        pitch_changing * 0.35 + 
        (yaw_stability_p + roll_stability_p) / 2 * 0.20
    )
    
    return min(1.0, roll_score), min(1.0, pitch_score)


def compute_tap_like_score(signal_dict):
    """
    Detects isolated sharp acceleration spikes (tap-like motion).
    A tap has a prominent peak in acceleration magnitude that is narrow in time
    and surrounded by lower-magnitude samples.
    
    Properties of a tap-like motion:
    - Peak acceleration magnitude is significantly larger than background (prominence > 2.0)
    - Peak is narrow in time (≤15% of window width at half-peak magnitude)
    - Surrounding samples outside the peak region have lower acceleration magnitude
    
    Args:
        signal_dict: signal dictionary with raw samples (from one window)
        
    Returns:
        float between 0-1: confidence score that motion is tap-like
    """
    if 'acc_xyz' not in signal_dict:
        return 0.0
    
    acc_magnitude = signal_dict['acc_xyz']
    
    # Find the peak
    peak_idx = np.argmax(acc_magnitude)
    peak_value = acc_magnitude[peak_idx]
    
    # Define peak region: ±5% of window around peak
    peak_range = max(1, int(len(acc_magnitude) * 0.05))
    start_idx = max(0, peak_idx - peak_range)
    end_idx = min(len(acc_magnitude), peak_idx + peak_range + 1)
    
    # Calculate background RMS from samples outside peak region
    surrounding_mask = np.ones(len(acc_magnitude), dtype=bool)
    surrounding_mask[start_idx:end_idx] = False
    
    if np.sum(surrounding_mask) < 2:
        return 0.0
    
    background_rms = np.sqrt(np.mean(acc_magnitude[surrounding_mask]**2))
    
    if background_rms < 1e-6:
        return 0.0
    
    # Prominence: peak value relative to background
    prominence = peak_value / (background_rms + 1e-6)
    prominence_score = min(1.0, max(0.0, (prominence - 2.0) / 5.0))  # Threshold 2.0, scale 5.0
    
    # Peak width: should be narrow (few samples)
    threshold = peak_value * 0.5
    width = np.sum(acc_magnitude > threshold)
    max_width = len(acc_magnitude) * 0.15  # Peak should be ≤15% of window width
    width_score = 1.0 - min(1.0, width / max_width)
    
    # Combine: emphasis on narrowness and prominence
    tap_score = (prominence_score * 0.5 + width_score * 0.5)
    
    return max(0.0, min(1.0, tap_score))


def compute_drop_like_score(signal_dict):
    """
    Detects sustained free fall motion (drop-like).
    A drop is characterized by sustained negative z-acceleration around -1 g.
    
    Properties of a drop-like motion:
    - acc_z is consistently around -1 g (gravitational acceleration)
    - Duration of consistent drop is significant (50+ samples at 106 Hz ≈ 0.5 seconds = score 1.0)
    - Minimal variation in acc_z during the drop period
    
    Args:
        signal_dict: signal dictionary with raw samples (from one window)
        
    Returns:
        float between 0-1: confidence score that motion is drop-like
    """
    if 'acc_z' not in signal_dict:
        return 0.0
    
    acc_z = signal_dict['acc_z']
    
    # Define drop criteria: acc_z should be negative and around -1 g
    # Tolerance: ±0.5 to account for sensor noise and gravity variations
    drop_threshold_low = -1.5
    drop_threshold_high = -0.5
    
    # Find samples that are in the drop range
    in_drop_range = (acc_z >= drop_threshold_low) & (acc_z <= drop_threshold_high)
    
    if np.sum(in_drop_range) == 0:
        return 0.0
    
    # Find the longest consecutive sequence of drop-like samples
    # Create a difference array to find transitions
    transitions = np.diff(in_drop_range.astype(int))
    
    # Find start and end indices of consecutive drop sequences
    drop_starts = np.where(transitions == 1)[0] + 1
    drop_ends = np.where(transitions == -1)[0] + 1
    
    # Handle edge cases where drop starts at beginning or ends at end
    if in_drop_range[0]:
        drop_starts = np.concatenate(([0], drop_starts))
    if in_drop_range[-1]:
        drop_ends = np.concatenate((drop_ends, [len(in_drop_range)]))
    
    if len(drop_starts) == 0 or len(drop_ends) == 0:
        return 0.0
    
    # Find longest drop sequence
    drop_lengths = drop_ends - drop_starts
    longest_drop_length = np.max(drop_lengths)
    longest_drop_start = drop_starts[np.argmax(drop_lengths)]
    longest_drop_end = drop_ends[np.argmax(drop_lengths)]
    
    # Threshold: 50 samples at 106 Hz ≈ 0.47 seconds
    drop_threshold_samples = 50
    
    # If drop is shorter than threshold, return low score
    if longest_drop_length < 10:
        return 0.0
    
    # Score based on duration: 50 samples = 1.0, linear scaling
    duration_score = min(1.0, longest_drop_length / drop_threshold_samples)
    
    # Score based on consistency: how close to -1 g are the values?
    drop_segment = acc_z[longest_drop_start:longest_drop_end]
    mean_drop_value = np.mean(drop_segment)
    std_drop_value = np.std(drop_segment)
    
    # Consistency: how close is mean to -1 g and how low is variance?
    mean_deviation = np.abs(mean_drop_value - (-1.0))
    mean_score = 1.0 - min(1.0, mean_deviation / 0.5)  # 0.5 is tolerance range
    
    # Variance score: lower variance = more consistent drop
    variance_score = 1.0 - min(1.0, std_drop_value / 0.3)
    
    # Combined score: emphasis on duration and consistency
    drop_score = (duration_score * 0.5 + mean_score * 0.25 + variance_score * 0.25)
    
    return min(1.0, max(0.0, drop_score))

# Lift like used when 'down' is included in classes, did not work

# def compute_lift_like_score(signal_dict):
#     """
#     Detects lift-like and down-like motion.
#     Returns a score from -1 to 1:
#     - Positive (0 to 1): upward acceleration (lift) with biggest positive before biggest negative
#     - Negative (-1 to 0): downward acceleration (down) with biggest negative before biggest positive
    
#     Properties of a lift-like motion:
#     - 20+ consecutive samples with acc_z positive and below 0.8 g
#     - At least 20% of these samples must be in the 0.3-0.8 g range (meaningful upward acceleration)
#     - Biggest positive value comes before biggest negative value
    
#     Properties of a down-like motion:
#     - 20+ consecutive samples with acc_z negative and above -0.8 g (around -1 g)
#     - At least 20% of these samples must be in the -0.8 to -0.3 g range (meaningful downward acceleration)
#     - Biggest negative value comes before biggest positive value
    
#     Args:
#         signal_dict: signal dictionary with raw samples (from one window)
        
#     Returns:
#         float between -1 and 1: confidence score for lift-like (positive) or down-like (negative)
#     """
#     if 'acc_z' not in signal_dict:
#         return 0.0
    
#     acc_z = signal_dict['acc_z']
    
#     min_samples_for_motion = 20
#     max_magnitude_for_motion = 0.95
#     min_magnitude_for_meaningful = 0.3
    
#     if len(acc_z) < min_samples_for_motion:
#         return 0.0
    
#     # Helper function to compute motion score
#     def compute_motion_score(acc_segment, is_upward):
#         """Compute score for either upward or downward motion."""
#         if is_upward:
#             # Upward motion: look for positive acc_z
#             valid_mask = (acc_z > 0.0) & (acc_z <= max_magnitude_for_motion)
#             min_meaningful = min_magnitude_for_meaningful
#             max_meaningful = max_magnitude_for_motion
#             target_acc = 0.4
#         else:
#             # Downward motion: look for negative acc_z (around -1 g)
#             valid_mask = (acc_z < 0.0) & (acc_z >= -max_magnitude_for_motion)
#             min_meaningful = -max_magnitude_for_motion
#             max_meaningful = -min_magnitude_for_meaningful
#             target_acc = -0.4
        
#         transitions = np.diff(valid_mask.astype(int))
#         valid_starts = np.where(transitions == 1)[0] + 1
#         valid_ends = np.where(transitions == -1)[0] + 1
        
#         # Handle edge cases
#         if valid_mask[0]:
#             valid_starts = np.concatenate([[0], valid_starts])
#         if valid_mask[-1]:
#             valid_ends = np.concatenate([valid_ends, [len(acc_z)]])
        
#         if len(valid_starts) == 0 or len(valid_ends) == 0:
#             return 0.0
        
#         # Find longest valid stretch
#         valid_lengths = valid_ends - valid_starts
#         max_valid_length = np.max(valid_lengths)
#         longest_idx = np.argmax(valid_lengths)
#         longest_start = valid_starts[longest_idx]
#         longest_end = valid_ends[longest_idx]
        
#         if max_valid_length < min_samples_for_motion:
#             return 0.0
        
#         valid_segment = acc_z[longest_start:longest_end]
        
#         # Check meaningful range
#         in_meaningful_range = np.sum((valid_segment >= min_meaningful) & (valid_segment <= max_meaningful))
#         meaningful_ratio = in_meaningful_range / len(valid_segment)
        
#         if meaningful_ratio < 0.2:
#             return 0.0
        
#         # Check before/after context
#         min_context_samples = 10
#         before_check_passed = False
#         after_check_passed = False
        
#         if longest_start >= min_context_samples:
#             before_segment = acc_z[longest_start - min_context_samples:longest_start]
#             before_mean = np.mean(before_segment)
#             if is_upward:
#                 # Lift should not be preceded by sustained downward motion
#                 if before_mean > -0.2:
#                     before_check_passed = True
#             else:
#                 # Down should not be preceded by sustained upward motion
#                 if before_mean < 0.2:
#                     before_check_passed = True
        
#         if longest_end <= len(acc_z) - min_context_samples:
#             after_segment = acc_z[longest_end:longest_end + min_context_samples]
#             after_mean = np.mean(after_segment)
#             if is_upward:
#                 # Lift should be followed by downward acceleration
#                 if after_mean < 0.0:
#                     after_check_passed = True
#             else:
#                 # Down should be followed by upward acceleration
#                 if after_mean > 0.0:
#                     after_check_passed = True
        
#         if not (before_check_passed or after_check_passed):
#             return 0.0
        
#         # Check ordering of peaks
#         max_positive_idx = np.argmax(acc_z)
#         max_negative_idx = np.argmin(acc_z)
        
#         if acc_z[max_positive_idx] > 0.0 and acc_z[max_negative_idx] < 0.0:
#             if is_upward:
#                 # For lift: biggest positive must come before biggest negative
#                 if max_positive_idx >= max_negative_idx:
#                     return 0.0
#             else:
#                 # For down: biggest negative must come before biggest positive
#                 if max_negative_idx >= max_positive_idx:
#                     return 0.0
        
#         # Compute component scores
#         meaningful_score = min(1.0, meaningful_ratio / 0.5)
#         duration_score = min(1.0, (max_valid_length - min_samples_for_motion) / (40 - min_samples_for_motion))
        
#         acc_std = np.std(valid_segment)
#         consistency_score = 1.0 - min(1.0, acc_std / 0.2)
        
#         mean_acc = np.mean(valid_segment)
#         deviation_from_target = abs(mean_acc - target_acc)
#         mean_score = 1.0 - min(1.0, deviation_from_target / 0.2)
        
#         motion_score = (meaningful_score * 0.35 + duration_score * 0.25 + 
#                        consistency_score * 0.25 + mean_score * 0.15)
        
#         return min(1.0, max(0.0, motion_score))
    
#     # Compute scores for both directions
#     upward_score = compute_motion_score(acc_z, is_upward=True)
#     downward_score = compute_motion_score(acc_z, is_upward=False)
    
#     # Return the stronger signal, with appropriate sign
#     if upward_score >= downward_score:
#         return upward_score
#     else:
#         return -downward_score


def compute_lift_like_score(signal_dict):
    """
    Detects lift-like motion: sustained upward acceleration (positive acc_z).
    
    Properties of a lift-like motion:
    - 20+ consecutive samples with acc_z positive and below 0.8 g
    - At least 20% of these samples must be in the 0.3-0.8 g range (meaningful upward acceleration)
    - Remaining samples can be lower (0.0-0.3 g) but still positive
    
    Args:
        signal_dict: signal dictionary with raw samples (from one window)
        
    Returns:
        float between 0-1: confidence score that motion is lift-like
    """
    if 'acc_z' not in signal_dict:
        return 0.0
    
    acc_z = signal_dict['acc_z']
    
    min_samples_for_lift = 20
    max_magnitude_for_lift = 0.95
    min_magnitude_for_meaningful = 0.3
    
    if len(acc_z) < min_samples_for_lift:
        return 0.0  # Window must be at least 20 samples
    
    # Find longest consecutive stretch where acc_z is positive and below 0.8 g
    valid_mask = (acc_z > 0.0) & (acc_z <= max_magnitude_for_lift)
    transitions = np.diff(valid_mask.astype(int))
    
    # Find start and end indices of consecutive valid stretches
    valid_starts = np.where(transitions == 1)[0] + 1
    valid_ends = np.where(transitions == -1)[0] + 1
    
    # Handle edge cases
    if valid_mask[0]:
        valid_starts = np.concatenate([[0], valid_starts])
    if valid_mask[-1]:
        valid_ends = np.concatenate([valid_ends, [len(acc_z)]])
    
    if len(valid_starts) == 0 or len(valid_ends) == 0:
        return 0.0
    
    # Find the longest valid stretch
    valid_lengths = valid_ends - valid_starts
    max_valid_length = np.max(valid_lengths)
    longest_idx = np.argmax(valid_lengths)
    longest_start = valid_starts[longest_idx]
    longest_end = valid_ends[longest_idx]
    
    # Require at least 20 samples of positive acceleration below 0.8 g
    if max_valid_length < min_samples_for_lift:
        return 0.0
    
    # Analyze the longest valid stretch
    valid_segment = acc_z[longest_start:longest_end]
    
    # Check that at least 20% of samples are in the meaningful range (0.3-0.8 g)
    in_meaningful_range = np.sum((valid_segment >= min_magnitude_for_meaningful) & (valid_segment <= max_magnitude_for_lift))
    meaningful_ratio = in_meaningful_range / len(valid_segment)
    
    # Require at least 20% in the meaningful range
    if meaningful_ratio < 0.2:
        return 0.0
    
    # NEW: Distinguish true lift from deceleration of downward motion
    # Check before the lift if enough space: acc_z should not be deeply negative (no sustained downward motion)
    # Otherwise check after: acc_z should become negative (lift is stopping/reversing)
    min_before_samples = 10  # Require at least 10 samples to check before
    min_after_samples = 10   # Require at least 10 samples to check after
    
    before_check_passed = False
    after_check_passed = False
    
    if longest_start >= min_before_samples:
        # We have enough space before the lift: check that acceleration wasn't deeply negative
        before_segment = acc_z[longest_start - min_before_samples:longest_start]
        before_mean = np.mean(before_segment)
        # Lift should not be preceded by sustained downward motion (acc_z > -0.2 g threshold)
        if before_mean > -0.2:
            before_check_passed = True
    
    if longest_end <= len(acc_z) - min_after_samples:
        # We have enough space after the lift: check that acceleration becomes negative (lift ends)
        after_segment = acc_z[longest_end:longest_end + min_after_samples]
        after_mean = np.mean(after_segment)
        # Lift should be followed by downward acceleration (acc_z < 0.0 g)
        if after_mean < 0.0:
            after_check_passed = True
    
    # Require at least one check to pass to validate this is a true lift
    if not (before_check_passed or after_check_passed):
        return 0.0
    
    # Score based on how well the segment matches the target
    # Favor higher meaningful_ratio (more time in 0.3-0.8 range)
    meaningful_score = min(1.0, meaningful_ratio / 0.5)  # 50% meaningful = good, 100% = excellent
    
    # Score based on duration: 20 samples = 0.5, 40+ samples = 1.0
    duration_score = min(1.0, (max_valid_length - min_samples_for_lift) / (40 - min_samples_for_lift))
    
    # Score based on consistency: low variance in valid segment
    acc_std = np.std(valid_segment)
    consistency_score = 1.0 - min(1.0, acc_std / 0.2)  # Threshold 0.2 g std
    
    # Score based on mean value: favor values in middle of range
    mean_acc = np.mean(valid_segment)
    target_acc = 0.4  # Slightly lower target to account for mix of high and low values
    deviation_from_target = abs(mean_acc - target_acc)
    mean_score = 1.0 - min(1.0, deviation_from_target / 0.2)
    
    # Combined score
    lift_score = (meaningful_score * 0.35 + duration_score * 0.25 + consistency_score * 0.25 + mean_score * 0.15)
    
    return min(1.0, max(0.0, lift_score))


def compute_idle_like_score(signal_dict):
    """
    Detects idle motion (minimal movement with low acceleration and angular velocity).
    A window is considered idle when accelerations and angular velocities are consistently low,
    with little to no spikes.
    
    Properties of idle-like motion:
    - Mean acceleration magnitude is close to 1.0 g (stationary on surface with gravity)
    - Mean angular velocity magnitude is low (< 50 deg/s)
    - Acceleration variance is low (consistent, no major spikes)
    - Angular velocity variance is low (consistent, no major spikes)
    - No sharp peaks in acceleration or angular velocity
    
    Args:
        signal_dict: signal dictionary with raw samples (from one window)
        
    Returns:
        float between 0-1: confidence score that motion is idle-like
    """
    # Check for required signals
    has_acc = 'acc_x' in signal_dict and 'acc_y' in signal_dict and 'acc_z' in signal_dict
    has_gyro = 'gyro_x' in signal_dict and 'gyro_y' in signal_dict and 'gyro_z' in signal_dict
    
    if not (has_acc and has_gyro):
        return 0.0
    
    # Compute acceleration magnitude and gyro magnitude
    acc_x = signal_dict['acc_x']
    acc_y = signal_dict['acc_y']
    acc_z = signal_dict['acc_z']
    gyro_x = signal_dict['gyro_x']
    gyro_y = signal_dict['gyro_y']
    gyro_z = signal_dict['gyro_z']
    
    acc_magnitude = np.sqrt(acc_x**2 + acc_y**2 + acc_z**2)
    gyro_magnitude = np.sqrt(gyro_x**2 + gyro_y**2 + gyro_z**2)
    
    # Criterion 1: Mean acceleration close to 1.0 g (stationary on surface)
    # For idle motion, the magnitude should be close to 1.0 g with little variation
    acc_mean = np.mean(acc_magnitude)
    acc_std = np.std(acc_magnitude)
    
    # Score for acceleration being close to 1.0 g (gravity)
    acc_deviation_from_1g = np.abs(acc_mean - 1.0)
    acc_mean_score = 1.0 - min(1.0, acc_deviation_from_1g / 0.2)  # Tolerance of 0.2 g
    
    # Score for low variance/std in acceleration
    acc_variance_score = 1.0 - min(1.0, acc_std / 0.15)  # Threshold 0.15 g std
    
    # Criterion 2: Low mean angular velocity
    gyro_mean = np.mean(gyro_magnitude)
    gyro_std = np.std(gyro_magnitude)
    
    # Score for low mean angular velocity (< 50 deg/s)
    gyro_mean_score = 1.0 - min(1.0, gyro_mean / 50.0)
    
    # Score for low variance/std in angular velocity
    gyro_variance_score = 1.0 - min(1.0, gyro_std / 50.0)
    
    # Criterion 3: No sharp spikes in acceleration
    acc_max = np.max(acc_magnitude)
    acc_peak_to_mean = (acc_max - acc_mean) / (acc_mean + 1e-6)
    acc_smoothness_score = 1.0 - min(1.0, acc_peak_to_mean / 2.0)  # Threshold: 2x mean
    
    # Criterion 4: No sharp spikes in angular velocity
    gyro_max = np.max(gyro_magnitude)
    gyro_peak_to_mean = (gyro_max - gyro_mean) / (gyro_mean + 1e-6)
    gyro_smoothness_score = 1.0 - min(1.0, gyro_peak_to_mean / 3.0)  # Threshold: 3x mean
    
    # Combined score: all criteria must be satisfied for high idle score
    # Equal weighting across all criteria
    idle_score = (
        acc_mean_score * 0.2 +
        acc_variance_score * 0.2 +
        gyro_mean_score * 0.2 +
        gyro_variance_score * 0.2 +
        acc_smoothness_score * 0.1 +
        gyro_smoothness_score * 0.1
    )
    
    return min(1.0, max(0.0, idle_score))


def extract_behavioral_features(signal_dict, fft_dict=None):
    """
    Extracts behavioral feature scores from raw signal and optional FFT data.
    Returns a dictionary with scores for shake-like, spin-like, roll-like, pitch-like, tap-like, drop-like, lift-like, and idle-like motions.
    
    Args:
        signal_dict: dictionary mapping signal names to numpy arrays (one window)
        fft_dict: optional dictionary of FFT magnitudes (if not provided, FFT is computed)
        
    Returns:
        Dictionary with keys: shake_like, spin_like, roll_like, pitch_like, tap_like, drop_like, lift_like, idle_like
    """
    behavioral_features = {}
    
    # Compute FFT if not provided
    if fft_dict is None:
        fft_dict = {}
        for key, signal in signal_dict.items():
            if key not in {'timestamp', 'label', 'markers'}:
                try:
                    fft_values = np.fft.fft(signal)
                    magnitude = np.abs(fft_values)
                    fft_dict[key] = magnitude
                except (TypeError, ValueError):
                    continue
        
        # Compute combined magnitudes for FFT
        if 'acc_x' in fft_dict and 'acc_y' in fft_dict and 'acc_z' in fft_dict:
            fft_dict['acc_xyz'] = np.sqrt(fft_dict['acc_x']**2 + fft_dict['acc_y']**2 + fft_dict['acc_z']**2)
    
    # Compute each behavioral score
    behavioral_features['shake_like'] = compute_shake_like_score(fft_dict)
    behavioral_features['spin_like'] = compute_spin_like_score(signal_dict)
    
    roll_score, pitch_score = compute_roll_pitch_like_scores(signal_dict)
    behavioral_features['roll_like'] = roll_score
    behavioral_features['pitch_like'] = pitch_score
    
    behavioral_features['tap_like'] = compute_tap_like_score(signal_dict)
    behavioral_features['drop_like'] = compute_drop_like_score(signal_dict)
    behavioral_features['lift_like'] = compute_lift_like_score(signal_dict)
    behavioral_features['idle_like'] = compute_idle_like_score(signal_dict)
    
    return behavioral_features


def extract_features_from_single_window_file(filename):
    """
    Extracts both time-domain and FFT features from a single CSV file (one window).
    Uses the same feature definitions as extract_time_domain_features and extract_fft_features.
    Also includes behavioral features (shake-like, spin-like, roll-like, pitch-like, tap-like, drop-like).
    
    Args:
        filename: path to the CSV file
        
    Returns:
        Dictionary containing combined time, frequency domain, and behavioral features
    """
    # Lazy import to avoid package-level circular imports with managers.__init__.
    from managers.csv_manager import CSVManager

    data = CSVManager.load_csv_file(filename)
    # Step 1: Unpack CSV data
    
    # Step 2: Convert data to numpy arrays for feature extraction
    signal_dict = {key: np.array([sample[key] for sample in data]) for key in data[0].keys()}
    
    # Add computed magnitudes to signal_dict for behavioral features
    if 'acc_x' in signal_dict and 'acc_y' in signal_dict and 'acc_z' in signal_dict:
        signal_dict['acc_xyz'] = np.sqrt(signal_dict['acc_x']**2 + signal_dict['acc_y']**2 + signal_dict['acc_z']**2)
    
    # Step 3: Extract time-domain features using the helper
    time_features = _extract_time_domain_features_from_signal_dict(signal_dict)
    
    # Step 4: Compute FFT and extract frequency-domain features using the helper
    fft_dict = {}
    for key, signal in signal_dict.items():
        if key not in {'timestamp', 'label', 'markers'}:
            try:
                fft_values = np.fft.fft(signal)
                magnitude = np.abs(fft_values)
                fft_dict[key] = magnitude
            except (TypeError, ValueError):
                continue
    
    freq_features = _extract_fft_features_from_signal_dict(fft_dict)
    
    # Step 5: Extract behavioral features
    behavioral_features = extract_behavioral_features(signal_dict, fft_dict)
    
    # Step 6: Combine all features
    combined_features = {**time_features, **freq_features, **behavioral_features}
    
    return combined_features