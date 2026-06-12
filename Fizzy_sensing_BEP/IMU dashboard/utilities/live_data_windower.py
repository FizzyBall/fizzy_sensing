import numpy as np
from utilities.data_windower import _extract_time_domain_features_from_signal_dict, _extract_fft_features_from_signal_dict

#Gets inputs as list of dictionaries, where each dict is one sample/row (e.g., [{'acc_x': 0.034, 'gyro_z': 0.0}, {...}])
#Made for live processing so each list of dictionaries is 128 samples for now.
#Applies hanning window to data. Since the list of dictionaries gets pushed every 64 samples, there is already overlap. 

def compute_fft(data):
    """
    Computes the FFT of the data.
    
    Args:
        data: dictionary of lists where each list corresponds to a key and contains the values for that key across all samples. For example, {'acc_x': [0.034, 0.035, ...], 'gyro_z': [0.0, 0.01, ...]}.
        
    Returns:
        dictionary of lists where each list corresponds to a key and contains the FFT magnitude for that key across all samples. For example, {'acc_x': [fft_value1, fft_value2, ...], 'gyro_z': [fft_value1, fft_value2, ...]}, .
    """
    data_length = len(next(iter(data.values()))) if data else 0
    
    # Create Hanning window
    hanning_window = np.hanning(data_length)
    fft_dict = {}
    magnitude_dict = {}
    hanning_windowed_dict = {}
    for key in data.keys():
        # Apply Hanning window to the data for the current key, subtracting the mean to remove DC components. 
        hanning_windowed_dict[key] = (np.array(data[key]) - np.mean(data[key])) * hanning_window
        fft_dict[key] = np.fft.rfft(hanning_windowed_dict[key])
        magnitude_dict[key] = np.abs(fft_dict[key])
    return magnitude_dict

def extract_frequency_domain_features(fft_magnitude_dict):
    """
    Extracts features from the FFT magnitude dictionary using canonical feature definitions.
    
    Args:
        fft_magnitude_dict: dictionary of lists where each list corresponds to a key and contains the FFT magnitude for that key across all samples.
    Returns:
        dictionary of features extracted from FFT data.
    """
    # Convert lists to numpy arrays
    magnitude_dict = {key: np.array(magnitude) for key, magnitude in fft_magnitude_dict.items()}
    # Exclude raw sensor components - we only want features from derived quantities (like magnitude signals)
    exclude_keys = {'acc_x', 'acc_y', 'acc_z', 'gyro_x', 'gyro_y', 'gyro_z'}
    # Use the helper function for canonical feature definitions
    features_dict = _extract_fft_features_from_signal_dict(magnitude_dict, exclude_keys=exclude_keys)
    return features_dict
def extract_time_domain_features(data):
    """
    Extracts time-domain features from the data using canonical feature definitions.
    
    Args:
        data: dictionary of lists where each list corresponds to a key and contains the values for that key across all samples.
        window_size: integer representing the size of the window for feature extraction (kept for compatibility).
    Returns:
        dictionary of features where each key is a feature name and the value is the computed feature value.
    """
    # Convert lists to numpy arrays
    signal_dict = {key: np.array(signal) for key, signal in data.items()}
    # Use the helper function for canonical feature definitions
    features_dict = _extract_time_domain_features_from_signal_dict(signal_dict)
    return features_dict

def extract_features(data):
    """
    Extracts both time-domain and frequency-domain features from the data.
    
    Args:
        data: dictionary of lists where each list corresponds to a key and contains the values for that key across all samples. For example, {'acc_x': [0.034, 0.035, ...], 'gyro_z': [0.0, 0.01, ...]}.
    """
    time_domain_features = extract_time_domain_features(data)
    fft_magnitude_dict = compute_fft(data)
    frequency_domain_features = extract_frequency_domain_features(fft_magnitude_dict)
    features_dict = {**frequency_domain_features, **time_domain_features}
    return features_dict
