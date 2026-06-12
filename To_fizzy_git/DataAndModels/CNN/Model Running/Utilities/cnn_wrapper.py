"""
CNN Model Wrapper - Makes PyTorch CNN models compatible with scikit-learn interface.

Wraps trained CNN models so they can be used in the classification analysis workflow
alongside Random Forest and other scikit-learn models.
"""

import os
import sys
import importlib.util
import logging
import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path

logger = logging.getLogger(__name__)


class CNNModelWrapper:
    """Wraps a PyTorch CNN model to provide scikit-learn-like predict() interface."""
    
    def __init__(self, model, class_names, mean, std, window_size=64, input_channels=6):
        """
        Initialize CNN wrapper.
        
        Args:
            model: Loaded PyTorch IMUNet model
            class_names: List of class names
            mean: Normalization mean (shape: input_channels or input_channels x 1)
            std: Normalization std (shape: input_channels or input_channels x 1)
            window_size: Expected window size in samples (default 64)
            input_channels: Number of input channels (default 6, can be 7)
        """
        self.model = model
        self.model.eval()
        self.class_names = class_names
        self.window_size = window_size
        self.input_channels = input_channels
        self.confidence_threshold = 0.5
        
        # Ensure mean and std are 1D numpy arrays
        self.mean = np.array(mean).flatten().astype(np.float32)
        self.std = np.array(std).flatten().astype(np.float32)
        
        # Trim or pad normalization stats to match input_channels
        if self.mean.shape[0] != input_channels:
            if self.mean.shape[0] > input_channels:
                # Trim to match
                logger.warning(
                    f"WARNING: Normalization stats have {self.mean.shape[0]} channels but model expects "
                    f"{input_channels}. Trimming to first {input_channels} channels. "
                    f"This assumes channels are in order [acc_x, acc_y, acc_z, gyro_x, gyro_y, gyro_z, ...]. "
                    f"Predictions may be inaccurate if channel order differs!"
                )
                self.mean = self.mean[:input_channels]
                self.std = self.std[:input_channels]
            else:
                # Pad to match
                logger.warning(
                    f"WARNING: Normalization stats have {self.mean.shape[0]} channels but model expects "
                    f"{input_channels}. Padding with zeros (mean) and ones (std) for {input_channels - self.mean.shape[0]} "
                    f"extra channels. This is a fallback - predictions may be inaccurate!"
                )
                self.mean = np.pad(self.mean, (0, input_channels - self.mean.shape[0]), 
                                  mode='constant', constant_values=0.0)
                self.std = np.pad(self.std, (0, input_channels - self.std.shape[0]), 
                                 mode='constant', constant_values=1.0)
    
    def predict(self, X):
        """
        Predict class labels for windows.
        
        Args:
            X: Input data, shape (n_windows, window_size, input_channels)
               or (n_windows, input_channels, window_size) if channels-first
               
        Returns:
            Array of predicted class indices, shape (n_windows,)
        """
        probas = self.predict_proba(X)
        return np.argmax(probas, axis=1)
    
    def predict_proba(self, X):
        """
        Predict class probabilities for windows.
        
        Args:
            X: Input data, shape (n_windows, window_size, input_channels)
               or (n_windows, input_channels, window_size)
               
        Returns:
            Array of class probabilities, shape (n_windows, n_classes)
        """
        X = np.asarray(X, dtype=np.float32)
        
        # Handle both (n, seq_len, channels) and (n, channels, seq_len) formats
        if X.ndim == 2:
            # Single window (window_size, input_channels) -> reshape
            X = X[np.newaxis, :, :]
        
        if X.shape[-1] == self.input_channels and X.shape[1] != self.input_channels:
            # Shape is (n_windows, window_size, input_channels) -> transpose to (n, channels, window_size)
            X = X.transpose(0, 2, 1)
        elif X.shape[1] != self.input_channels:
            # Unknown shape, try to infer
            if X.shape[-1] == self.input_channels:
                X = X.transpose(0, 2, 1)
            else:
                raise ValueError(
                    f"Cannot infer shape: got {X.shape}, expected "
                    f"(n_windows, {self.window_size}, {self.input_channels}) or "
                    f"(n_windows, {self.input_channels}, {self.window_size})"
                )
        
        # Normalize using precomputed mean/std
        # X shape: (n_windows, input_channels, window_size)
        mean_2d = self.mean[:, np.newaxis]  # (input_channels, 1)
        std_2d = self.std[:, np.newaxis]    # (input_channels, 1)
        X = (X - mean_2d) / (std_2d + 1e-8)
        
        # Convert to PyTorch and predict
        X_torch = torch.from_numpy(X).float()
        
        with torch.no_grad():
            logits = self.model(X_torch)
            probs = F.softmax(logits, dim=1).cpu().numpy()
        
        return probs
    
    def predict_with_confidence(self, X, threshold=None):
        """
        Predict with confidence filtering.
        
        Args:
            X: Input data as above
            threshold: Confidence threshold. If None, uses self.confidence_threshold.
                      Returns -1 for low-confidence predictions.
                      
        Returns:
            Tuple of (predicted_labels, confidences)
            predicted_labels: Class indices, or -1 for "unsure"
            confidences: Confidence scores for the predicted class
        """
        if threshold is None:
            threshold = self.confidence_threshold
            
        probs = self.predict_proba(X)
        pred_labels = np.argmax(probs, axis=1)
        confidences = np.max(probs, axis=1)
        
        # Mark low-confidence predictions as -1 (unsure)
        pred_labels = pred_labels.astype(np.float32)  # Allow -1.0
        pred_labels[confidences < threshold] = -1.0
        
        return pred_labels, confidences
    
    def get_class_names(self):
        """Get list of class names."""
        return self.class_names
    
    def get_n_classes(self):
        """Get number of classes."""
        return len(self.class_names)
    
    @classmethod
    def from_model_file(cls, model_path, data_version='7', model_number=None, device='cpu',
                        class_names=None, mean_path=None, std_path=None, architecture_file=None):
        """
        Load a CNN model from disk.
        
        Args:
            model_path: Path to the trained model file (.pth)
            data_version: Version of normalization statistics (e.g., '7', '10')
            model_number: Model number (used if model_path is just a directory)
            device: Device to load model on ('cpu' or 'cuda')
            
        Returns:
            CNNModelWrapper instance
        """
        model_path = Path(model_path)
        
        # If it's a directory, construct model filename
        if model_path.is_dir():
            if model_number is None:
                # Try to find any .pth file in the directory
                pth_files = list(model_path.glob("*.pth"))
                if not pth_files:
                    raise FileNotFoundError(f"No .pth files found in {model_path}")
                model_path = pth_files[0]
            else:
                model_path = model_path / f"Trained_{model_number}.pth"
        
        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        # Load the state dict first to infer architecture
        state_dict = torch.load(str(model_path), map_location=device)
        
        # Infer input and output channels from the checkpoint
        conv1_weight = state_dict.get('conv1.weight')
        fc_weight = state_dict.get('fc.weight')
        
        if conv1_weight is None or fc_weight is None:
            raise ValueError("Invalid model checkpoint: missing conv1.weight or fc.weight")
        
        input_channels = conv1_weight.shape[1]  # (out_channels, in_channels, kernel_size)
        output_channels = fc_weight.shape[0]    # (out_channels, in_channels)
        fc_input_size = fc_weight.shape[1]      # Input features to FC layer (infer from checkpoint)
        
        # Load model architecture
        IMUNet = None
        model_dir = model_path.parent

        # Use the explicitly provided architecture file when given (preferred path)
        if architecture_file is not None:
            arch_path = Path(architecture_file)
            arch_dir = str(arch_path.parent)
            sys.path.insert(0, arch_dir)
            try:
                spec = importlib.util.spec_from_file_location("model_module", arch_path)
                if spec and spec.loader:
                    module = importlib.util.module_from_spec(spec)
                    spec.loader.exec_module(module)
                    if hasattr(module, 'IMUNet'):
                        IMUNet = module.IMUNet
                    else:
                        raise ValueError(f"No IMUNet class found in {arch_path}")
            finally:
                if arch_dir in sys.path:
                    sys.path.remove(arch_dir)
        
        # Define fallback dynamic class with adaptive pooling
        class DynamicIMUNet(torch.nn.Module):
            def __init__(self, input_channels, output_channels, fc_input_size=32):
                super().__init__()
                self.fc_input_size = fc_input_size
                self._adaptive_pooling_warned = False
                
                self.conv1 = torch.nn.Conv1d(input_channels, 16, kernel_size=7, padding=3)
                self.bn1 = torch.nn.BatchNorm1d(16)
                self.conv2 = torch.nn.Conv1d(16, 32, kernel_size=5, padding=2)
                self.bn2 = torch.nn.BatchNorm1d(32)
                self.conv3 = torch.nn.Conv1d(32, 32, kernel_size=3, padding=1)
                self.bn3 = torch.nn.BatchNorm1d(32)
                
                # Calculate pooling strategy to match expected fc_input_size
                # Assuming 64 input samples and 32 final conv channels
                expected_samples = fc_input_size // 32
                
                # Try to infer pooling: if fc_input_size is 256, we need 8 samples
                # If 128, we need 4 samples, etc.
                # Pattern: Pool1(4) -> Pool2(k) -> Pool3(m) where (64 / 4 / k / m) * 32 = fc_input_size
                if expected_samples == 1:  # Standard: 64 / 4 / 4 / 4 = 1
                    self.pool_strategy = "standard"  # (4, 4, 4)
                elif expected_samples == 8:  # 64 / 4 / 2 / 1 = 8
                    self.pool_strategy = "alt1"  # (4, 2, 1)
                elif expected_samples == 4:  # 64 / 4 / 4 / 1 = 4
                    self.pool_strategy = "alt2"  # (4, 4, 1)
                elif expected_samples == 16:  # 64 / 2 / 2 / 1 = 16
                    self.pool_strategy = "alt3"  # (2, 2, 1)
                else:
                    # Default to standard
                    self.pool_strategy = "standard"
                
                self.fc = torch.nn.Linear(fc_input_size, output_channels)

            def forward(self, x):
                x = torch.nn.functional.relu(self.bn1(self.conv1(x)))
                
                # First pooling
                x = torch.nn.functional.max_pool1d(x, 4)
                
                x = torch.nn.functional.relu(self.bn2(self.conv2(x)))
                
                # Second pooling (adaptive based on strategy)
                if self.pool_strategy in ["standard", "alt2"]:
                    x = torch.nn.functional.max_pool1d(x, 4)
                else:  # alt1, alt3, etc.
                    x = torch.nn.functional.max_pool1d(x, 2)
                
                x = torch.nn.functional.relu(self.bn3(self.conv3(x)))
                
                # Third pooling (only for standard)
                if self.pool_strategy == "standard":
                    x = torch.nn.functional.avg_pool1d(x, 4)
                # else: no third pooling for alt strategies
                
                # Use adaptive pooling to ensure we get the right fc_input_size
                # regardless of the exact pooling strategy used during training
                target_samples = self.fc_input_size // 32
                x_shape_before_adaptive = x.shape[2]
                x = torch.nn.functional.adaptive_avg_pool1d(x, target_samples)
                
                # Warn if adaptive pooling actually changed the shape
                if x_shape_before_adaptive != target_samples and not self._adaptive_pooling_warned:
                    logger.warning(
                        f"WARNING: Using adaptive pooling to reshape features from {x_shape_before_adaptive} "
                        f"to {target_samples} samples. The detected pooling strategy ({self.pool_strategy}) "
                        f"may not match training architecture. Predictions may be inaccurate!"
                    )
                    self._adaptive_pooling_warned = True
                
                x = torch.flatten(x, 1)
                x = self.fc(x)
                return x
        
        # Create model instance
        net = None
        if IMUNet is not None:
            try:
                # Try with input/output channels and fc_input_size
                net = IMUNet(input_channels, output_channels, fc_input_size)
            except TypeError:
                try:
                    # Try with just input/output channels
                    net = IMUNet(input_channels, output_channels)
                except TypeError:
                    # Try without arguments
                    try:
                        net = IMUNet()
                    except Exception:
                        net = None
        
        # Use dynamic fallback if needed (with correct FC input size)
        if net is None:
            net = DynamicIMUNet(input_channels, output_channels, fc_input_size)
        
        # Load state dict
        net.load_state_dict(state_dict)
        
        # Load normalization statistics
        mean = None
        std = None
        found_version = data_version
        searched_versions = []

        # Use explicitly provided mean/std paths if given (skip auto-search).
        # Flatten on load: the files are saved as (1, C, 1), so without flattening
        # the channel-count check below reads shape[0]==1 and spuriously pads/trims.
        if mean_path is not None and std_path is not None:
            mean = np.load(str(mean_path)).astype(np.float32).flatten()
            std = np.load(str(std_path)).astype(np.float32).flatten()

        if mean is None:
            # Try to find normalization files, trying multiple versions if needed
            search_versions = [data_version]
            for v in ['7', '10', '8', '6', '4']:
                if v not in search_versions:
                    search_versions.append(v)
            searched_versions = search_versions

            for try_version in search_versions:
                _mp = None
                _sp = None

                # Same directory as model
                candidate_mean = model_dir / "Mean, std" / f"imu_mean_v{try_version}.npy"
                candidate_std = model_dir / "Mean, std" / f"imu_std_v{try_version}.npy"
                if candidate_mean.exists() and candidate_std.exists():
                    _mp, _sp = candidate_mean, candidate_std

                # Parent directory (e.g. CNN/Mean, std when model is in CNN/Models)
                if _mp is None:
                    candidate_mean = model_dir.parent / "Mean, std" / f"imu_mean_v{try_version}.npy"
                    candidate_std = model_dir.parent / "Mean, std" / f"imu_std_v{try_version}.npy"
                    if candidate_mean.exists() and candidate_std.exists():
                        _mp, _sp = candidate_mean, candidate_std

                # Fallback: no version suffix (same dir then parent)
                if _mp is None and try_version == data_version:
                    for base_dir in [model_dir, model_dir.parent]:
                        cm = base_dir / "Mean, std" / "imu_mean.npy"
                        cs = base_dir / "Mean, std" / "imu_std.npy"
                        if cm.exists() and cs.exists():
                            _mp, _sp = cm, cs
                            break

                if _mp is not None:
                    mean_temp = np.load(str(_mp)).astype(np.float32).flatten()
                    std_temp = np.load(str(_sp)).astype(np.float32).flatten()
                    if mean_temp.shape[0] == input_channels:
                        mean = mean_temp
                        std = std_temp
                        found_version = try_version
                        if try_version != data_version:
                            logger.warning(
                                f"WARNING: Requested data version {data_version} not found. "
                                f"Using version {try_version} instead."
                            )
                        break
                    else:
                        logger.debug(f"Found normalization v{try_version} but flattened shape {mean_temp.shape[0]} != {input_channels}")

            # Last resort: any normalization file, pad/trim to fit
            if mean is None:
                logger.warning(f"WARNING: Searching for any compatible normalization file...")
                for search_dir in [model_dir / "Mean, std", model_dir.parent / "Mean, std"]:
                    if search_dir.exists():
                        mean_files = sorted(search_dir.glob("imu_mean_v*.npy")) or sorted(search_dir.glob("imu_mean.npy"))
                        if mean_files:
                            _mp = mean_files[0]
                            _sp = _mp.parent / _mp.name.replace("mean", "std")
                            if _sp.exists():
                                mean = np.load(str(_mp)).astype(np.float32).flatten()
                                std = np.load(str(_sp)).astype(np.float32).flatten()
                                logger.warning(f"Using fallback normalization: {_mp.name} (shape: {mean.shape})")
                                break

        # Pad or trim normalization stats if needed
        if mean is not None and std is not None:
            if mean.shape[0] < input_channels:
                logger.warning(
                    f"WARNING: Normalization has {mean.shape[0]} channels but model needs {input_channels}. Padding."
                )
                mean = np.pad(mean.flatten(), (0, input_channels - mean.shape[0]), mode='constant', constant_values=0.0)
                std = np.pad(std.flatten(), (0, input_channels - std.shape[0]), mode='constant', constant_values=1.0)
            elif mean.shape[0] > input_channels:
                logger.warning(
                    f"WARNING: Normalization has {mean.shape[0]} channels but model needs {input_channels}. Trimming."
                )
                mean = mean[:input_channels].flatten()
                std = std[:input_channels].flatten()

        if mean is None or std is None:
            raise FileNotFoundError(
                f"Normalization files not found for {input_channels} channels. Searched versions: {searched_versions}"
            )

        if class_names is None:
            raise ValueError(
                "class_names must be provided explicitly. Pass class_names matching the CNN mode "
                "(e.g. ['idle','shake','tap','drop','lift','kick'] for normal, "
                "['idle','tap','shake','drop'] for hand, ['idle','lift','kick'] for ground)."
            )
        
        return cls(
            model=net,
            class_names=class_names,
            mean=mean,
            std=std,
            window_size=64,
            input_channels=input_channels,
        )
    
    def __repr__(self):
        return (
            f"CNNModelWrapper("
            f"classes={self.class_names}, "
            f"input_channels={self.input_channels}, "
            f"window_size={self.window_size})"
        )
