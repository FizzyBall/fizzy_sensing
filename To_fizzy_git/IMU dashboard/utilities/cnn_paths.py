"""Locate the CNN code used by the dashboard's classification tabs.

The CNN lives under ``DataAndModels/CNN``. The wrapper, the ``settings`` module
and the ``CNN_fizzy*.py`` architecture files are split across that tree; this
module exposes their locations and puts the code + settings dirs on ``sys.path``
so ``import cnn_wrapper`` / ``import settings`` work no matter which folder the
dashboard is launched from.

Layout in DataAndModels/CNN::

    CNN/
        Mean, std/                 <- MEAN_STD_DIR  (imu_mean_v*.npy / imu_std_v*.npy)
        Models/                    <- *.pth model files
        Model Running/             <- CNN_SETTINGS_DIR
            settingsSingleModel.py
            settingsHandGroundModel.py
            Utilities/             <- CNN_CODE_DIR
                settings.py        <- CNN_SETTINGS_FILE (shim re-exporting both files)
                cnn_wrapper.py
                CNN_fizzy.py       <- architecture (normal)
                CNN_fizzy_hand.py  <- architecture (hand)
                CNN_fizzy_ground.py<- architecture (ground)
"""

from __future__ import annotations

import sys
from pathlib import Path

_DASHBOARD_DIR = Path(__file__).resolve().parent.parent  # "IMU dashboard"
_CNN_DIR = _DASHBOARD_DIR.parent / "DataAndModels" / "CNN"

#: Folder containing the ``settings`` module (and the two split settings files).
CNN_SETTINGS_DIR = _CNN_DIR / "Model Running"

#: Folder containing ``cnn_wrapper.py`` and the ``CNN_fizzy*.py`` architectures.
CNN_CODE_DIR = CNN_SETTINGS_DIR / "Utilities"

#: The unified ``settings`` shim (re-exports the two split settings files). Lives
#: in the Utilities folder alongside the code that imports it.
CNN_SETTINGS_FILE = CNN_CODE_DIR / "settings.py"

#: Architecture files used to rebuild the IMUNet for each CNN mode.
ARCHITECTURE_FILES = {
    "normal": CNN_CODE_DIR / "CNN_fizzy.py",
    "hand":   CNN_CODE_DIR / "CNN_fizzy_hand.py",
    "ground": CNN_CODE_DIR / "CNN_fizzy_ground.py",
}

#: Folder containing the trained CNN ``*.pth`` weight files.
MODELS_DIR = _CNN_DIR / "Models"

#: Default folder to browse for normalization mean/std .npy files.
MEAN_STD_DIR = _CNN_DIR / "Mean, std"


def ensure_on_path() -> None:
    """Put the CNN code + settings dirs on ``sys.path`` (idempotent)."""
    for directory in (str(CNN_SETTINGS_DIR), str(CNN_CODE_DIR)):
        if directory not in sys.path:
            sys.path.insert(0, directory)


# Make `import cnn_wrapper` / `import settings` work as soon as this module loads.
ensure_on_path()
