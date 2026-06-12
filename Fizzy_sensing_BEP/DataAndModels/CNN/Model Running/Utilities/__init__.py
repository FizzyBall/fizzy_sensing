"""
Utilities package for the CNN_final live-classification scripts.

Bundles the model definitions and hardware/data helpers used by the dashboards
in the parent 'Model Running' folder:

    - CNN_fizzy            : single-model IMUNet (settings.INPUT_AMOUNT / OUTPUT_AMOUNT)
    - CNN_fizzy_ground     : ground-model IMUNet (settings.*_GROUND, POOL_AMOUNT)
    - CNN_fizzy_hand       : in-hand-model IMUNet (settings.*_HAND)
    - Fizzy_CNNudp         : Fizzy UDP communication class
    - imu_CNN_data_extractor : IMU packet -> feature extraction helpers

The model modules read their configuration from the 'settings.py' that lives in
the parent 'Model Running' folder; each adds that folder to sys.path on import,
so they work regardless of the current working directory.
"""

from .CNN_fizzy import IMUNet
from .CNN_fizzy_ground import IMUNet as IMUNetGround
from .CNN_fizzy_hand import IMUNet as IMUNetHand
from .Fizzy_CNNudp import Fizzy
from .imu_CNN_data_extractor import (
    extract_linear_acceleration_from_packet,
    extract_angular_velocity_from_packet,
)

__all__ = [
    "IMUNet",
    "IMUNetGround",
    "IMUNetHand",
    "Fizzy",
    "extract_linear_acceleration_from_packet",
    "extract_angular_velocity_from_packet",
]
