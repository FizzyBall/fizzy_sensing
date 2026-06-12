"""settings.py — compatibility shim for the IMU dashboard's CNN tabs.

The CNN settings were split into two files in this folder:

  * settingsSingleModel.py     — single-model live UI (class_names, data_version,
                                  INPUT_AMOUNT, OUTPUT_AMOUNT, the *_LIVE dicts, …)
  * settingsHandGroundModel.py — hand/ground dashboard (CLASS_NAMES_GROUND/HAND,
                                  DATA_VERSION_GROUND/HAND, INPUT/OUTPUT_AMOUNT_*, …)

The dashboard's "Live Classify" and "Classification Analysis" tabs still do a
single ``import settings`` and expect every constant in one namespace, so this
module re-exports both source files. The panels/managers call
``importlib.reload(settings)`` before each model load to pick up on-disk edits;
we reload the two sources here too so edits to either file take effect without
restarting the dashboard.

Standalone scripts (LiveClassifier.py / LiveClassifyDashboard.py) import from the
two split files directly and do not use this shim.
"""

import importlib

import settingsSingleModel as _single
import settingsHandGroundModel as _handground

# Re-import the sources so edits on disk are picked up whenever this shim is
# (re)loaded. Reloading updates the module objects in place, so the star imports
# below then expose the fresh values.
_single = importlib.reload(_single)
_handground = importlib.reload(_handground)

# Order matters only for the one name both files define (CONFIDENCE_THRESHOLD);
# the hand/ground value wins, which is what the multi-model tabs expect.
from settingsSingleModel import *        # noqa: F401,F403,E402
from settingsHandGroundModel import *    # noqa: F401,F403,E402
