"""
Audio Manager Module

Handles sound tone generation and playback using winsound.
Provides countdown beeps, start beep, and stop notification beeps.
"""

import threading
import sys

# Try to import winsound only on Windows
try:
    if sys.platform == "win32":
        import winsound
    else:
        winsound = None
except ImportError:
    winsound = None

class AudioManager:
    """Manages audio playback for recording feedback using Windows beeps."""
    
    def __init__(self):
        """Initialize audio manager."""
        pass
    
    def play_countdown_beep(self):
        """Play a low beep for countdown (400 Hz, 100ms)."""
        threading.Thread(
            target=lambda: winsound.Beep(400, 100),
            daemon=True
        ).start()
    
    def play_start_beep(self):
        """Play a high beep for recording start (800 Hz, 200ms)."""
        threading.Thread(
            target=lambda: winsound.Beep(800, 200),
            daemon=True
        ).start()
    
    def play_stop_beeps(self):
        """Play double mid beeps for recording stop (600 Hz, 150ms each with 100ms gap)."""
        def play_sequence():
            winsound.Beep(600, 150)
            threading.Event().wait(0.05)  # 100ms gap
            winsound.Beep(600, 150)
        
        threading.Thread(target=play_sequence, daemon=True).start()
    
    def cleanup(self):
        """Cleanup audio resources."""
        pass
