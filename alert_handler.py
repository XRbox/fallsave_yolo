"""
FallSave Step 5.2: Fall Alert Sound and Snapshot Handler

Provides a standalone, modular AlertHandler for:
- Playing alert sound (asynchronously in background thread to avoid video frame drops)
- Capturing and saving frame snapshots to output/alerts/ with timestamped filenames
- Enforcing strict duplicate suppression (only once per confirmed fall event)
- Graceful error handling for missing/corrupted frames or sound device failures
- Full mockability for unit testing without physical webcam or speakers
"""

import os
import sys
import time
import logging
import threading
from datetime import datetime
from pathlib import Path
from typing import Optional, Tuple, Dict, Any, Callable
import cv2
import numpy as np

logger = logging.getLogger(__name__)


def default_sound_player() -> None:
    """
    Default cross-platform emergency alert sound generator.
    On Windows: uses winsound for a dual-tone emergency alert beep.
    Fallback: writes terminal bell / log.
    """
    try:
        if sys.platform == "win32":
            import winsound
            # High-attention dual emergency beep
            winsound.Beep(1500, 250)
            winsound.Beep(2000, 400)
        else:
            sys.stdout.write("\a")
            sys.stdout.flush()
    except Exception as e:
        logger.warning(f"[AlertHandler] Audio playback failed: {e}")


class AlertHandler:
    """
    Handles fall alert events: audio alert and timestamped snapshot recording.
    
    Guarantees:
    - Thread-safe, non-blocking audio (runs in daemon thread by default)
    - Exactly-once trigger per confirmed fall event (duplicate suppression)
    - Safe error handling for missing, None, or empty image frames
    - Automatic creation of dedicated alert directory
    """

    def __init__(
        self,
        output_dir: str = "output/alerts",
        enable_sound: bool = True,
        async_sound: bool = True,
        custom_sound_player: Optional[Callable[[], None]] = None
    ):
        """
        Initializes the AlertHandler.
        
        Args:
            output_dir: Folder to store fall snapshot images.
            enable_sound: Whether audio alerts are enabled.
            async_sound: Whether sound plays in background daemon thread (recommended).
            custom_sound_player: Optional callable for testing or custom audio logic.
        """
        self.output_dir = Path(output_dir)
        self.enable_sound = enable_sound
        self.async_sound = async_sound
        self.sound_player = custom_sound_player or default_sound_player

        # Internal state tracking
        self._is_alert_active: bool = False
        self._alert_count: int = 0
        self._last_alert_time: Optional[float] = None
        self._last_snapshot_path: Optional[str] = None
        self._lock = threading.Lock()

        # Ensure directory exists on startup
        try:
            self.output_dir.mkdir(parents=True, exist_ok=True)
        except Exception as e:
            logger.warning(f"[AlertHandler] Failed to create output directory {self.output_dir}: {e}")

    @property
    def is_alert_active(self) -> bool:
        """Returns True if an alert has fired and has not yet been reset."""
        return self._is_alert_active

    @property
    def alert_count(self) -> int:
        """Total number of confirmed fall alerts triggered."""
        return self._alert_count

    @property
    def last_snapshot_path(self) -> Optional[str]:
        """Path to the most recently saved snapshot image."""
        return self._last_snapshot_path

    def _play_sound_safely(self) -> None:
        """Safely invokes the sound player without crashing on error."""
        try:
            self.sound_player()
        except Exception as e:
            logger.warning(f"[AlertHandler] Sound player raised an exception: {e}")

    def save_snapshot(
        self,
        frame: Optional[np.ndarray],
        timestamp_str: Optional[str] = None
    ) -> Tuple[bool, Optional[str]]:
        """
        Saves a snapshot of the frame to the output directory.
        
        Args:
            frame: OpenCV BGR image array.
            timestamp_str: Optional custom timestamp string for filename.
            
        Returns:
            Tuple of (success: bool, saved_filepath: Optional[str])
        """
        if frame is None:
            logger.warning("[AlertHandler] Cannot save snapshot: frame is None.")
            return False, None

        if not isinstance(frame, np.ndarray) or frame.size == 0:
            logger.warning("[AlertHandler] Cannot save snapshot: invalid or empty frame array.")
            return False, None

        try:
            self.output_dir.mkdir(parents=True, exist_ok=True)
            
            if timestamp_str is None:
                timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
                
            filename = f"fall_alert_{timestamp_str}.jpg"
            filepath = self.output_dir / filename

            # Deduplicate filename if multiple snapshots occur in the same second
            counter = 1
            while filepath.exists():
                ms_str = datetime.now().strftime("%f")[:3]
                filename = f"fall_alert_{timestamp_str}_{ms_str}_{counter}.jpg"
                filepath = self.output_dir / filename
                counter += 1

            success = cv2.imwrite(str(filepath), frame)
            if not success:
                logger.warning(f"[AlertHandler] cv2.imwrite failed to write to: {filepath}")
                return False, None

            self._last_snapshot_path = str(filepath)
            logger.info(f"[AlertHandler] Snapshot saved: {filepath}")
            return True, str(filepath)

        except Exception as e:
            logger.warning(f"[AlertHandler] Error saving snapshot: {e}")
            return False, None

    def trigger_alert(
        self,
        frame: Optional[np.ndarray] = None,
        timestamp: Optional[float] = None
    ) -> Dict[str, Any]:
        """
        Triggers the fall alert: plays sound and saves snapshot.
        Enforces duplicate suppression: executes only once per confirmed fall event.
        
        Args:
            frame: Current webcam / video frame to save as snapshot.
            timestamp: Monotonic or wall-clock timestamp of the event.
            
        Returns:
            Dictionary containing trigger status, paths, and metadata.
        """
        with self._lock:
            if self._is_alert_active:
                logger.debug("[AlertHandler] Duplicate alert suppressed. Alert is already active.")
                return {
                    "triggered": False,
                    "reason": "duplicate_suppressed",
                    "alert_count": self._alert_count,
                    "snapshot_path": self._last_snapshot_path
                }

            self._is_alert_active = True
            self._alert_count += 1
            self._last_alert_time = timestamp or time.time()

        # Play sound (async or sync)
        sound_played = False
        if self.enable_sound:
            if self.async_sound:
                threading.Thread(target=self._play_sound_safely, daemon=True).start()
            else:
                self._play_sound_safely()
            sound_played = True

        # Save snapshot
        snapshot_saved, snapshot_path = self.save_snapshot(frame)

        return {
            "triggered": True,
            "alert_count": self._alert_count,
            "sound_played": sound_played,
            "snapshot_saved": snapshot_saved,
            "snapshot_path": snapshot_path,
            "timestamp": self._last_alert_time
        }

    def reset(self) -> None:
        """
        Resets the active alert status so future fall events can trigger anew.
        Called when the state machine recovers back to NORMAL.
        """
        with self._lock:
            self._is_alert_active = False
            logger.debug("[AlertHandler] Alert state reset to inactive.")
