"""
FallSave Step 5.2: Unit Tests for Fall Alert Sound and Snapshot Handler

Tests:
1. Alert trigger calls sound player and saves snapshot
2. Duplicate prevention suppresses repeated triggers for the same event
3. Reset re-arms the handler for subsequent fall events
4. Sound disabled flag correctly skips audio playback
5. Missing frame (None) is handled gracefully without crashing
6. Empty or invalid image arrays are handled gracefully
7. cv2.imwrite failure or exception is handled gracefully
8. Sound player exceptions do not crash the pipeline
9. Snapshot folder auto-creation
10. Integration test with FallStateMachine lifecycle
"""

import os
import shutil
import tempfile
import unittest
from unittest.mock import MagicMock, patch
import numpy as np
import cv2

from alert_handler import AlertHandler
from fall_state_machine import FallStateMachine, StateMachineConfig, State


class TestAlertHandler(unittest.TestCase):

    def setUp(self):
        # Create a clean temporary directory for snapshots
        self.test_dir = tempfile.mkdtemp()
        self.mock_sound = MagicMock()
        self.handler = AlertHandler(
            output_dir=self.test_dir,
            enable_sound=True,
            async_sound=False,  # Synchronous for deterministic unit tests
            custom_sound_player=self.mock_sound
        )
        # Dummy frame (100x100 BGR)
        self.dummy_frame = np.full((100, 100, 3), 128, dtype=np.uint8)

    def tearDown(self):
        # Clean up temporary directory
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)

    def test_alert_trigger_sound_and_snapshot(self):
        """Verifies sound is played and snapshot file is written on alert."""
        result = self.handler.trigger_alert(frame=self.dummy_frame, timestamp=10.0)

        self.assertTrue(result["triggered"])
        self.assertEqual(result["alert_count"], 1)
        self.assertTrue(result["sound_played"])
        self.assertTrue(result["snapshot_saved"])
        self.assertIsNotNone(result["snapshot_path"])
        self.assertTrue(os.path.exists(result["snapshot_path"]))
        self.mock_sound.assert_called_once()
        self.assertTrue(self.handler.is_alert_active)

    def test_duplicate_prevention_same_event(self):
        """Consecutive calls during the same fall event must NOT re-play sound or write snapshots."""
        res1 = self.handler.trigger_alert(frame=self.dummy_frame, timestamp=10.0)
        self.assertTrue(res1["triggered"])
        self.mock_sound.assert_called_once()

        # Simulate 10 subsequent frames in FALL_CONFIRMED
        for frame_t in range(11, 21):
            res_dup = self.handler.trigger_alert(frame=self.dummy_frame, timestamp=float(frame_t))
            self.assertFalse(res_dup["triggered"])
            self.assertEqual(res_dup["reason"], "duplicate_suppressed")

        # Sound and snapshot count must remain exactly 1
        self.assertEqual(self.mock_sound.call_count, 1)
        self.assertEqual(self.handler.alert_count, 1)
        saved_files = os.listdir(self.test_dir)
        self.assertEqual(len(saved_files), 1)

    def test_reset_allows_new_alert(self):
        """After resetting (recovery/standup), a new confirmed fall can trigger again."""
        self.handler.trigger_alert(frame=self.dummy_frame, timestamp=1.0)
        self.assertEqual(self.handler.alert_count, 1)
        self.mock_sound.assert_called_once()

        # Reset handler (e.g. state machine returned to NORMAL)
        self.handler.reset()
        self.assertFalse(self.handler.is_alert_active)

        # Trigger second alert
        res2 = self.handler.trigger_alert(frame=self.dummy_frame, timestamp=20.0)
        self.assertTrue(res2["triggered"])
        self.assertEqual(self.handler.alert_count, 2)
        self.assertEqual(self.mock_sound.call_count, 2)
        self.assertEqual(len(os.listdir(self.test_dir)), 2)

    def test_sound_disabled_config(self):
        """When enable_sound=False, snapshot is saved but audio player is never invoked."""
        quiet_handler = AlertHandler(
            output_dir=self.test_dir,
            enable_sound=False,
            async_sound=False,
            custom_sound_player=self.mock_sound
        )
        res = quiet_handler.trigger_alert(frame=self.dummy_frame, timestamp=5.0)

        self.assertTrue(res["triggered"])
        self.assertFalse(res["sound_played"])
        self.assertTrue(res["snapshot_saved"])
        self.mock_sound.assert_not_called()

    def test_missing_frame_graceful_handling(self):
        """When frame is None, sound still plays and function returns safely without crash."""
        res = self.handler.trigger_alert(frame=None, timestamp=5.0)

        self.assertTrue(res["triggered"])
        self.assertTrue(res["sound_played"])
        self.assertFalse(res["snapshot_saved"])
        self.assertIsNone(res["snapshot_path"])
        self.mock_sound.assert_called_once()

    def test_empty_or_invalid_frame_graceful_handling(self):
        """Empty or non-array frames must fail gracefully without raising exceptions."""
        empty_frame = np.zeros((0, 0, 3), dtype=np.uint8)
        res = self.handler.trigger_alert(frame=empty_frame, timestamp=5.0)

        self.assertTrue(res["triggered"])
        self.assertFalse(res["snapshot_saved"])
        self.assertIsNone(res["snapshot_path"])

    @patch("cv2.imwrite", return_value=False)
    def test_cv2_imwrite_failure_handling(self, mock_imwrite):
        """If cv2.imwrite fails (returns False), handler logs warning and does not crash."""
        res = self.handler.trigger_alert(frame=self.dummy_frame, timestamp=5.0)

        self.assertTrue(res["triggered"])
        self.assertFalse(res["snapshot_saved"])
        self.assertIsNone(res["snapshot_path"])

    def test_sound_player_exception_does_not_crash(self):
        """If sound player throws an unexpected exception, snapshot is still saved cleanly."""
        buggy_sound = MagicMock(side_effect=RuntimeError("Audio device missing"))
        resilient_handler = AlertHandler(
            output_dir=self.test_dir,
            enable_sound=True,
            async_sound=False,
            custom_sound_player=buggy_sound
        )

        res = resilient_handler.trigger_alert(frame=self.dummy_frame, timestamp=5.0)
        self.assertTrue(res["triggered"])
        self.assertTrue(res["snapshot_saved"])
        self.assertTrue(os.path.exists(res["snapshot_path"]))

    def test_directory_auto_creation(self):
        """If the target folder does not exist, it is created automatically."""
        nested_dir = os.path.join(self.test_dir, "nested", "sub_alerts")
        handler = AlertHandler(
            output_dir=nested_dir,
            enable_sound=False,
            async_sound=False
        )
        self.assertTrue(os.path.exists(nested_dir))

    def test_integration_with_state_machine(self):
        """End-to-end integration: StateMachine driving AlertHandler through full fall cycle."""
        fsm = FallStateMachine(StateMachineConfig(ground_confirm_time=2.0))
        t = 0.0

        # 1. Normal state: tracked person
        for _ in range(10):
            out = fsm.update(timestamp=t, p_normal=0.9, p_falling=0.05, p_lying=0.05, person_detected=True)
            self.assertEqual(out.state, State.NORMAL)
            t += 0.033

        # 2. Fall happens
        out = fsm.update(timestamp=t, p_normal=0.05, p_falling=0.90, p_lying=0.05, person_detected=True)
        self.assertEqual(out.state, State.FALLING_SUSPECTED)
        t += 0.033

        # 3. Ground check entered
        out = fsm.update(timestamp=t, p_normal=0.05, p_falling=0.05, p_lying=0.90, person_detected=True)
        self.assertEqual(out.state, State.GROUND_CHECK)
        t += 0.033

        # 4. Sustained lying until ALERT_TRIGGERED
        alert_event_fired = False
        while t < 3.0:
            out = fsm.update(timestamp=t, p_normal=0.05, p_falling=0.05, p_lying=0.90, person_detected=True)
            if out.event == "ALERT_TRIGGERED":
                alert_event_fired = True
                self.handler.trigger_alert(frame=self.dummy_frame, timestamp=t)
            elif out.state == State.FALL_CONFIRMED:
                # Even if user loop redundantly invokes trigger_alert in FALL_CONFIRMED:
                self.handler.trigger_alert(frame=self.dummy_frame, timestamp=t)
            t += 0.033

        self.assertTrue(alert_event_fired)
        self.assertEqual(fsm.current_state, State.FALL_CONFIRMED)
        self.assertEqual(self.handler.alert_count, 1)
        self.assertEqual(self.mock_sound.call_count, 1)

        # 5. Recovery standup: post-fall recovery resets both
        for _ in range(160):  # > 5.0 seconds of sustained normal
            out = fsm.update(timestamp=t, p_normal=0.95, p_falling=0.02, p_lying=0.03, person_detected=True)
            if out.event == "POST_FALL_RESET":
                self.handler.reset()
            t += 0.033

        self.assertEqual(fsm.current_state, State.NORMAL)
        self.assertFalse(self.handler.is_alert_active)


if __name__ == "__main__":
    unittest.main()
