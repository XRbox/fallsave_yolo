"""
FallSave Step 4.2: Unit Tests for FallStateMachine

Tests cover:
1. Normal activity (routine walks, sits, standing)
2. Isolated falling probability spike (does not trigger alert)
3. Genuine fall with sustained lying posture (triggers alert exactly once)
4. False alarm / recovery (stumble followed by quick stand-up)
5. Fall followed by complete pose-detection loss (missing person on floor)
6. Intermittent missing detections during ground check (ground timer persists)
7. Irregular frame rate / timestamp jitter
8. Manual reset functionality
"""

import unittest
from fall_state_machine import FallStateMachine, StateMachineConfig, State


class TestFallStateMachine(unittest.TestCase):

    def setUp(self):
        self.config = StateMachineConfig(
            p_fall_threshold=0.45,
            p_lying_threshold=0.35,
            p_normal_recovery=0.65,
            suspect_timeout=1.5,
            quick_recovery_time=0.4,
            ground_confirm_time=2.5,
            standup_recovery_time=1.0,
            post_confirm_reset_time=5.0
        )
        self.fsm = FallStateMachine(self.config)

    def test_normal_activity_remains_in_normal(self):
        """Routine normal activity should never leave NORMAL state."""
        t = 0.0
        events = []
        for _ in range(60):  # 2 seconds at 30 FPS
            out = self.fsm.update(
                timestamp=t,
                p_normal=0.92,
                p_falling=0.04,
                p_lying=0.04,
                person_detected=True
            )
            if out.event:
                events.append(out.event)
            self.assertEqual(out.state, State.NORMAL)
            t += 0.033

        self.assertEqual(len(events), 0)
        self.assertFalse(self.fsm.has_alerted)

    def test_isolated_falling_spike_does_not_alert(self):
        """A single high P(falling) spike followed by normal posture must not alert."""
        t = 0.0
        # 1. Normal state for 1s
        for _ in range(30):
            out = self.fsm.update(t, p_normal=0.90, p_falling=0.05, p_lying=0.05, person_detected=True)
            t += 0.033

        # 2. Single isolated spike
        out = self.fsm.update(t, p_normal=0.05, p_falling=0.92, p_lying=0.03, person_detected=True)
        self.assertEqual(out.state, State.FALLING_SUSPECTED)
        self.assertEqual(out.event, "FALL_SUSPECTED")
        t += 0.033

        # 3. Immediately returns to normal upright posture
        events = []
        for _ in range(30):  # 1 second of normal
            out = self.fsm.update(t, p_normal=0.88, p_falling=0.06, p_lying=0.06, person_detected=True)
            if out.event:
                events.append(out.event)
            t += 0.033

        # Must clear back to NORMAL without entering GROUND_CHECK or ALERT
        self.assertEqual(self.fsm.current_state, State.NORMAL)
        self.assertIn("SUSPECT_CLEARED", events)
        self.assertNotIn("ALERT_TRIGGERED", events)
        self.assertFalse(self.fsm.has_alerted)

    def test_genuine_fall_with_lying_triggers_alert_once(self):
        """A genuine fall followed by lying triggers ALERT_TRIGGERED exactly once."""
        t = 0.0
        # 1. Normal walking (1 second)
        for _ in range(30):
            self.fsm.update(t, p_normal=0.90, p_falling=0.05, p_lying=0.05, person_detected=True)
            t += 0.033

        # 2. Fall descent (3 frames of high falling probability)
        for _ in range(3):
            out = self.fsm.update(t, p_normal=0.10, p_falling=0.80, p_lying=0.10, person_detected=True)
            t += 0.033
        self.assertEqual(self.fsm.current_state, State.FALLING_SUSPECTED)

        # 3. Ground contact: lying posture begins
        out = self.fsm.update(t, p_normal=0.05, p_falling=0.15, p_lying=0.80, person_detected=True)
        self.assertEqual(out.state, State.GROUND_CHECK)
        self.assertEqual(out.event, "GROUND_CHECK_ENTERED")
        t += 0.033

        # 4. Remains lying on ground for 3.0 seconds (confirmation limit is 2.5s)
        alert_events = []
        for _ in range(90):
            out = self.fsm.update(t, p_normal=0.05, p_falling=0.05, p_lying=0.90, person_detected=True)
            if out.event == "ALERT_TRIGGERED":
                alert_events.append(t)
            t += 0.033

        self.assertEqual(self.fsm.current_state, State.FALL_CONFIRMED)
        self.assertTrue(self.fsm.has_alerted)
        # CRITICAL: Alert event must have fired EXACTLY ONCE
        self.assertEqual(len(alert_events), 1)

    def test_false_alarm_quick_recovery(self):
        """Stumble or sudden sit-down where person stands back up cancels the alert."""
        t = 0.0
        # 1. Normal
        for _ in range(15):
            self.fsm.update(t, p_normal=0.90, p_falling=0.05, p_lying=0.05, person_detected=True)
            t += 0.033

        # 2. Stumble (falling suspected)
        self.fsm.update(t, p_normal=0.10, p_falling=0.75, p_lying=0.15, person_detected=True)
        t += 0.033

        # 3. Touch ground momentarily (ground check entered)
        out = self.fsm.update(t, p_normal=0.10, p_falling=0.10, p_lying=0.80, person_detected=True)
        self.assertEqual(out.state, State.GROUND_CHECK)
        t += 0.033

        # 4. Stays grounded for only 0.8 seconds (less than 2.5s confirmation threshold)
        for _ in range(24):
            self.fsm.update(t, p_normal=0.10, p_falling=0.10, p_lying=0.80, person_detected=True)
            t += 0.033

        # 5. Stands up: sustained normal posture for 1.2s (standup_recovery_time is 1.0s)
        events = []
        for _ in range(36):
            out = self.fsm.update(t, p_normal=0.92, p_falling=0.04, p_lying=0.04, person_detected=True)
            if out.event:
                events.append(out.event)
            t += 0.033

        self.assertEqual(self.fsm.current_state, State.NORMAL)
        self.assertIn("RECOVERED", events)
        self.assertNotIn("ALERT_TRIGGERED", events)
        self.assertFalse(self.fsm.has_alerted)

    def test_fall_followed_by_pose_detection_loss(self):
        """Fall followed by complete keypoint detection loss (person flat on floor) confirms alert."""
        t = 0.0
        # 1. Normal
        for _ in range(20):
            self.fsm.update(t, p_normal=0.90, p_falling=0.05, p_lying=0.05, person_detected=True)
            t += 0.033

        # 2. Rapid fall
        out = self.fsm.update(t, p_normal=0.10, p_falling=0.85, p_lying=0.05, person_detected=True)
        self.assertEqual(out.state, State.FALLING_SUSPECTED)
        t += 0.033

        # 3. Person drops to floor: YOLO detects 0 keypoints (person_detected=False)
        out = self.fsm.update(t, p_normal=0.0, p_falling=0.0, p_lying=0.0, person_detected=False)
        self.assertEqual(out.state, State.GROUND_CHECK)
        self.assertEqual(out.event, "GROUND_CHECK_ENTERED")
        t += 0.033

        # 4. Person remains missing for 3.0 seconds
        alert_fired = False
        for _ in range(90):
            out = self.fsm.update(t, p_normal=0.0, p_falling=0.0, p_lying=0.0, person_detected=False)
            if out.event == "ALERT_TRIGGERED":
                alert_fired = True
            t += 0.033

        self.assertEqual(self.fsm.current_state, State.FALL_CONFIRMED)
        self.assertTrue(alert_fired)
        self.assertTrue(self.fsm.has_alerted)

    def test_intermittent_missing_detections_during_ground_check(self):
        """Intermittent keypoint dropouts do not reset ground check timer."""
        t = 0.0
        # 1. Fall suspected
        self.fsm.update(t, p_normal=0.10, p_falling=0.80, p_lying=0.10, person_detected=True)
        t += 0.033

        # 2. Enters ground check
        out = self.fsm.update(t, p_normal=0.05, p_falling=0.05, p_lying=0.90, person_detected=True)
        self.assertEqual(out.state, State.GROUND_CHECK)
        t += 0.033

        # 3. Simulate flickering detection: 2 frames detected lying, 2 frames missing, etc. for 3.0s
        alert_count = 0
        for i in range(90):
            detected = (i % 4 < 2)
            p_lie = 0.85 if detected else 0.0
            p_norm = 0.05 if detected else 0.0
            out = self.fsm.update(t, p_normal=p_norm, p_falling=0.05, p_lying=p_lie, person_detected=detected)
            if out.event == "ALERT_TRIGGERED":
                alert_count += 1
            t += 0.033

        # Ground timer progressed despite flickering detection
        self.assertEqual(self.fsm.current_state, State.FALL_CONFIRMED)
        self.assertEqual(alert_count, 1)

    def test_timestamp_jitter_and_framerate_variation(self):
        """State machine behaves accurately with irregular dt intervals."""
        t = 0.0
        # Fall suspected
        self.fsm.update(t, p_normal=0.10, p_falling=0.80, p_lying=0.10, person_detected=True)

        # Large jump of 0.2s: enters ground check
        t += 0.2
        self.fsm.update(t, p_normal=0.05, p_falling=0.05, p_lying=0.90, person_detected=True)
        self.assertEqual(self.fsm.current_state, State.GROUND_CHECK)

        # Variable time increments totaling 2.6 seconds
        time_steps = [0.05, 0.1, 0.02, 0.08, 0.15, 0.5, 0.7, 1.0]  # total = 2.6s
        alert_found = False
        for dt in time_steps:
            t += dt
            out = self.fsm.update(t, p_normal=0.05, p_falling=0.05, p_lying=0.90, person_detected=True)
            if out.event == "ALERT_TRIGGERED":
                alert_found = True

        self.assertTrue(alert_found)
        self.assertEqual(self.fsm.current_state, State.FALL_CONFIRMED)

    def test_manual_reset(self):
        """Calling reset() returns state machine to NORMAL from FALL_CONFIRMED."""
        # Force into FALL_CONFIRMED
        self.fsm.current_state = State.FALL_CONFIRMED
        self.fsm.has_alerted = True

        self.fsm.reset()
        self.assertEqual(self.fsm.current_state, State.NORMAL)
        self.assertFalse(self.fsm.has_alerted)
        self.assertIsNone(self.fsm.ground_start_time)


if __name__ == "__main__":
    unittest.main()
