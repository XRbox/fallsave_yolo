"""
Unit tests for FallSave Step 5.3: Motion-Based Fall Event Detection

Verifies the 10 focused test scenarios required by Step 5.3:
1. Person standing still: no fall.
2. Person already lying down when first detected: no fall.
3. Person slowly lying down intentionally: no fall.
4. Person bending down: no fall.
5. Person doing a yoga-like movement: no alert based only on horizontal posture.
6. Person rapidly moving downward and transitioning from upright to horizontal: enters fall suspicion.
7. Person rapidly descending and remaining on the ground: confirms fall.
8. Person rapidly descending but recovering quickly: cancels suspected fall.
9. Person disappears after a credible suspected fall: confirms fall.
10. Temporary detection loss without previous fall evidence: no alert.
"""

import unittest
from motion_detector import MotionFallDetector, MotionConfig, PostureState, MotionEvidence
from fall_state_machine import FallStateMachine, StateMachineConfig, State


class TestMotionFallDetection(unittest.TestCase):
    """Test suite covering the 10 required Step 5.3 scenarios."""

    def setUp(self):
        self.motion_cfg = MotionConfig(
            upright_angle_max=40.0,
            horizontal_angle_min=60.0,
            min_downward_velocity=2.0,
            min_delta_hip_y=0.20,
            min_delta_torso_angle=18.0,
            min_upright_duration=0.3,
            max_time_since_upright=1.8,
            slow_motion_velocity_limit=1.2
        )
        self.fsm_cfg = StateMachineConfig(
            p_fall_threshold=0.45,
            p_lying_threshold=0.35,
            p_normal_recovery=0.65,
            suspect_timeout=1.5,
            quick_recovery_time=0.4,
            ground_confirm_time=2.5,
            standup_recovery_time=1.0,
            enable_motion_validation=True,
            require_upright_before_fall=True
        )
        self.detector = MotionFallDetector(self.motion_cfg)
        self.fsm = FallStateMachine(self.fsm_cfg)

    # -------------------------------------------------------------------------
    # Scenario 1: Person standing still -> No fall
    # -------------------------------------------------------------------------
    def test_scenario_01_person_standing_still(self):
        """Person standing still produces upright posture, zero downward velocity, and remains NORMAL."""
        t = 0.0
        dt = 0.033
        events = []

        for _ in range(60):  # 2.0 seconds
            ev = self.detector.update(
                timestamp=t,
                torso_angle=8.0,
                downward_velocity=0.1,
                delta_hip_y=0.01,
                delta_torso_angle=0.5,
                person_detected=True,
                p_lying=0.02
            )
            out = self.fsm.update(
                timestamp=t,
                p_normal=0.95,
                p_falling=0.03,
                p_lying=0.02,
                person_detected=True,
                motion_evidence=ev
            )
            if out.event:
                events.append(out.event)
            t += dt

        self.assertEqual(ev.current_posture, PostureState.UPRIGHT)
        self.assertFalse(ev.has_rapid_descent)
        self.assertFalse(ev.has_posture_transition)
        self.assertFalse(ev.is_credible_fall_motion)
        self.assertEqual(self.fsm.current_state, State.NORMAL)
        self.assertEqual(len(events), 0)

    # -------------------------------------------------------------------------
    # Scenario 2: Person already lying down when first detected -> No fall
    # -------------------------------------------------------------------------
    def test_scenario_02_already_lying_down_when_first_detected(self):
        """Person already lying flat upon arrival is flagged as is_initial_lying and never triggers a fall."""
        t = 0.0
        dt = 0.033
        events = []

        # Person enters camera already lying flat for 4.0 seconds
        for _ in range(120):
            ev = self.detector.update(
                timestamp=t,
                torso_angle=85.0,
                downward_velocity=0.1,
                delta_hip_y=0.0,
                delta_torso_angle=-0.5,
                person_detected=True,
                p_lying=0.92
            )
            out = self.fsm.update(
                timestamp=t,
                p_normal=0.05,
                p_falling=0.10,
                p_lying=0.85,
                person_detected=True,
                motion_evidence=ev
            )
            if out.event:
                events.append(out.event)
            t += dt

        self.assertTrue(ev.is_initial_lying)
        self.assertFalse(ev.was_upright_recently)
        self.assertFalse(ev.is_credible_fall_motion)
        self.assertEqual(self.fsm.current_state, State.NORMAL)
        self.assertNotIn("FALL_SUSPECTED", events)
        self.assertNotIn("ALERT_TRIGGERED", events)

    # -------------------------------------------------------------------------
    # Scenario 3: Person slowly lying down intentionally -> No fall
    # -------------------------------------------------------------------------
    def test_scenario_03_person_slowly_lying_down_intentionally(self):
        """Slow, controlled transition to horizontal (velocity < limit) is classified as intentional and rejected."""
        t = 0.0
        dt = 0.033
        events = []

        # 1. Standing upright for 1.0s
        for _ in range(30):
            ev = self.detector.update(
                t, torso_angle=10.0, downward_velocity=0.2,
                delta_hip_y=0.02, delta_torso_angle=0.0, person_detected=True
            )
            self.fsm.update(t, p_normal=0.95, p_falling=0.02, p_lying=0.03, person_detected=True, motion_evidence=ev)
            t += dt

        # 2. Slow, intentional reclining over 2.0s: low velocity (0.6 scale units/s)
        for i in range(60):
            angle = 10.0 + (75.0 - 10.0) * (i / 60.0)
            ev = self.detector.update(
                t, torso_angle=angle, downward_velocity=0.6,
                delta_hip_y=0.12, delta_torso_angle=19.0, person_detected=True, p_lying=0.40
            )
            out = self.fsm.update(
                t, p_normal=0.40, p_falling=0.48, p_lying=0.50,
                person_detected=True, motion_evidence=ev
            )
            if out.event:
                events.append(out.event)
            t += dt

        # 3. Lying flat for 3.0s
        for _ in range(90):
            ev = self.detector.update(
                t, torso_angle=80.0, downward_velocity=0.05,
                delta_hip_y=0.0, delta_torso_angle=0.0, person_detected=True, p_lying=0.95
            )
            out = self.fsm.update(
                t, p_normal=0.05, p_falling=0.05, p_lying=0.90,
                person_detected=True, motion_evidence=ev
            )
            if out.event:
                events.append(out.event)
            t += dt

        self.assertEqual(self.fsm.current_state, State.NORMAL)
        self.assertNotIn("FALL_SUSPECTED", events)
        self.assertNotIn("ALERT_TRIGGERED", events)

    # -------------------------------------------------------------------------
    # Scenario 4: Person bending down -> No fall
    # -------------------------------------------------------------------------
    def test_scenario_04_person_bending_down(self):
        """Bending down (torso angle tilts, but hip displacement is low and returns upright) does not confirm fall."""
        t = 0.0
        dt = 0.033
        events = []

        # 1. Standing upright for 1.0s
        for _ in range(30):
            ev = self.detector.update(t, torso_angle=10.0, downward_velocity=0.1, delta_hip_y=0.0, delta_torso_angle=0.0)
            self.fsm.update(t, p_normal=0.92, p_falling=0.05, p_lying=0.03, person_detected=True, motion_evidence=ev)
            t += dt

        # 2. Bending forward to pick up keys for 1.0s: torso tilts to 50 deg, but hip drop is only 0.08
        for _ in range(30):
            ev = self.detector.update(
                t, torso_angle=50.0, downward_velocity=1.0,
                delta_hip_y=0.08, delta_torso_angle=25.0, person_detected=True
            )
            out = self.fsm.update(t, p_normal=0.60, p_falling=0.35, p_lying=0.05, person_detected=True, motion_evidence=ev)
            if out.event:
                events.append(out.event)
            t += dt

        # 3. Stands back upright for 1.0s
        for _ in range(30):
            ev = self.detector.update(t, torso_angle=12.0, downward_velocity=0.1, delta_hip_y=-0.05, delta_torso_angle=-38.0)
            out = self.fsm.update(t, p_normal=0.95, p_falling=0.02, p_lying=0.03, person_detected=True, motion_evidence=ev)
            if out.event:
                events.append(out.event)
            t += dt

        self.assertEqual(self.fsm.current_state, State.NORMAL)
        self.assertNotIn("ALERT_TRIGGERED", events)

    # -------------------------------------------------------------------------
    # Scenario 5: Person doing yoga-like movement -> No alert on horizontal posture alone
    # -------------------------------------------------------------------------
    def test_scenario_05_yoga_movement_horizontal_posture_only(self):
        """High P(lying) during yoga mat floor exercises does not trigger an alert without prior rapid fall event."""
        t = 0.0
        dt = 0.033
        events = []

        # Person performing yoga floor exercise (Cobra / Child's pose) for 5.0 seconds
        for _ in range(150):
            ev = self.detector.update(
                t, torso_angle=72.0, downward_velocity=0.3,
                delta_hip_y=0.05, delta_torso_angle=2.0, person_detected=True, p_lying=0.88
            )
            out = self.fsm.update(
                t, p_normal=0.10, p_falling=0.08, p_lying=0.82,
                person_detected=True, motion_evidence=ev
            )
            if out.event:
                events.append(out.event)
            t += dt

        self.assertFalse(ev.is_credible_fall_motion)
        self.assertEqual(self.fsm.current_state, State.NORMAL)
        self.assertNotIn("FALL_SUSPECTED", events)
        self.assertNotIn("ALERT_TRIGGERED", events)

    # -------------------------------------------------------------------------
    # Scenario 6: Rapid descent + upright-to-horizontal -> Enters fall suspicion
    # -------------------------------------------------------------------------
    def test_scenario_06_rapid_descent_enters_fall_suspicion(self):
        """Rapid downward velocity and significant angle transition correctly transitions to FALLING_SUSPECTED."""
        t = 0.0
        dt = 0.033

        # 1. Standing upright for 1.0s
        for _ in range(30):
            ev = self.detector.update(t, torso_angle=10.0, downward_velocity=0.2, delta_hip_y=0.01, delta_torso_angle=0.0)
            self.fsm.update(t, p_normal=0.92, p_falling=0.05, p_lying=0.03, person_detected=True, motion_evidence=ev)
            t += dt

        # 2. Sudden fall drop
        ev = self.detector.update(
            t, torso_angle=75.0, downward_velocity=6.2,
            delta_hip_y=0.75, delta_torso_angle=58.0, person_detected=True,
            mean_downward_velocity=1.5, max_torso_angle=80.0
        )
        out = self.fsm.update(
            t, p_normal=0.10, p_falling=0.85, p_lying=0.05,
            person_detected=True, motion_evidence=ev
        )

        self.assertTrue(ev.has_rapid_descent)
        self.assertTrue(ev.has_posture_transition)
        self.assertTrue(ev.is_credible_fall_motion)
        self.assertEqual(out.state, State.FALLING_SUSPECTED)
        self.assertEqual(out.event, "FALL_SUSPECTED")

    # -------------------------------------------------------------------------
    # Scenario 7: Rapid descent + remaining on ground -> Confirms fall alert
    # -------------------------------------------------------------------------
    def test_scenario_07_rapid_descent_remaining_on_ground_confirms_fall(self):
        """Rapid fall followed by staying on the ground for >= 2.5s confirms fall and triggers alert exactly once."""
        t = 0.0
        dt = 0.033
        alert_events = []

        # 1. Standing upright for 1.0s
        for _ in range(30):
            ev = self.detector.update(t, torso_angle=10.0, downward_velocity=0.2, delta_hip_y=0.01, delta_torso_angle=0.0)
            self.fsm.update(t, p_normal=0.92, p_falling=0.05, p_lying=0.03, person_detected=True, motion_evidence=ev)
            t += dt

        # 2. Rapid fall spike
        ev = self.detector.update(
            t, torso_angle=75.0, downward_velocity=6.5,
            delta_hip_y=0.85, delta_torso_angle=60.0, person_detected=True,
            mean_downward_velocity=1.6, max_torso_angle=82.0
        )
        self.fsm.update(t, p_normal=0.10, p_falling=0.88, p_lying=0.02, person_detected=True, motion_evidence=ev)
        t += dt

        # 3. Ground contact: lying on floor for 3.0s (> 2.5s confirm limit)
        for _ in range(90):
            ev = self.detector.update(
                t, torso_angle=85.0, downward_velocity=0.1,
                delta_hip_y=0.01, delta_torso_angle=0.0, person_detected=True, p_lying=0.95
            )
            out = self.fsm.update(
                t, p_normal=0.05, p_falling=0.05, p_lying=0.90,
                person_detected=True, motion_evidence=ev
            )
            if out.event == "ALERT_TRIGGERED":
                alert_events.append(out.event)
            t += dt

        self.assertEqual(self.fsm.current_state, State.FALL_CONFIRMED)
        self.assertEqual(len(alert_events), 1)

    # -------------------------------------------------------------------------
    # Scenario 8: Rapid descent but recovering quickly -> Cancels suspected fall
    # -------------------------------------------------------------------------
    def test_scenario_08_rapid_descent_quick_recovery_cancels_fall(self):
        """Person stumbles rapidly but immediately regains upright posture within quick_recovery_time clears suspect."""
        t = 0.0
        dt = 0.033
        events = []

        # 1. Standing upright for 1.0s
        for _ in range(30):
            ev = self.detector.update(t, torso_angle=10.0, downward_velocity=0.2, delta_hip_y=0.0, delta_torso_angle=0.0)
            self.fsm.update(t, p_normal=0.92, p_falling=0.05, p_lying=0.03, person_detected=True, motion_evidence=ev)
            t += dt

        # 2. Sudden stumble
        ev = self.detector.update(
            t, torso_angle=65.0, downward_velocity=4.5,
            delta_hip_y=0.60, delta_torso_angle=45.0, person_detected=True,
            mean_downward_velocity=1.2, max_torso_angle=70.0
        )
        out = self.fsm.update(t, p_normal=0.15, p_falling=0.75, p_lying=0.10, person_detected=True, motion_evidence=ev)
        self.assertEqual(out.state, State.FALLING_SUSPECTED)
        t += dt

        # 3. Immediately stands back upright within 0.1s and sustains it for 0.6s (> 0.4s quick recovery)
        for _ in range(20):
            ev = self.detector.update(t, torso_angle=12.0, downward_velocity=-0.5, delta_hip_y=-0.4, delta_torso_angle=-40.0)
            out = self.fsm.update(t, p_normal=0.90, p_falling=0.05, p_lying=0.05, person_detected=True, motion_evidence=ev)
            if out.event:
                events.append(out.event)
            t += dt

        self.assertEqual(self.fsm.current_state, State.NORMAL)
        self.assertIn("SUSPECT_CLEARED", events)
        self.assertNotIn("ALERT_TRIGGERED", events)

    # -------------------------------------------------------------------------
    # Scenario 9: Person disappears after credible suspected fall -> Confirms fall
    # -------------------------------------------------------------------------
    def test_scenario_09_person_disappears_after_credible_fall_confirms(self):
        """When YOLO loses keypoints right after a credible rapid fall, ground check timer persists and alerts."""
        t = 0.0
        dt = 0.033
        alert_events = []

        # 1. Standing upright for 1.0s
        for _ in range(30):
            ev = self.detector.update(t, torso_angle=10.0, downward_velocity=0.2, delta_hip_y=0.0, delta_torso_angle=0.0)
            self.fsm.update(t, p_normal=0.92, p_falling=0.05, p_lying=0.03, person_detected=True, motion_evidence=ev)
            t += dt

        # 2. Sudden fall impact
        ev = self.detector.update(
            t, torso_angle=75.0, downward_velocity=5.8,
            delta_hip_y=0.80, delta_torso_angle=55.0, person_detected=True,
            mean_downward_velocity=1.4, max_torso_angle=80.0
        )
        self.fsm.update(t, p_normal=0.10, p_falling=0.85, p_lying=0.05, person_detected=True, motion_evidence=ev)
        t += dt

        # 3. Person drops behind bed/furniture: YOLO loses person (person_detected=False) for 3.0s (> 2.5s)
        for _ in range(90):
            ev = self.detector.update(
                t, torso_angle=0.0, downward_velocity=0.0, delta_hip_y=0.0, delta_torso_angle=0.0, person_detected=False
            )
            out = self.fsm.update(
                t, p_normal=0.0, p_falling=0.0, p_lying=0.0, person_detected=False, motion_evidence=ev
            )
            if out.event == "ALERT_TRIGGERED":
                alert_events.append(out.event)
            t += dt

        self.assertEqual(self.fsm.current_state, State.FALL_CONFIRMED)
        self.assertEqual(len(alert_events), 1)

    # -------------------------------------------------------------------------
    # Scenario 10: Temporary detection loss without previous fall -> No alert
    # -------------------------------------------------------------------------
    def test_scenario_10_temporary_detection_loss_without_previous_fall(self):
        """Temporary tracking dropout or person walking out of camera without fall motion never triggers an alert."""
        t = 0.0
        dt = 0.033
        events = []

        # 1. Standing upright for 1.0s
        for _ in range(30):
            ev = self.detector.update(t, torso_angle=10.0, downward_velocity=0.2, delta_hip_y=0.0, delta_torso_angle=0.0)
            self.fsm.update(t, p_normal=0.92, p_falling=0.05, p_lying=0.03, person_detected=True, motion_evidence=ev)
            t += dt

        # 2. Person walks behind pillar or leaves room: person_detected=False for 4.0 seconds
        for _ in range(120):
            ev = self.detector.update(
                t, torso_angle=0.0, downward_velocity=0.0, delta_hip_y=0.0, delta_torso_angle=0.0, person_detected=False
            )
            out = self.fsm.update(
                t, p_normal=1.0, p_falling=0.0, p_lying=0.0, person_detected=False, motion_evidence=ev
            )
            if out.event:
                events.append(out.event)
            t += dt

        self.assertEqual(self.fsm.current_state, State.NORMAL)
        self.assertNotIn("FALL_SUSPECTED", events)
        self.assertNotIn("GROUND_CHECK_ENTERED", events)
        self.assertNotIn("ALERT_TRIGGERED", events)


if __name__ == "__main__":
    unittest.main()
