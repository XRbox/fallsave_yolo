"""
FallSave Step 5.3: Motion-Based Fall Event Detection Module

Analyzes biomechanical signals over a temporal sliding window to detect
an actual falling event rather than classifying a person as falling simply
because their body resembles a lying or horizontal posture.

Features analyzed:
1. Initial Body State: Distinguishes whether the person was standing/upright
   or already lying down when first tracked.
2. Downward Movement Velocity: Tracks vertical velocity and displacement of torso/hip.
3. Posture Transition: Tracks torso angle changes from upright toward horizontal.
4. Ground-Level Stability: Verifies persistent horizontal posture on the ground.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, Dict, Any, List
import numpy as np


class PostureState(Enum):
    """Categorization of body posture relative to vertical axis."""
    UPRIGHT = "UPRIGHT"          # Standing, walking, sitting upright (torso angle < upright_angle_max)
    BENT = "BENT"                # Leaning forward, bending down, crouching
    HORIZONTAL = "HORIZONTAL"    # Lying flat, prone, supine (torso angle >= horizontal_angle_min)
    UNKNOWN = "UNKNOWN"          # No person detected or tracking lost


@dataclass
class MotionConfig:
    """
    Configurable thresholds for motion-based fall event detection.
    
    Defaults are empirically grounded on URFD Training and Validation distributions:
    - Normal median mean_torso_angle: ~3.5 deg (90% <= 50.5 deg)
    - Falling median delta_torso_angle: ~42.6 deg (Train) / ~37.2 deg (Val)
    - Falling median max_downward_velocity: ~4.9 scale units/s (Train) / ~12.2 (Val)
    - Normal median max_downward_velocity: ~1.9 scale units/s (Train) / ~1.65 (Val)
    - Lying median delta_torso_angle: -1.44 deg (already flat on ground)
    """
    # Posture angle thresholds (degrees relative to vertical axis: 0=upright, 90=horizontal)
    upright_angle_max: float = 40.0         # Below this angle is classified as UPRIGHT
    horizontal_angle_min: float = 60.0      # Above this angle is classified as HORIZONTAL
    
    # Motion & Velocity thresholds (scale-normalized by torso length)
    min_downward_velocity: float = 2.0      # Min peak downward velocity to qualify as rapid drop (scale units/s)
    min_mean_downward_velocity: float = 0.3 # Min mean downward velocity over window (scale units/s)
    min_delta_hip_y: float = 0.20           # Min downward displacement of hip over window (scale units)
    min_delta_torso_angle: float = 18.0     # Min torso angle increase towards horizontal (degrees)
    
    # Temporal history thresholds
    min_upright_duration: float = 0.3       # Min seconds person must be tracked upright before a fall is credible
    max_time_since_upright: float = 1.8     # Max seconds allowed since last upright posture for a fall event
    slow_motion_velocity_limit: float = 1.2 # Peak downward velocity below which motion is deliberate/slow
    slow_motion_angle_rate: float = 35.0    # Angular change rate (deg/s) below which posture change is slow
    
    # Initial detection observation
    initial_settle_frames: int = 5          # Number of initial frames to determine starting body state


@dataclass
class MotionEvidence:
    """Structured evidence summarizing motion analysis for the current time step."""
    timestamp: float
    current_posture: PostureState
    torso_angle: float
    was_upright_recently: bool
    is_initial_lying: bool                  # Person was already lying when first detected
    has_rapid_descent: bool                 # Rapid downward velocity and/or vertical displacement
    has_posture_transition: bool           # Significant posture change from upright toward horizontal
    is_slow_intentional_movement: bool     # Posture changed but speed was slow / controlled
    is_credible_fall_motion: bool           # True when descent + transition + prior upright match fall mechanics
    downward_velocity: float
    vertical_displacement: float
    delta_angle: float
    confidence: float = 1.0
    details: Dict[str, Any] = field(default_factory=dict)


class MotionFallDetector:
    """
    Stateful temporal analyzer for motion-based fall event detection.
    
    Maintains history of body states, detects rapid vertical descents,
    filters out persons who were already lying down upon arrival, and rejects
    slow intentional movements (exercise, yoga, slow bed transitions).
    """

    def __init__(self, config: Optional[MotionConfig] = None):
        self.config = config or MotionConfig()
        self.reset()

    def reset(self):
        """Resets all history and state tracking."""
        self.last_timestamp: Optional[float] = None
        self.tracked_duration: float = 0.0
        self.upright_duration: float = 0.0
        self.last_upright_time: Optional[float] = None
        
        # Initial detection state
        self.initial_frame_count: int = 0
        self.initial_lying_frames: int = 0
        self.is_initial_lying: bool = False
        self.has_ever_stood_up: bool = False
        
        # Rolling status
        self.current_posture: PostureState = PostureState.UNKNOWN
        self.latest_torso_angle: float = 0.0

    def classify_posture(self, torso_angle: float) -> PostureState:
        """Classifies a given torso angle into PostureState enum."""
        if np.isnan(torso_angle):
            return PostureState.UNKNOWN
        if torso_angle < self.config.upright_angle_max:
            return PostureState.UPRIGHT
        elif torso_angle >= self.config.horizontal_angle_min:
            return PostureState.HORIZONTAL
        else:
            return PostureState.BENT

    def update(
        self,
        timestamp: float,
        torso_angle: float,
        downward_velocity: float,
        delta_hip_y: float,
        delta_torso_angle: float,
        person_detected: bool = True,
        mean_downward_velocity: Optional[float] = None,
        max_torso_angle: Optional[float] = None,
        p_lying: float = 0.0,
        person_confidence: float = 1.0
    ) -> MotionEvidence:
        """
        Updates the motion detector with current frame/window biomechanical metrics.
        
        Args:
            timestamp: Monotonic time in seconds.
            torso_angle: Current or mean torso angle (degrees relative to vertical).
            downward_velocity: Peak downward velocity of torso/hip (scale units/s).
            delta_hip_y: Downward displacement of hip over the window (scale units).
            delta_torso_angle: Change in torso angle over the window (degrees).
            person_detected: True if person keypoints are present.
            mean_downward_velocity: Mean downward velocity over the window.
            max_torso_angle: Maximum torso angle observed in the window.
            p_lying: Classifier probability for lying (if available).
            person_confidence: Keypoint / person detection confidence.
            
        Returns:
            MotionEvidence object.
        """
        if self.last_timestamp is None:
            self.last_timestamp = timestamp
        dt = max(0.0, timestamp - self.last_timestamp)
        self.last_timestamp = timestamp

        if not person_detected:
            # Person missing in current frame
            self.current_posture = PostureState.UNKNOWN
            was_upright = False
            if self.last_upright_time is not None:
                was_upright = (timestamp - self.last_upright_time) <= self.config.max_time_since_upright

            return MotionEvidence(
                timestamp=timestamp,
                current_posture=PostureState.UNKNOWN,
                torso_angle=self.latest_torso_angle,
                was_upright_recently=was_upright,
                is_initial_lying=self.is_initial_lying,
                has_rapid_descent=False,
                has_posture_transition=False,
                is_slow_intentional_movement=False,
                is_credible_fall_motion=False,
                downward_velocity=0.0,
                vertical_displacement=0.0,
                delta_angle=0.0,
                confidence=0.0,
                details={"reason": "no_person_detected"}
            )

        self.tracked_duration += dt
        self.latest_torso_angle = torso_angle
        posture = self.classify_posture(torso_angle)
        self.current_posture = posture

        # -------------------------------------------------------------
        # 1. INITIAL BODY STATE & UPRIGHT HISTORY TRACKING
        # -------------------------------------------------------------
        if self.initial_frame_count < self.config.initial_settle_frames:
            self.initial_frame_count += 1
            if posture == PostureState.HORIZONTAL or p_lying >= 0.50:
                self.initial_lying_frames += 1

            # If majority of initial observation is horizontal without prior standing:
            if self.initial_lying_frames >= (self.initial_frame_count // 2 + 1):
                if not self.has_ever_stood_up:
                    self.is_initial_lying = True

        # Track upright posture duration
        if posture == PostureState.UPRIGHT:
            self.upright_duration += dt
            self.last_upright_time = timestamp
            if self.upright_duration >= self.config.min_upright_duration:
                self.has_ever_stood_up = True
                self.is_initial_lying = False  # Stood up stably; no longer initial lying
        else:
            # Not upright
            self.upright_duration = 0.0

        # Check if person was upright recently
        was_upright_recently = False
        if self.has_ever_stood_up and (self.last_upright_time is not None):
            was_upright_recently = (timestamp - self.last_upright_time) <= self.config.max_time_since_upright

        # -------------------------------------------------------------
        # 2. DOWNWARD MOVEMENT VELOCITY ANALYSIS
        # -------------------------------------------------------------
        # Rapid downward velocity and/or vertical displacement towards floor
        mean_v = mean_downward_velocity if mean_downward_velocity is not None else (downward_velocity * 0.3)
        has_downward_velocity = downward_velocity >= self.config.min_downward_velocity
        has_hip_displacement = delta_hip_y >= self.config.min_delta_hip_y
        has_positive_mean_v = mean_v >= self.config.min_mean_downward_velocity

        has_rapid_descent = (has_downward_velocity or has_hip_displacement) and (has_positive_mean_v or delta_hip_y > 0.15)

        # -------------------------------------------------------------
        # 3. POSTURE TRANSITION ANALYSIS
        # -------------------------------------------------------------
        # Transition from upright towards horizontal
        peak_angle = max_torso_angle if max_torso_angle is not None else torso_angle
        has_angle_delta = delta_torso_angle >= self.config.min_delta_torso_angle
        reaches_horizontal = peak_angle >= (self.config.horizontal_angle_min - 5.0)

        has_posture_transition = has_angle_delta and reaches_horizontal

        # -------------------------------------------------------------
        # 4. SLOW INTENTIONAL MOVEMENT FILTER (Yoga, slow lying down)
        # -------------------------------------------------------------
        # If posture transitioned to horizontal BUT velocity is below slow limit:
        is_slow_intentional = False
        if has_posture_transition:
            if downward_velocity < self.config.slow_motion_velocity_limit and delta_hip_y < (self.config.min_delta_hip_y * 0.8):
                is_slow_intentional = True

        # If person is already horizontal and stays horizontal:
        if posture == PostureState.HORIZONTAL and not has_angle_delta and not has_rapid_descent:
            is_slow_intentional = True

        # -------------------------------------------------------------
        # 5. CREDIBLE FALL MOTION SYNTHESIS
        # -------------------------------------------------------------
        is_credible_fall = (
            person_detected
            and not self.is_initial_lying
            and was_upright_recently
            and has_rapid_descent
            and has_posture_transition
            and not is_slow_intentional
        )

        return MotionEvidence(
            timestamp=timestamp,
            current_posture=posture,
            torso_angle=torso_angle,
            was_upright_recently=was_upright_recently,
            is_initial_lying=self.is_initial_lying,
            has_rapid_descent=has_rapid_descent,
            has_posture_transition=has_posture_transition,
            is_slow_intentional_movement=is_slow_intentional,
            is_credible_fall_motion=is_credible_fall,
            downward_velocity=downward_velocity,
            vertical_displacement=delta_hip_y,
            delta_angle=delta_torso_angle,
            confidence=person_confidence,
            details={
                "has_ever_stood_up": self.has_ever_stood_up,
                "upright_duration": self.upright_duration,
                "peak_angle": peak_angle,
                "mean_v": mean_v
            }
        )

    def analyze_feature_vector(
        self,
        features: np.ndarray,
        timestamp: float,
        person_detected: bool = True,
        p_lying: float = 0.0
    ) -> MotionEvidence:
        """
        Convenience method to compute MotionEvidence directly from the
        17-element biomechanical feature vector produced by LiveFeatureBuffer.
        
        Feature indices:
            0: mean_torso_angle
            1: max_torso_angle
            2: min_torso_angle
            3: delta_torso_angle
            4: std_torso_angle
            5: max_downward_velocity
            6: mean_vertical_velocity
            7: delta_hip_y
            ...
            16: mean_person_confidence
        """
        if not person_detected or features is None or len(features) < 17:
            return self.update(
                timestamp=timestamp,
                torso_angle=0.0,
                downward_velocity=0.0,
                delta_hip_y=0.0,
                delta_torso_angle=0.0,
                person_detected=False
            )

        mean_angle = float(features[0])
        max_angle = float(features[1])
        delta_angle = float(features[3])
        max_drop_v = float(features[5])
        mean_v = float(features[6])
        delta_hip = float(features[7])
        conf = float(features[16]) if len(features) > 16 else 1.0

        return self.update(
            timestamp=timestamp,
            torso_angle=mean_angle,
            downward_velocity=max_drop_v,
            delta_hip_y=delta_hip,
            delta_torso_angle=delta_angle,
            person_detected=person_detected,
            mean_downward_velocity=mean_v,
            max_torso_angle=max_angle,
            p_lying=p_lying,
            person_confidence=conf
        )
