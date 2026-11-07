"""
FallSave Step 5.3: Motion-Based Fall Event Detection Module (Refined)

Analyzes biomechanical signals over a temporal sliding window to detect
an actual falling event rather than classifying a person as falling simply
because their body resembles a lying or horizontal posture.

Key Improvements:
1. Frame-Level Pose Extraction: Detects instantaneous torso angle and hip position
   directly from keypoints, ensuring is_initial_lying is immediately detected
   from Frame 1 rather than defaulting to upright (0.0 deg) during buffer warmup.
2. Complete Temporal History: Tracks posture progression over time (last 2.0s) to
   distinguish intentional, multi-stage, or controlled reclining (e.g. bed, sofa,
   yoga) from an uncontrolled free-fall collapse.
3. Decoupled Descent & Reclining: Differentiates torso tilting onto an elevated
   surface (where hips remain at bed height) from an uncontrolled floor collapse
   (where hips and torso crash to the floor with high downward velocity).
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, Dict, Any, List, Tuple
from collections import deque
import numpy as np


class PostureState(Enum):
    """Categorization of body posture relative to vertical axis."""
    UPRIGHT = "UPRIGHT"          # Standing, walking, sitting upright (torso angle < upright_angle_max)
    BENT = "BENT"                # Leaning forward, bending down, intermediate seated
    HORIZONTAL = "HORIZONTAL"    # Lying flat, prone, supine (torso angle >= horizontal_angle_min)
    UNKNOWN = "UNKNOWN"          # No person detected or tracking lost


def extract_single_frame_pose(
    kps: Optional[np.ndarray],
    conf_threshold: float = 0.20
) -> Dict[str, Any]:
    """
    Extracts frame-level pose metrics (torso_angle, hip_y, torso_len, posture)
    directly from a single frame's 17 keypoints.
    
    Keypoint indices:
        5: left shoulder, 6: right shoulder
        11: left hip, 12: right hip
    """
    if kps is None or len(kps) != 17:
        return {
            "valid": False,
            "torso_angle": np.nan,
            "hip_y": np.nan,
            "torso_len": 50.0,
            "posture": PostureState.UNKNOWN
        }

    # Shoulder joints (5, 6)
    valid_sh = [i for i in (5, 6) if kps[i, 2] >= conf_threshold]
    # Hip joints (11, 12)
    valid_hip = [i for i in (11, 12) if kps[i, 2] >= conf_threshold]

    if not valid_sh or not valid_hip:
        return {
            "valid": False,
            "torso_angle": np.nan,
            "hip_y": np.nan,
            "torso_len": 50.0,
            "posture": PostureState.UNKNOWN
        }

    sh_x = float(np.mean([kps[i, 0] for i in valid_sh]))
    sh_y = float(np.mean([kps[i, 1] for i in valid_sh]))
    hip_x = float(np.mean([kps[i, 0] for i in valid_hip]))
    hip_y = float(np.mean([kps[i, 1] for i in valid_hip]))

    dx = sh_x - hip_x
    dy = sh_y - hip_y
    torso_len = max(float(np.sqrt(dx**2 + dy**2)), 15.0)

    angle = float(np.degrees(np.arctan2(np.abs(dx), np.abs(dy))))
    if sh_y > hip_y:
        angle = 180.0 - angle  # Inverted (head below hips)

    if angle < 40.0:
        posture = PostureState.UPRIGHT
    elif angle >= 60.0:
        posture = PostureState.HORIZONTAL
    else:
        posture = PostureState.BENT

    return {
        "valid": True,
        "torso_angle": angle,
        "hip_y": hip_y,
        "torso_len": torso_len,
        "posture": posture
    }


@dataclass
class MotionConfig:
    """Configurable thresholds for motion-based fall event detection."""
    # Posture angle thresholds (degrees relative to vertical axis: 0=upright, 90=horizontal)
    upright_angle_max: float = 40.0         # Below this angle is classified as UPRIGHT
    horizontal_angle_min: float = 60.0      # Above this angle is classified as HORIZONTAL
    
    # Rapid descent thresholds (scale-normalized by torso length)
    # Empirically: Fall median max_downward_v = 4.91 (Train) / 12.18 (Val), Normal median = 1.93
    min_downward_velocity: float = 2.8      # Min peak downward velocity for rapid freefall (scale units/s)
    min_mean_downward_velocity: float = 0.35# Min mean downward velocity over window
    min_delta_hip_y: float = 0.25           # Min downward displacement of hip over window (scale units)
    min_delta_torso_angle: float = 20.0     # Min torso angle increase towards horizontal (degrees)
    min_angular_velocity: float = 160.0     # Min peak angular velocity for violent tipping (deg/s)
    
    # Intentional lying down discrimination thresholds
    max_intentional_downward_v: float = 2.5 # Max downward velocity in controlled recline
    max_intentional_angular_v: float = 150.0# Max angular velocity in controlled recline
    min_intentional_duration: float = 1.0   # Seconds over which intentional posture change occurs
    min_seated_pause_duration: float = 0.30 # Seconds spent in bent/seated posture before reclining
    
    # Temporal history thresholds
    min_upright_duration: float = 0.3       # Min seconds person must be tracked upright before a fall is credible
    max_time_since_upright: float = 1.8     # Max seconds allowed since last upright posture for a fall event
    tracking_lost_reset_time: float = 2.0   # Seconds of missing person before resetting session history


@dataclass
class MotionEvidence:
    """Structured evidence summarizing motion analysis for the current time step."""
    timestamp: float
    current_posture: PostureState
    torso_angle: float
    was_upright_recently: bool
    is_initial_lying: bool                  # Person was already lying when first detected
    has_rapid_descent: bool                 # True ONLY if downward velocity AND hip displacement meet free-fall thresholds
    has_posture_transition: bool           # Angle transitioned from upright toward horizontal
    is_slow_intentional_movement: bool     # Posture changed but kinematics show intentional controlled transition
    is_credible_fall_motion: bool           # True when descent + transition + prior upright match fall mechanics
    downward_velocity: float
    vertical_displacement: float
    delta_angle: float
    angular_velocity: float = 0.0
    confidence: float = 1.0
    rejection_reason: Optional[str] = None
    trigger_reason: Optional[str] = None
    details: Dict[str, Any] = field(default_factory=dict)


class MotionFallDetector:
    """
    Stateful temporal analyzer for motion-based fall event detection.
    
    Maintains rolling temporal history of body positions, detects rapid
    vertical descents, filters out persons who were already lying down upon
    detection, and discriminates intentional reclining from sudden falls.
    """

    def __init__(self, config: Optional[MotionConfig] = None):
        self.config = config or MotionConfig()
        self.reset()

    def reset(self):
        """Resets all history and state tracking."""
        self.last_timestamp: Optional[float] = None
        self.last_detection_timestamp: Optional[float] = None
        self.tracked_duration: float = 0.0
        self.upright_duration: float = 0.0
        self.last_upright_time: Optional[float] = None
        
        # Initial detection state
        self.has_ever_stood_up: bool = False
        self.is_initial_lying: bool = False
        
        # Rolling history buffer: store (timestamp, posture, angle, hip_y, vy, ang_v)
        self.history: deque = deque(maxlen=90)  # ~3.0s at 30 FPS
        
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
        max_angular_velocity: Optional[float] = None,
        p_lying: float = 0.0,
        person_confidence: float = 1.0
    ) -> MotionEvidence:
        """
        Updates the motion detector with current frame/window biomechanical metrics.
        """
        if self.last_timestamp is None:
            self.last_timestamp = timestamp
        dt = max(0.0, timestamp - self.last_timestamp)
        self.last_timestamp = timestamp

        # Reset session if person was missing for longer than tracking_lost_reset_time
        if person_detected:
            if (self.last_detection_timestamp is not None and 
                (timestamp - self.last_detection_timestamp) > self.config.tracking_lost_reset_time):
                # New person or re-entry after prolonged absence -> reset session
                self.has_ever_stood_up = False
                self.is_initial_lying = False
                self.upright_duration = 0.0
                self.last_upright_time = None
                self.history.clear()
            self.last_detection_timestamp = timestamp
        else:
            # Person missing in current frame
            self.current_posture = PostureState.UNKNOWN
            was_upright = False
            if self.has_ever_stood_up and (self.last_upright_time is not None):
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
                angular_velocity=0.0,
                confidence=0.0,
                rejection_reason="person_not_detected",
                details={"reason": "no_person_detected"}
            )

        self.tracked_duration += dt
        self.latest_torso_angle = torso_angle
        posture = self.classify_posture(torso_angle)
        self.current_posture = posture

        ang_v = max_angular_velocity if max_angular_velocity is not None else 0.0
        self.history.append((timestamp, posture, torso_angle, delta_hip_y, downward_velocity, ang_v))

        # -------------------------------------------------------------
        # 1. INITIAL BODY STATE & UPRIGHT HISTORY TRACKING
        # -------------------------------------------------------------
        # If the person has NEVER stood up in this session and is in HORIZONTAL posture:
        if not self.has_ever_stood_up:
            if posture == PostureState.HORIZONTAL or p_lying >= 0.50 or torso_angle >= self.config.horizontal_angle_min:
                self.is_initial_lying = True

        # Track upright posture duration
        if posture == PostureState.UPRIGHT:
            self.upright_duration += dt
            self.last_upright_time = timestamp
            if self.upright_duration >= self.config.min_upright_duration:
                self.has_ever_stood_up = True
                self.is_initial_lying = False  # Confirmed upright; clear initial lying flag
        else:
            self.upright_duration = 0.0

        # Check if person was upright recently
        was_upright_recently = False
        if self.has_ever_stood_up and (self.last_upright_time is not None):
            was_upright_recently = (timestamp - self.last_upright_time) <= self.config.max_time_since_upright

        # -------------------------------------------------------------
        # 2. DOWNWARD MOVEMENT VELOCITY ANALYSIS (STRICT CO-OCCURRENCE)
        # -------------------------------------------------------------
        # Rapid descent requires BOTH sufficient peak downward velocity AND hip displacement
        mean_v = mean_downward_velocity if mean_downward_velocity is not None else (downward_velocity * 0.3)
        has_downward_velocity = downward_velocity >= self.config.min_downward_velocity
        has_hip_displacement = delta_hip_y >= self.config.min_delta_hip_y
        has_positive_mean_v = mean_v >= self.config.min_mean_downward_velocity

        # Crucial fix: require downward velocity AND displacement together!
        has_rapid_descent = has_downward_velocity and has_hip_displacement and has_positive_mean_v

        # -------------------------------------------------------------
        # 3. POSTURE TRANSITION ANALYSIS
        # -------------------------------------------------------------
        peak_angle = max_torso_angle if max_torso_angle is not None else torso_angle
        has_angle_delta = delta_torso_angle >= self.config.min_delta_torso_angle
        reaches_horizontal = peak_angle >= (self.config.horizontal_angle_min - 5.0)

        has_posture_transition = has_angle_delta and reaches_horizontal

        # -------------------------------------------------------------
        # 4. DISCRIMINATE INTENTIONAL LYING DOWN (BED, SOFA, YOGA)
        # -------------------------------------------------------------
        is_intentional = False
        rejection_reason = None

        if self.is_initial_lying:
            is_intentional = True
            rejection_reason = "Person was already lying down upon arrival (never stood up)"
        elif not was_upright_recently:
            is_intentional = True
            rejection_reason = "No upright standing history prior to horizontal posture"
        elif has_posture_transition:
            # Check kinematics over recent history (last 1.5s):
            recent_frames = [f for f in self.history if (timestamp - f[0]) <= 1.5]
            
            # Indicator A: Controlled downward velocity
            is_controlled_v = downward_velocity < self.config.max_intentional_downward_v
            
            # Indicator B: Smooth angular velocity (no violent tipping)
            is_smooth_ang_v = (max_angular_velocity is not None and 
                               max_angular_velocity < self.config.max_intentional_angular_v)
            
            # Indicator C: Intermediate seated / bent stage (sitting on bed before reclining)
            bent_time = sum(0.033 for f in recent_frames if f[1] == PostureState.BENT)
            had_seated_pause = bent_time >= self.config.min_seated_pause_duration
            
            # Indicator D: Decoupled hip descent (torso reclines, but hips remain at bed height)
            is_decoupled_hip = delta_hip_y < (self.config.min_delta_hip_y * 0.9) and delta_torso_angle >= 25.0
            
            # Indicator E: Gradual transition (> 1.0 second from upright to horizontal)
            oldest_upright = [f for f in recent_frames if f[1] == PostureState.UPRIGHT]
            transition_duration = (timestamp - oldest_upright[0][0]) if oldest_upright else 0.0
            is_gradual = transition_duration >= self.config.min_intentional_duration

            # If motion exhibits intentional characteristics and lacks violent impact:
            if is_controlled_v or is_smooth_ang_v or had_seated_pause or is_decoupled_hip or is_gradual:
                if not (downward_velocity >= (self.config.min_downward_velocity * 1.3) and 
                        (max_angular_velocity or 0.0) >= self.config.min_angular_velocity):
                    is_intentional = True
                    rejection_reason = (
                        f"Intentional posture transition (Vy={downward_velocity:.2f}<{self.config.max_intentional_downward_v}, "
                        f"dHipY={delta_hip_y:.2f}, angV={ang_v:.0f}, seatedPause={had_seated_pause})"
                    )

        # If already horizontal and staying horizontal without downward drop
        if posture == PostureState.HORIZONTAL and not has_angle_delta and not has_rapid_descent:
            is_intentional = True
            if not rejection_reason:
                rejection_reason = "Stable horizontal posture without downward descent"

        # -------------------------------------------------------------
        # 5. CREDIBLE FALL MOTION SYNTHESIS
        # -------------------------------------------------------------
        is_credible_fall = (
            person_detected
            and not self.is_initial_lying
            and was_upright_recently
            and has_rapid_descent
            and has_posture_transition
            and not is_intentional
        )

        trigger_reason = None
        if is_credible_fall:
            trigger_reason = (
                f"Credible rapid descent (Vy={downward_velocity:.2f}>={self.config.min_downward_velocity}, "
                f"dHipY={delta_hip_y:.2f}>={self.config.min_delta_hip_y}) + "
                f"posture transition (dAngle={delta_torso_angle:.1f}>={self.config.min_delta_torso_angle}) "
                f"from prior upright stance"
            )

        return MotionEvidence(
            timestamp=timestamp,
            current_posture=posture,
            torso_angle=torso_angle,
            was_upright_recently=was_upright_recently,
            is_initial_lying=self.is_initial_lying,
            has_rapid_descent=has_rapid_descent,
            has_posture_transition=has_posture_transition,
            is_slow_intentional_movement=is_intentional,
            is_credible_fall_motion=is_credible_fall,
            downward_velocity=downward_velocity,
            vertical_displacement=delta_hip_y,
            delta_angle=delta_torso_angle,
            angular_velocity=ang_v,
            confidence=person_confidence,
            rejection_reason=rejection_reason,
            trigger_reason=trigger_reason,
            details={
                "has_ever_stood_up": self.has_ever_stood_up,
                "upright_duration": self.upright_duration,
                "peak_angle": peak_angle,
                "mean_v": mean_v,
                "max_angular_velocity": ang_v
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
            3: delta_torso_angle
            5: max_downward_velocity
            6: mean_vertical_velocity
            7: delta_hip_y
            9: max_angular_velocity
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
        max_ang_v = float(features[9]) if len(features) > 9 else 0.0
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
            max_angular_velocity=max_ang_v,
            p_lying=p_lying,
            person_confidence=conf
        )
