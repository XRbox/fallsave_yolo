"""
FallSave Step 5.3: Standalone Fall Detection State Machine with Motion Validation

Implements the four-state rule-based decision logic:
- NORMAL
- FALLING_SUSPECTED
- GROUND_CHECK
- FALL_CONFIRMED

Integrates:
- ML Classifier probabilities (p_normal, p_falling, p_lying)
- Motion-based fall event evidence (MotionEvidence from motion_detector.py)
- Temporal confirmation and recovery timers

Decoupled from OpenCV, webcam, audio, and UI.
"""

from enum import Enum
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, Tuple

try:
    from motion_detector import MotionEvidence, PostureState
except ImportError:
    MotionEvidence = Any
    PostureState = None


class State(Enum):
    NORMAL = "NORMAL"
    FALLING_SUSPECTED = "FALLING_SUSPECTED"
    GROUND_CHECK = "GROUND_CHECK"
    FALL_CONFIRMED = "FALL_CONFIRMED"


@dataclass
class StateMachineConfig:
    """Configurable thresholds and timers for FallStateMachine."""
    p_fall_threshold: float = 0.45          # Min P(falling) to trigger suspect state
    p_lying_threshold: float = 0.35         # Min P(lying) to indicate grounded posture
    p_normal_recovery: float = 0.65         # Min P(normal) required to count toward recovery
    suspect_timeout: float = 1.5            # Max seconds in FALLING_SUSPECTED without ground evidence
    quick_recovery_time: float = 0.4        # Seconds of sustained normal to clear suspect state quickly
    ground_confirm_time: float = 2.5        # Seconds grounded or undetected to confirm fall
    standup_recovery_time: float = 1.0      # Seconds of sustained normal posture to recover from GROUND_CHECK
    post_confirm_reset_time: float = 5.0    # Seconds of upright posture to reset from FALL_CONFIRMED
    min_tracked_time: float = 0.1           # Min seconds person was actively tracked before allowing suspect fall
    
    # Motion-based event detection configuration (Step 5.3)
    enable_motion_validation: bool = True   # When True and motion_evidence provided, validates motion
    require_upright_before_fall: bool = True # Reject fall if person was not upright prior to event


@dataclass
class StateMachineOutput:
    """Structured output returned on every update tick."""
    state: State
    event: Optional[str]                   # e.g. "ALERT_TRIGGERED", "FALL_SUSPECTED", "RECOVERED", etc.
    elapsed_ground_time: float             # Seconds accumulated in ground check
    recovery_time: float                   # Current continuous recovery seconds
    metadata: Dict[str, Any] = field(default_factory=dict)


class FallStateMachine:
    """
    Robust 4-state fall detection state machine for FallSave.
    
    Operates on streaming predictions: (timestamp, p_normal, p_falling, p_lying, person_detected)
    and optional MotionEvidence from motion_detector.py.
    
    Uses timestamps for true elapsed time, accommodating variable frame rates and dropped frames.
    """
    
    def __init__(self, config: Optional[StateMachineConfig] = None):
        self.config = config or StateMachineConfig()
        self.reset()
        
    def reset(self):
        """Resets all states, timers, and event flags to default initial state."""
        self.current_state = State.NORMAL
        self.state_entry_time: Optional[float] = None
        self.last_timestamp: Optional[float] = None
        self.suspect_start_time: Optional[float] = None
        self.ground_start_time: Optional[float] = None
        self.continuous_recovery_time: float = 0.0
        self.tracked_duration: float = 0.0
        self.had_credible_fall: bool = False
        self.has_alerted: bool = False

    def update(
        self,
        timestamp: float,
        p_normal: float,
        p_falling: float,
        p_lying: float,
        person_detected: bool = True,
        motion_evidence: Optional[Any] = None
    ) -> StateMachineOutput:
        """
        Updates the state machine with a new frame/window observation.
        
        Args:
            timestamp: Monotonic timestamp in seconds.
            p_normal: Classifier probability for Normal (0.0 to 1.0).
            p_falling: Classifier probability for Falling (0.0 to 1.0).
            p_lying: Classifier probability for Lying (0.0 to 1.0).
            person_detected: True if pose was detected, False if person missing/occluded.
            motion_evidence: Optional MotionEvidence instance from MotionFallDetector.
            
        Returns:
            StateMachineOutput containing current state, triggered event (if any), and timers.
        """
        if self.last_timestamp is None:
            self.last_timestamp = timestamp
            self.state_entry_time = timestamp
            
        dt = max(0.0, timestamp - self.last_timestamp)
        self.last_timestamp = timestamp
        
        event: Optional[str] = None
        
        # Track person presence in NORMAL state
        if person_detected:
            self.tracked_duration += dt
        else:
            self.tracked_duration = 0.0

        # Track continuous normal recovery
        # Recovery requires either high P(normal) or confirmed upright posture
        is_recovering = person_detected and (
            p_normal >= self.config.p_normal_recovery
            or (motion_evidence is not None and getattr(motion_evidence, 'current_posture', None) == getattr(PostureState, 'UPRIGHT', None))
        )
        
        if is_recovering:
            self.continuous_recovery_time += dt
        else:
            self.continuous_recovery_time = 0.0
            
        # -------------------------------------------------------------
        # STATE: NORMAL
        # -------------------------------------------------------------
        transition_reason = None
        rejection_reason = None

        if self.current_state == State.NORMAL:
            self.continuous_recovery_time = 0.0
            self.ground_start_time = None
            self.had_credible_fall = False
            
            allow_suspect = False
            
            if person_detected:
                if motion_evidence is not None and self.config.enable_motion_validation:
                    # Strict Rejection Filters:
                    # 1. Person was already lying down upon arrival (never stood up)
                    # 2. Movement was intentional / controlled (reclining on bed/sofa, yoga)
                    # 3. Person was never tracked upright prior to suspected event
                    # 4. Movement lacked rapid downward descent (velocity < threshold)
                    if motion_evidence.is_initial_lying:
                        is_rejected = True
                        rejection_reason = "Person was already lying down upon arrival (never stood up)"
                    elif motion_evidence.is_slow_intentional_movement:
                        is_rejected = True
                        rejection_reason = motion_evidence.rejection_reason or "Intentional controlled posture transition"
                    elif self.config.require_upright_before_fall and not motion_evidence.was_upright_recently:
                        is_rejected = True
                        rejection_reason = "No upright standing history prior to horizontal posture"
                    elif not motion_evidence.has_rapid_descent:
                        is_rejected = True
                        rejection_reason = "No rapid downward descent detected (Vy < min threshold)"
                    else:
                        is_rejected = False
                    
                    if not is_rejected:
                        # An actual fall requires BOTH rapid downward descent AND posture transition
                        has_valid_kinematics = (
                            motion_evidence.has_rapid_descent
                            and motion_evidence.has_posture_transition
                        )
                        has_ml_suspicion = (p_falling >= self.config.p_fall_threshold) and has_valid_kinematics
                        is_motion_fall = motion_evidence.is_credible_fall_motion
                        
                        allow_suspect = (
                            (has_ml_suspicion or is_motion_fall)
                            and (self.tracked_duration >= self.config.min_tracked_time)
                        )
                        if allow_suspect:
                            transition_reason = motion_evidence.trigger_reason or "Rapid downward descent + posture transition"
                    else:
                        allow_suspect = False
                else:
                    # Fallback / legacy path when motion evidence is not provided
                    allow_suspect = (
                        (p_falling >= self.config.p_fall_threshold)
                        and (self.tracked_duration >= self.config.min_tracked_time)
                    )
                    if allow_suspect:
                        transition_reason = f"ML P(falling)={p_falling:.2f} >= threshold"

            if allow_suspect:
                self._transition_to(State.FALLING_SUSPECTED, timestamp)
                self.suspect_start_time = timestamp
                self.had_credible_fall = True
                event = "FALL_SUSPECTED"
                
        # -------------------------------------------------------------
        # STATE: FALLING_SUSPECTED
        # -------------------------------------------------------------
        elif self.current_state == State.FALLING_SUSPECTED:
            time_in_suspect = timestamp - (self.suspect_start_time or timestamp)
            
            # Condition A: Ground contact detected (via ML lying probability or horizontal posture)
            is_lying = person_detected and (
                (p_lying >= self.config.p_lying_threshold)
                or (motion_evidence is not None and getattr(motion_evidence, 'current_posture', None) == getattr(PostureState, 'HORIZONTAL', None))
            )

            # Condition B: Person disappeared AFTER a credible tracked fall
            # (Matches empirical finding where YOLO loses pose upon floor impact)
            is_lost_after_fall = (not person_detected) and self.had_credible_fall
            
            if is_lying or is_lost_after_fall:
                self._transition_to(State.GROUND_CHECK, timestamp)
                self.ground_start_time = timestamp
                self.continuous_recovery_time = 0.0
                event = "GROUND_CHECK_ENTERED"
                transition_reason = (
                    "Ground contact detected (P_lying/horizontal) after suspect fall"
                    if is_lying else "Person lost from camera after credible fall"
                )
                
            # Condition C: Fast recovery (person regained upright posture)
            elif self.continuous_recovery_time >= self.config.quick_recovery_time:
                self._transition_to(State.NORMAL, timestamp)
                self.had_credible_fall = False
                event = "SUSPECT_CLEARED"
                
            # Condition D: Suspect timeout (no ground evidence, no further fall motion)
            elif time_in_suspect >= self.config.suspect_timeout:
                self._transition_to(State.NORMAL, timestamp)
                self.had_credible_fall = False
                event = "SUSPECT_TIMEOUT"
                
        # -------------------------------------------------------------
        # STATE: GROUND_CHECK
        # -------------------------------------------------------------
        elif self.current_state == State.GROUND_CHECK:
            ground_elapsed = timestamp - (self.ground_start_time or timestamp)
            
            # Condition A: Self-recovery (person stood up and remained upright)
            if self.continuous_recovery_time >= self.config.standup_recovery_time:
                self._transition_to(State.NORMAL, timestamp)
                self.ground_start_time = None
                self.had_credible_fall = False
                event = "RECOVERED"
                
            # Condition B: Sustained ground presence / undetected on floor beyond confirm limit
            elif ground_elapsed >= self.config.ground_confirm_time:
                self._transition_to(State.FALL_CONFIRMED, timestamp)
                self.has_alerted = True
                event = "ALERT_TRIGGERED"  # Triggered EXACTLY ONCE on transition tick
                
        # -------------------------------------------------------------
        # STATE: FALL_CONFIRMED
        # -------------------------------------------------------------
        elif self.current_state == State.FALL_CONFIRMED:
            # Alert is already active; do not re-emit "ALERT_TRIGGERED"
            # Condition: Person stood up stably after confirmed fall for reset time
            if self.continuous_recovery_time >= self.config.post_confirm_reset_time:
                self._transition_to(State.NORMAL, timestamp)
                self.has_alerted = False
                self.ground_start_time = None
                self.had_credible_fall = False
                event = "POST_FALL_RESET"
                
        elapsed_ground = (
            timestamp - self.ground_start_time
            if (self.ground_start_time is not None and self.current_state in (State.GROUND_CHECK, State.FALL_CONFIRMED))
            else 0.0
        )
        
        meta = {
            "p_normal": p_normal,
            "p_falling": p_falling,
            "p_lying": p_lying,
            "person_detected": person_detected,
            "time_in_state": timestamp - (self.state_entry_time or timestamp),
            "has_alerted": self.has_alerted,
            "motion_evidence": motion_evidence,
            "transition_reason": transition_reason,
            "rejection_reason": rejection_reason
        }
        
        return StateMachineOutput(
            state=self.current_state,
            event=event,
            elapsed_ground_time=elapsed_ground,
            recovery_time=self.continuous_recovery_time,
            metadata=meta
        )

    def _transition_to(self, new_state: State, timestamp: float):
        """Helper to manage clean state transitions and entry timestamps."""
        self.current_state = new_state
        self.state_entry_time = timestamp
