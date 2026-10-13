"""
FallSave Step 4.2: Standalone Fall Detection State Machine

Implements the four-state rule-based decision logic:
- NORMAL
- FALLING_SUSPECTED
- GROUND_CHECK
- FALL_CONFIRMED

Completely decoupled from OpenCV, webcam, audio, and UI.
Driven by classifier probabilities, detection status, and timestamps.
"""

from enum import Enum
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, Tuple


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
    
    Operates on streaming predictions: (timestamp, p_normal, p_falling, p_lying, person_detected).
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
        self.has_alerted: bool = False

    def update(
        self,
        timestamp: float,
        p_normal: float,
        p_falling: float,
        p_lying: float,
        person_detected: bool = True
    ) -> StateMachineOutput:
        """
        Updates the state machine with a new frame/window observation.
        
        Args:
            timestamp: Monotonic timestamp in seconds.
            p_normal: Classifier probability for Normal (0.0 to 1.0).
            p_falling: Classifier probability for Falling (0.0 to 1.0).
            p_lying: Classifier probability for Lying (0.0 to 1.0).
            person_detected: True if pose was detected, False if person missing/occluded.
            
        Returns:
            StateMachineOutput containing current state, triggered event (if any), and timers.
        """
        if self.last_timestamp is None:
            self.last_timestamp = timestamp
            self.state_entry_time = timestamp
            
        dt = max(0.0, timestamp - self.last_timestamp)
        self.last_timestamp = timestamp
        
        event: Optional[str] = None
        
        # Track continuous normal recovery
        if person_detected and (p_normal >= self.config.p_normal_recovery):
            self.continuous_recovery_time += dt
        else:
            self.continuous_recovery_time = 0.0
            
        # -------------------------------------------------------------
        # STATE: NORMAL
        # -------------------------------------------------------------
        if self.current_state == State.NORMAL:
            self.continuous_recovery_time = 0.0
            self.ground_start_time = None
            
            if person_detected and (p_falling >= self.config.p_fall_threshold):
                self._transition_to(State.FALLING_SUSPECTED, timestamp)
                self.suspect_start_time = timestamp
                event = "FALL_SUSPECTED"
                
        # -------------------------------------------------------------
        # STATE: FALLING_SUSPECTED
        # -------------------------------------------------------------
        elif self.current_state == State.FALLING_SUSPECTED:
            time_in_suspect = timestamp - (self.suspect_start_time or timestamp)
            
            # Condition A: Ground contact detected OR person disappeared right after suspect fall
            # (Matches empirical finding where YOLO loses pose upon floor impact)
            is_lying = person_detected and (p_lying >= self.config.p_lying_threshold)
            is_lost_after_fall = not person_detected
            
            if is_lying or is_lost_after_fall:
                self._transition_to(State.GROUND_CHECK, timestamp)
                self.ground_start_time = timestamp
                self.continuous_recovery_time = 0.0
                event = "GROUND_CHECK_ENTERED"
                
            # Condition B: Fast recovery (person regained upright posture)
            elif self.continuous_recovery_time >= self.config.quick_recovery_time:
                self._transition_to(State.NORMAL, timestamp)
                event = "SUSPECT_CLEARED"
                
            # Condition C: Suspect timeout (no ground evidence, no further fall motion)
            elif time_in_suspect >= self.config.suspect_timeout:
                self._transition_to(State.NORMAL, timestamp)
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
            "has_alerted": self.has_alerted
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
