"""
FallSave Step 5.3: Live AI Fall Detection Pipeline with Motion Validation

Integrates:
1. Video/Webcam capture
2. YOLO11n-Pose keypoint extraction (17 joints)
3. LiveFeatureBuffer matching exact training feature schema & order
4. StandardScaler (loaded from output/feature_scaler.joblib)
5. Logistic Regression baseline (loaded from models/logistic_regression_balanced.joblib)
6. MotionFallDetector (Step 5.3: Motion-based downward velocity, posture transition, & history)
7. FallStateMachine (4-state temporal decision logic with motion validation)
8. AlertHandler (Audio beep alert & timestamped snapshot saving)
9. On-screen real-time HUD and debug overlay
"""

import os
import sys
import time
import argparse
from collections import deque
from typing import Optional, Tuple, Dict, Any, List

import cv2
import joblib
import numpy as np
import pandas as pd
from ultralytics import YOLO

from fall_state_machine import FallStateMachine, StateMachineConfig, State, StateMachineOutput
from alert_handler import AlertHandler
from motion_detector import MotionFallDetector, MotionConfig, PostureState, MotionEvidence

# -------------------------------------------------------------
# Paths & Schema Constants (Matching Steps 2.3 - 3.2 exactly)
# -------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
YOLO_MODEL_PATH = os.path.join(BASE_DIR, "yolo11n-pose.pt")
SCALER_PATH = os.path.join(BASE_DIR, "output", "feature_scaler.joblib")
CLASSIFIER_PATH = os.path.join(BASE_DIR, "models", "logistic_regression_balanced.joblib")

WINDOW_LENGTH = 20          # 20 frames (~0.67s at 30 FPS)
CONF_THRESHOLD = 0.20       # Keypoint confidence threshold
INTERP_LIMIT = 3            # Micro-gap interpolation limit
ASSUMED_FPS = 30.0

FEATURE_COLS = [
    "mean_torso_angle",
    "max_torso_angle",
    "min_torso_angle",
    "delta_torso_angle",
    "std_torso_angle",
    "max_downward_velocity",
    "mean_vertical_velocity",
    "delta_hip_y",
    "max_hip_y",
    "max_angular_velocity",
    "mean_angular_velocity",
    "mean_torso_span_y",
    "delta_torso_span_y",
    "mean_kp_aspect_ratio",
    "delta_kp_aspect_ratio",
    "missing_ratio",
    "mean_person_confidence"
]


class LiveFeatureBuffer:
    """
    Sliding FIFO buffer that maintains the last 20 frames of keypoints
    and extracts temporal features using the exact math from Step 2.3.
    """
    def __init__(self, window_length: int = WINDOW_LENGTH, fps: float = ASSUMED_FPS):
        self.window_length = window_length
        self.fps = fps
        self.buffer = deque(maxlen=window_length)

    def append(
        self,
        person_detected: bool,
        person_confidence: float,
        kps: Optional[np.ndarray]  # shape (17, 3): x, y, conf
    ):
        """Appends one frame's keypoint observation to the buffer."""
        if person_detected and kps is not None and len(kps) == 17:
            frame_entry = {
                "detected": 1,
                "confidence": float(person_confidence),
                "kps": kps.copy()
            }
        else:
            # Missing detection entry
            frame_entry = {
                "detected": 0,
                "confidence": 0.0,
                "kps": np.full((17, 3), np.nan)
            }
        self.buffer.append(frame_entry)

    def is_ready(self) -> bool:
        """Returns True once the buffer is full (20 frames)."""
        return len(self.buffer) == self.window_length

    def get_detected_count(self) -> int:
        """Returns the number of frames in the buffer with valid person detection."""
        return sum(1 for item in self.buffer if item["detected"] == 1)

    def get_missing_ratio(self) -> float:
        """Returns the ratio of missing person frames in the current buffer."""
        if not self.buffer:
            return 1.0
        return 1.0 - (self.get_detected_count() / len(self.buffer))

    def extract_features(self) -> np.ndarray:
        """
        Computes the 17 temporal features across the current buffer.
        Replicates the exact preprocessing of build_feature_dataset.py.
        """
        assert self.is_ready(), "Buffer is not yet full!"

        # 1. Build DataFrame representation for vector operations
        data = []
        for item in self.buffer:
            row = {
                "person_detected": item["detected"],
                "person_confidence": item["confidence"]
            }
            kps = item["kps"]
            for i in range(17):
                row[f"kp_{i}_x"] = kps[i, 0]
                row[f"kp_{i}_y"] = kps[i, 1]
                row[f"kp_{i}_confidence"] = kps[i, 2]
            data.append(row)

        df = pd.DataFrame(data)

        # 2. Mask low confidence keypoints (< CONF_THRESHOLD) to NaN
        for i in range(17):
            mask = df[f"kp_{i}_confidence"] < CONF_THRESHOLD
            df.loc[mask, f"kp_{i}_x"] = np.nan
            df.loc[mask, f"kp_{i}_y"] = np.nan

        # 3. Linear interpolate micro-gaps <= INTERP_LIMIT within buffer
        x_cols = [f"kp_{i}_x" for i in range(17)]
        y_cols = [f"kp_{i}_y" for i in range(17)]
        df[x_cols] = df[x_cols].interpolate(method="linear", limit=INTERP_LIMIT)
        df[y_cols] = df[y_cols].interpolate(method="linear", limit=INTERP_LIMIT)

        # 4. Core Torso Joints (Mid-Shoulder 5,6 and Mid-Hip 11,12)
        sh_x = df[["kp_5_x", "kp_6_x"]].mean(axis=1)
        sh_y = df[["kp_5_y", "kp_6_y"]].mean(axis=1)
        hip_x = df[["kp_11_x", "kp_12_x"]].mean(axis=1)
        hip_y = df[["kp_11_y", "kp_12_y"]].mean(axis=1)

        dx = sh_x - hip_x
        dy = sh_y - hip_y
        torso_len = np.sqrt(dx**2 + dy**2)

        valid_lengths = torso_len.dropna()
        scale = float(valid_lengths.median()) if len(valid_lengths) > 0 else 50.0
        scale = max(scale, 15.0)

        # Torso Angle (0 deg = upright, 90 deg = horizontal, >90 = inverted)
        torso_angle = np.degrees(np.arctan2(np.abs(dx), np.abs(dy)))
        inverted = sh_y > hip_y
        torso_angle = np.where(inverted, 180.0 - torso_angle, torso_angle)

        # Normalized coordinates & signals
        hip_y_norm = hip_y / scale
        hip_v_y = hip_y_norm.diff() * self.fps
        ang_v = pd.Series(torso_angle).diff().abs() * self.fps
        torso_span_y = np.abs(dy) / scale

        kp_xs = df[x_cols]
        kp_ys = df[y_cols]
        kp_w = kp_xs.max(axis=1) - kp_xs.min(axis=1)
        kp_h = kp_ys.max(axis=1) - kp_ys.min(axis=1)
        kp_aspect_ratio = kp_w / np.maximum(kp_h, 15.0)

        torso_missing = hip_y.isna() | sh_y.isna()
        missing_ratio = float(torso_missing.sum() / self.window_length)

        # Summary statistics over the window
        w_angle = pd.Series(torso_angle).dropna()
        w_hip_y = hip_y_norm.dropna()
        w_hip_v = hip_v_y.dropna()
        w_ang_v = ang_v.dropna()
        w_span_y = torso_span_y.dropna()
        w_ar = kp_aspect_ratio.dropna()
        w_conf = df["person_confidence"]

        mean_angle = float(w_angle.mean()) if len(w_angle) else 0.0
        max_angle = float(w_angle.max()) if len(w_angle) else 0.0
        min_angle = float(w_angle.min()) if len(w_angle) else 0.0
        delta_angle = float(w_angle.iloc[-1] - w_angle.iloc[0]) if len(w_angle) >= 2 else 0.0
        std_angle = float(w_angle.std()) if len(w_angle) >= 2 else 0.0

        max_drop_v = float(w_hip_v.max()) if len(w_hip_v) else 0.0
        mean_v = float(w_hip_v.mean()) if len(w_hip_v) else 0.0

        delta_hip_y = float(w_hip_y.iloc[-1] - w_hip_y.iloc[0]) if len(w_hip_y) >= 2 else 0.0
        max_hip_y = float(w_hip_y.max()) if len(w_hip_y) else 0.0

        max_ang_v = float(w_ang_v.max()) if len(w_ang_v) else 0.0
        mean_ang_v = float(w_ang_v.mean()) if len(w_ang_v) else 0.0

        mean_span_y = float(w_span_y.mean()) if len(w_span_y) else 0.0
        delta_span_y = float(w_span_y.iloc[-1] - w_span_y.iloc[0]) if len(w_span_y) >= 2 else 0.0

        mean_ar = float(w_ar.mean()) if len(w_ar) else 0.0
        delta_ar = float(w_ar.iloc[-1] - w_ar.iloc[0]) if len(w_ar) >= 2 else 0.0

        mean_conf = float(w_conf.mean())

        feature_vector = np.array([
            mean_angle,
            max_angle,
            min_angle,
            delta_angle,
            std_angle,
            max_drop_v,
            mean_v,
            delta_hip_y,
            max_hip_y,
            max_ang_v,
            mean_ang_v,
            mean_span_y,
            delta_span_y,
            mean_ar,
            delta_ar,
            missing_ratio,
            mean_conf
        ], dtype=float)

        return feature_vector


def draw_hud(
    frame: np.ndarray,
    state: State,
    p_normal: float,
    p_falling: float,
    p_lying: float,
    person_detected: bool,
    elapsed_ground: float,
    confirm_limit: float,
    fps: float,
    alert_active: bool = False,
    motion_evidence: Optional[MotionEvidence] = None
):
    """Renders debug HUD overlay on the display frame."""
    h, w = frame.shape[:2]

    # Colors (BGR)
    colors = {
        State.NORMAL: (50, 180, 50),            # Green
        State.FALLING_SUSPECTED: (30, 200, 240), # Yellow/Orange
        State.GROUND_CHECK: (200, 50, 200),      # Magenta/Purple
        State.FALL_CONFIRMED: (40, 40, 230),     # Red
    }
    state_color = colors.get(state, (200, 200, 200))

    # Top Status Banner
    cv2.rectangle(frame, (0, 0), (w, 50), state_color, -1)
    banner_text = f"STATE: {state.value}"
    if alert_active:
        banner_text += " [ALERT ACTIVE!]"
    cv2.putText(frame, banner_text, (15, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)

    # Detection status & FPS
    det_str = "PERSON DETECTED" if person_detected else "NO PERSON DETECTED"
    det_color = (0, 255, 0) if person_detected else (0, 0, 255)
    cv2.putText(frame, det_str, (w - 280, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.55, det_color, 2)
    cv2.putText(frame, f"FPS: {fps:.1f}", (w - 100, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (220, 220, 220), 1)

    # Semi-transparent overlay box for probabilities and motion
    box_w = 340
    box_h = 165 if (alert_active or motion_evidence is not None) else 110
    overlay = frame.copy()
    cv2.rectangle(overlay, (10, 60), (10 + box_w, 60 + box_h), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.65, frame, 0.35, 0, frame)

    # Probabilities Text
    cv2.putText(frame, f"P(Normal)  : {p_normal:.2f}", (20, 85), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 255, 200), 1)
    cv2.putText(frame, f"P(Falling) : {p_falling:.2f}", (20, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (100, 220, 255), 1)
    cv2.putText(frame, f"P(Lying)   : {p_lying:.2f}", (20, 135), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 180, 255), 1)

    # Step 5.3 Motion Evidence Overlay
    if motion_evidence is not None and motion_evidence.current_posture != PostureState.UNKNOWN:
        motion_tag = "FALL-MOTION" if motion_evidence.is_credible_fall_motion else "STABLE"
        if motion_evidence.is_initial_lying:
            motion_tag = "ALREADY-LYING"
        elif motion_evidence.is_slow_intentional_movement:
            motion_tag = "SLOW-INTENTIONAL"
        posture_str = f"Posture: {motion_evidence.current_posture.value} ({motion_evidence.torso_angle:.0f}deg) | [{motion_tag}]"
        cv2.putText(frame, posture_str, (20, 160), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 220, 150), 1)

    # Ground Check countdown/timer or Alert notification
    if state in (State.GROUND_CHECK, State.FALL_CONFIRMED):
        timer_text = f"Ground Timer: {elapsed_ground:.1f}s / {confirm_limit:.1f}s"
        cv2.putText(frame, timer_text, (20, 185 if motion_evidence else 160), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2)

    if alert_active:
        cv2.putText(frame, "SNAPSHOT SAVED & ALARM ON", (20, 210 if motion_evidence else 185), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 140, 255), 2)

    # Controls instruction
    cv2.putText(frame, "Press 'q' to quit | 'r' to reset", (15, h - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1)


def run_pipeline(
    camera_index: int = 0,
    sequence_path: Optional[str] = None,
    video_path: Optional[str] = None,
    headless: bool = False,
    max_frames: Optional[int] = None,
    yolo_conf: float = 0.25,
    hold_last_seconds: float = 0.0,
    enable_sound: bool = True,
    alert_dir: str = "output/alerts",
    enable_motion_validation: bool = True,
    min_downward_velocity: float = 2.0,
    ground_confirm_time: float = 2.5
):
    print("=" * 65)
    print("   FALLSAVE STEP 5.3: LIVE AI FALL DETECTION PIPELINE   ")
    print("=" * 65)

    # 1. Load Pretrained Models & Artifacts
    print(f"[1/5] Loading YOLO11n-Pose from: {YOLO_MODEL_PATH}")
    if not os.path.exists(YOLO_MODEL_PATH):
        raise FileNotFoundError(f"Missing YOLO model checkpoint at: {YOLO_MODEL_PATH}")
    yolo_model = YOLO(YOLO_MODEL_PATH)

    print(f"[2/5] Loading StandardScaler from: {SCALER_PATH}")
    if not os.path.exists(SCALER_PATH):
        raise FileNotFoundError(f"Missing feature scaler at: {SCALER_PATH}")
    scaler = joblib.load(SCALER_PATH)

    print(f"[3/5] Loading Classifier from: {CLASSIFIER_PATH}")
    if not os.path.exists(CLASSIFIER_PATH):
        raise FileNotFoundError(f"Missing classifier baseline at: {CLASSIFIER_PATH}")
    classifier = joblib.load(CLASSIFIER_PATH)

    print(f"[4/5] Initializing MotionFallDetector, FallStateMachine & LiveFeatureBuffer")
    motion_config = MotionConfig(
        min_downward_velocity=min_downward_velocity
    )
    motion_detector = MotionFallDetector(motion_config)

    fsm_config = StateMachineConfig(
        p_fall_threshold=0.45,
        p_lying_threshold=0.35,
        p_normal_recovery=0.65,
        suspect_timeout=1.5,
        quick_recovery_time=0.4,
        ground_confirm_time=ground_confirm_time,
        standup_recovery_time=1.0,
        post_confirm_reset_time=5.0,
        enable_motion_validation=enable_motion_validation
    )
    fsm = FallStateMachine(fsm_config)
    feature_buffer = LiveFeatureBuffer(window_length=WINDOW_LENGTH, fps=ASSUMED_FPS)

    print(f"[5/5] Initializing AlertHandler (sound={'ENABLED' if enable_sound else 'DISABLED'}, alert_dir='{alert_dir}')")
    alert_handler = AlertHandler(
        output_dir=alert_dir,
        enable_sound=enable_sound,
        async_sound=True
    )

    # 2. Setup Video / Webcam Source
    is_sequence = sequence_path is not None
    is_video = video_path is not None
    cap = None
    image_files = []
    source_fps = ASSUMED_FPS

    if is_sequence:
        print(f"\n[SOURCE] Running in Sequence Simulation mode: {sequence_path}")
        if not os.path.exists(sequence_path):
            raise FileNotFoundError(f"Sequence directory not found: {sequence_path}")
        image_files = sorted([
            os.path.join(sequence_path, f) for f in os.listdir(sequence_path)
            if f.lower().endswith(('.png', '.jpg', '.jpeg'))
        ])
        if not image_files:
            raise ValueError(f"No image files found in sequence directory: {sequence_path}")
        print(f"Loaded {len(image_files)} sequence frames for simulation.")
    elif is_video:
        print(f"\n[SOURCE] Running with Video File: {video_path}")
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video file not found: {video_path}")
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise RuntimeError(f"Failed to open video file: {video_path}")
        vid_fps = cap.get(cv2.CAP_PROP_FPS)
        source_fps = float(vid_fps) if vid_fps and vid_fps > 5.0 else ASSUMED_FPS
        feature_buffer.fps = source_fps
        print(f"Video opened successfully (detected FPS: {source_fps:.1f})")
    else:
        print(f"\n[SOURCE] Opening webcam at index: {camera_index}")
        cap = cv2.VideoCapture(camera_index)
        if not cap.isOpened():
            err_msg = (
                f"[ERROR] Failed to open camera device index {camera_index}.\n"
                "  - No physical webcam is available in this environment.\n"
                "  - Tip: You can test the end-to-end AI pipeline using a sequence folder via:\n"
                "         python live_fall_detection.py --sequence dataset/fall/fall-01-cam0-rgb\n"
                "    Or with a video file via:\n"
                "         python live_fall_detection.py --video path_to_video.mp4"
            )
            print(err_msg, file=sys.stderr)
            return False, err_msg

    print("\nStarting detection loop... Press 'q' to quit.")
    frame_idx = 0
    t_start = time.time()
    last_frame_time = t_start
    alert_events_logged = 0

    try:
        while True:
            current_time = time.time() - t_start
            dt = max(0.001, current_time - (last_frame_time - t_start))
            last_frame_time = time.time()
            fps_live = 1.0 / dt

            # Read frame
            if is_sequence:
                extra_hold_frames = int(hold_last_seconds * ASSUMED_FPS)
                total_sim_frames = len(image_files) + extra_hold_frames
                if frame_idx >= total_sim_frames:
                    print("[INFO] Reached end of test sequence (including post-fall hold).")
                    break
                # If beyond sequence length, hold last frame
                actual_file_idx = min(frame_idx, len(image_files) - 1)
                raw_bytes = np.fromfile(image_files[actual_file_idx], dtype=np.uint8)
                frame = cv2.imdecode(raw_bytes, cv2.IMREAD_COLOR)
                sim_timestamp = frame_idx / ASSUMED_FPS
                active_timestamp = sim_timestamp
            else:
                ret, frame = cap.read()
                if not ret or frame is None:
                    print("[INFO] Video stream ended or failed to grab frame.")
                    break
                active_timestamp = (frame_idx / source_fps) if is_video else current_time

            frame_idx += 1
            if max_frames and frame_idx > max_frames:
                print(f"[INFO] Reached max_frames limit ({max_frames}). Stopping.")
                break

            # 3. YOLO11n-Pose Keypoint Extraction
            yolo_results = yolo_model.predict(
                source=frame,
                conf=yolo_conf,
                verbose=False
            )[0]

            person_detected = False
            person_confidence = 0.0
            primary_kps = None

            if yolo_results.keypoints is not None and len(yolo_results.keypoints.data) > 0:
                all_kps = yolo_results.keypoints.data.cpu().numpy()
                if yolo_results.boxes is not None and len(yolo_results.boxes) > 0:
                    confs = yolo_results.boxes.conf.cpu().numpy()
                    best_person_idx = int(confs.argmax())
                    person_confidence = float(confs[best_person_idx])
                else:
                    best_person_idx = 0
                    person_confidence = 0.5

                primary_kps = all_kps[best_person_idx]
                person_detected = True

            # 4. Push to LiveFeatureBuffer
            feature_buffer.append(
                person_detected=person_detected,
                person_confidence=person_confidence,
                kps=primary_kps
            )

            # 5. Extract Features & Predict Probabilities
            # A window is valid for AI classification ONLY IF:
            # 1. The buffer is full (20 frames)
            # 2. A person is actively detected in the current frame
            # 3. Missing ratio in buffer <= 0.20 (within trained feature distribution)
            raw_feat = None
            if (
                feature_buffer.is_ready()
                and person_detected
                and (feature_buffer.get_missing_ratio() <= 0.20)
            ):
                raw_feat = feature_buffer.extract_features()
                feat_df = pd.DataFrame([raw_feat], columns=FEATURE_COLS)
                scaled_feat = scaler.transform(feat_df)
                probs = classifier.predict_proba(scaled_feat)[0]
                p_norm = float(probs[0])
                p_fall = float(probs[1])
                p_lie = float(probs[2])
            else:
                # When no person is detected or buffer has > 20% missing frames:
                # Do NOT pass out-of-distribution empty keypoints to the classifier.
                p_norm = 1.0 if not person_detected else 0.8
                p_fall = 0.0
                p_lie = 0.0

            # 6. Step 5.3: Motion-Based Fall Event Analysis
            if raw_feat is not None:
                motion_evidence = motion_detector.analyze_feature_vector(
                    features=raw_feat,
                    timestamp=active_timestamp,
                    person_detected=person_detected,
                    p_lying=p_lie
                )
            else:
                motion_evidence = motion_detector.update(
                    timestamp=active_timestamp,
                    torso_angle=0.0,
                    downward_velocity=0.0,
                    delta_hip_y=0.0,
                    delta_torso_angle=0.0,
                    person_detected=person_detected,
                    p_lying=p_lie
                )

            # 7. Update FallStateMachine with ML probabilities and motion evidence
            fsm_out = fsm.update(
                timestamp=active_timestamp,
                p_normal=p_norm,
                p_falling=p_fall,
                p_lying=p_lie,
                person_detected=person_detected,
                motion_evidence=motion_evidence
            )

            # 8. Render Preview Base Frame
            display_frame = yolo_results.plot() if person_detected else frame.copy()

            # 9. Check & Log Events
            if fsm_out.event == "ALERT_TRIGGERED":
                alert_events_logged += 1
                alert_res = alert_handler.trigger_alert(
                    frame=display_frame,
                    timestamp=active_timestamp
                )
                print(f"\n>>> [ALERT_TRIGGERED] at t={active_timestamp:.2f}s (Frame #{frame_idx})!")
                print(f"    State transitioned to: {fsm_out.state.value}")
                print(f"    Ground Duration: {fsm_out.elapsed_ground_time:.2f}s | Probabilities: N={p_norm:.2f}, F={p_fall:.2f}, L={p_lie:.2f}")
                if alert_res.get("snapshot_saved"):
                    print(f"    [SNAPSHOT] Saved to: {alert_res['snapshot_path']}")
                if alert_res.get("sound_played"):
                    print(f"    [AUDIO] Fall alert sound dispatched.")
                print()
            elif fsm_out.event in ("RECOVERED", "POST_FALL_RESET", "SUSPECT_CLEARED", "SUSPECT_TIMEOUT"):
                print(f"[EVENT] {fsm_out.event} at t={active_timestamp:.2f}s (State: {fsm_out.state.value})")
                alert_handler.reset()
            elif fsm_out.event:
                print(f"[EVENT] {fsm_out.event} at t={active_timestamp:.2f}s (State: {fsm_out.state.value})")

            # 10. Render HUD Overlay with Motion Indicators
            draw_hud(
                frame=display_frame,
                state=fsm_out.state,
                p_normal=p_norm,
                p_falling=p_fall,
                p_lying=p_lie,
                person_detected=person_detected,
                elapsed_ground=fsm_out.elapsed_ground_time,
                confirm_limit=fsm_config.ground_confirm_time,
                fps=fps_live,
                alert_active=alert_handler.is_alert_active,
                motion_evidence=motion_evidence
            )

            if not headless:
                cv2.imshow("FallSave Live Detection", display_frame)
                wait_time = max(1, int(1000.0 / source_fps)) if is_video else 1
                key = cv2.waitKey(wait_time) & 0xFF
                if key == ord('q'):
                    print("[INFO] User pressed 'q'. Exiting...")
                    break
                elif key == ord('r'):
                    print("[INFO] User pressed 'r'. Resetting State Machine, MotionDetector, & AlertHandler.")
                    fsm.reset()
                    motion_detector.reset()
                    alert_handler.reset()

        print(f"\nExecution finished. Processed {frame_idx} frames. Total ALERT events: {alert_events_logged}.")
        return True, "Pipeline executed successfully."

    finally:
        if cap is not None:
            cap.release()
        if not headless:
            cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser(description="FallSave Live Detection Pipeline")
    parser.add_argument("--camera", type=int, default=0, help="Webcam device index (default: 0)")
    parser.add_argument("--video", type=str, default=None, help="Path to video file (.mp4, .avi, etc.)")
    parser.add_argument("--sequence", type=str, default=None, help="Directory containing sequence image frames for simulation")
    parser.add_argument("--headless", action="store_true", help="Run without cv2.imshow GUI display")
    parser.add_argument("--max-frames", type=int, default=None, help="Stop after processing max frames")
    parser.add_argument("--conf", type=float, default=0.25, help="YOLO person confidence threshold (default: 0.25)")
    parser.add_argument("--hold-last-frame", type=float, default=0.0, help="Hold last frame for N seconds in sequence simulation mode")
    parser.add_argument("--no-sound", action="store_true", help="Disable alert beep sound")
    parser.add_argument("--alert-dir", type=str, default="output/alerts", help="Directory to save snapshot images on alert (default: output/alerts)")
    parser.add_argument("--min-downward-velocity", type=float, default=2.0, help="Min peak downward velocity (scale units/s) for fall motion (default: 2.0)")
    parser.add_argument("--ground-confirm-time", type=float, default=2.5, help="Seconds on ground to confirm fall alert (default: 2.5)")
    parser.add_argument("--no-motion-validation", action="store_true", help="Disable Step 5.3 motion-based validation and use ML-only fallback")

    args = parser.parse_args()

    success, msg = run_pipeline(
        camera_index=args.camera,
        sequence_path=args.sequence,
        video_path=args.video,
        headless=args.headless,
        max_frames=args.max_frames,
        yolo_conf=args.conf,
        hold_last_seconds=args.hold_last_frame,
        enable_sound=not args.no_sound,
        alert_dir=args.alert_dir,
        enable_motion_validation=not args.no_motion_validation,
        min_downward_velocity=args.min_downward_velocity,
        ground_confirm_time=args.ground_confirm_time
    )

    if not success:
        sys.exit(1)


if __name__ == "__main__":
    main()
