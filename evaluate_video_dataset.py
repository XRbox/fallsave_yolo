"""
FallSave: VideoDataset Benchmark Evaluation Pipeline

Evaluates the end-to-end FallSave hybrid detection pipeline
(YOLO11n-Pose + Biomechanical Features + Balanced Logistic Regression + FallStateMachine)
against real-world video clips in the VideoDataset directory (Fall and ADL).
"""

import os
import sys
import time
import json
import argparse
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple

import cv2
import joblib
import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt

from ultralytics import YOLO
from live_fall_detection import LiveFeatureBuffer, FEATURE_COLS
from fall_state_machine import FallStateMachine, StateMachineConfig, State


# ==============================================================================
# Configuration Defaults
# ==============================================================================
BASE_DIR = Path(__file__).resolve().parent
DEFAULT_DATASET_DIR = BASE_DIR / "VideoDataset"
YOLO_MODEL_PATH = BASE_DIR / "yolo11n-pose.pt"
SCALER_PATH = BASE_DIR / "output" / "feature_scaler.joblib"
MODEL_PATH = BASE_DIR / "models" / "logistic_regression_balanced.joblib"
DEFAULT_OUTPUT_DIR = BASE_DIR / "output" / "video_dataset_eval"


def locate_dataset_root(base_dataset_dir: Path) -> Path:
    """Finds the root folder containing ADL and Fall subfolders."""
    if (base_dataset_dir / "VideoDataset" / "Fall").exists():
        return base_dataset_dir / "VideoDataset"
    elif (base_dataset_dir / "Fall").exists():
        return base_dataset_dir
    raise FileNotFoundError(f"Cannot locate 'Fall' folder inside {base_dataset_dir}")


def gather_video_records(dataset_root: Path) -> List[Dict[str, Any]]:
    """
    Scans dataset_root for all Fall and ADL video clips.
    Extracts metadata from folder structure and metadata.json.
    """
    records = []
    
    # 1. Fall videos (Label = 1)
    fall_dir = dataset_root / "Fall"
    for mp4_path in fall_dir.rglob("*.mp4"):
        # Ignore timelapses
        if "timelapse" in mp4_path.name.lower():
            continue
        
        meta = {}
        meta_file = mp4_path.parent / "metadata.json"
        if meta_file.exists():
            try:
                with open(meta_file, "r", encoding="utf-8") as f:
                    meta = json.load(f)
            except Exception:
                pass
                
        records.append({
            "video_path": str(mp4_path),
            "filename": mp4_path.name,
            "category": "Fall",
            "true_label": 1,
            "subject_id": meta.get("subjectId", None),
            "location_id": meta.get("locationId", None),
            "action_id": meta.get("actionId", None),
            "side": meta.get("side", None),
            "attempt": meta.get("attempt", None)
        })

    # 2. ADL videos (Label = 0)
    adl_dir = dataset_root / "ADL"
    for mp4_path in adl_dir.rglob("*.mp4"):
        if "timelapse" in mp4_path.name.lower():
            continue
            
        meta = {}
        meta_file = mp4_path.parent / "metadata.json"
        if meta_file.exists():
            try:
                with open(meta_file, "r", encoding="utf-8") as f:
                    meta = json.load(f)
            except Exception:
                pass
                
        records.append({
            "video_path": str(mp4_path),
            "filename": mp4_path.name,
            "category": "ADL",
            "true_label": 0,
            "subject_id": meta.get("subjectId", None),
            "location_id": meta.get("locationId", None),
            "action_id": meta.get("actionId", None),
            "side": meta.get("side", None),
            "attempt": meta.get("attempt", None)
        })

    return records


def sample_records(
    records: List[Dict[str, Any]],
    n_samples: int,
    random_seed: int = 42
) -> List[Dict[str, Any]]:
    """
    Creates a stratified sample balanced between Fall and ADL,
    and distributed across action types and subjects.
    """
    np.random.seed(random_seed)
    fall_recs = [r for r in records if r["true_label"] == 1]
    adl_recs = [r for r in records if r["true_label"] == 0]

    n_fall = min(n_samples // 2, len(fall_recs))
    n_adl = min(n_samples - n_fall, len(adl_recs))

    df_fall = pd.DataFrame(fall_recs)
    df_adl = pd.DataFrame(adl_recs)

    # Stratify by action_id if available
    sampled_fall_idx = []
    if "action_id" in df_fall.columns and df_fall["action_id"].notna().sum() > 0:
        grouped = df_fall.groupby("action_id", group_keys=False)
        # sample proportionally
        rate = n_fall / len(df_fall)
        sample_df = grouped.apply(lambda g: g.sample(max(1, int(round(len(g) * rate))), random_state=random_seed))
        sampled_fall_idx = sample_df.index[:n_fall].tolist()
    else:
        sampled_fall_idx = np.random.choice(len(df_fall), size=n_fall, replace=False).tolist()

    sampled_adl_idx = []
    if "action_id" in df_adl.columns and df_adl["action_id"].notna().sum() > 0:
        grouped = df_adl.groupby("action_id", group_keys=False)
        rate = n_adl / len(df_adl)
        sample_df = grouped.apply(lambda g: g.sample(max(1, int(round(len(g) * rate))), random_state=random_seed))
        sampled_adl_idx = sample_df.index[:n_adl].tolist()
    else:
        sampled_adl_idx = np.random.choice(len(df_adl), size=n_adl, replace=False).tolist()

    selected = [fall_recs[i] for i in sampled_fall_idx] + [adl_recs[i] for i in sampled_adl_idx]
    np.random.shuffle(selected)
    return selected


def evaluate_single_video(
    video_path: str,
    yolo_model: YOLO,
    scaler: Any,
    classifier: Any,
    ground_confirm_time: float = 2.0,
    hold_last_seconds: float = 2.0,
    yolo_conf: float = 0.25
) -> Dict[str, Any]:
    """
    Runs the full FallSave detection pipeline on one video file.
    Returns structured results including states, events, and max probabilities.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return {
            "error": f"Failed to open {video_path}",
            "frames_processed": 0,
            "pred_strict": 0,
            "pred_suspected": 0
        }

    raw_fps = cap.get(cv2.CAP_PROP_FPS)
    fps = float(raw_fps) if raw_fps and raw_fps > 5.0 else 30.0
    total_video_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    fsm_config = StateMachineConfig(ground_confirm_time=ground_confirm_time)
    fsm = FallStateMachine(fsm_config)
    buffer = LiveFeatureBuffer(window_length=20, fps=fps)

    frame_idx = 0
    events_triggered = []
    max_p_fall = 0.0
    max_p_lie = 0.0
    min_p_norm = 1.0

    last_kps = None
    last_conf = 0.0
    last_detected = False

    t_eval_start = time.time()

    # Process all video frames
    while True:
        ret, frame = cap.read()
        if not ret or frame is None:
            break

        frame_idx += 1
        t = frame_idx / fps

        # 1. YOLO inference
        res = yolo_model.predict(source=frame, conf=yolo_conf, verbose=False)[0]
        person_detected = False
        primary_kps = None
        person_conf = 0.0

        if res.keypoints is not None and len(res.keypoints.data) > 0:
            all_kps = res.keypoints.data.cpu().numpy()
            if res.boxes is not None and len(res.boxes) > 0:
                confs = res.boxes.conf.cpu().numpy()
                best_idx = int(confs.argmax())
                person_conf = float(confs[best_idx])
            else:
                best_idx = 0
                person_conf = 0.5
            primary_kps = all_kps[best_idx]
            person_detected = True

        last_kps = primary_kps
        last_conf = person_conf
        last_detected = person_detected

        # 2. Push to buffer
        buffer.append(person_detected, person_conf, primary_kps)

        # 3. Feature extraction & prediction
        if (
            buffer.is_ready()
            and person_detected
            and (buffer.get_missing_ratio() <= 0.20)
        ):
            feat = buffer.extract_features()
            feat_df = pd.DataFrame([feat], columns=FEATURE_COLS)
            scaled = scaler.transform(feat_df)
            probs = classifier.predict_proba(scaled)[0]
            p_n, p_f, p_l = float(probs[0]), float(probs[1]), float(probs[2])
        else:
            p_n = 1.0 if not person_detected else 0.8
            p_f, p_l = 0.0, 0.0

        max_p_fall = max(max_p_fall, p_f)
        max_p_lie = max(max_p_lie, p_l)
        min_p_norm = min(min_p_norm, p_n)

        # 4. State Machine update
        fsm_out = fsm.update(t, p_n, p_f, p_l, person_detected)
        if fsm_out.event:
            events_triggered.append((frame_idx, round(t, 2), fsm_out.event, fsm_out.state.value))

    cap.release()

    # Hold post-video frames to simulate continuous monitoring (for short dataset clips)
    if hold_last_seconds > 0:
        hold_frames = int(hold_last_seconds * fps)
        for _ in range(hold_frames):
            frame_idx += 1
            t = frame_idx / fps
            buffer.append(last_detected, last_conf, last_kps)

            if (
                buffer.is_ready()
                and last_detected
                and (buffer.get_missing_ratio() <= 0.20)
            ):
                feat = buffer.extract_features()
                feat_df = pd.DataFrame([feat], columns=FEATURE_COLS)
                scaled = scaler.transform(feat_df)
                probs = classifier.predict_proba(scaled)[0]
                p_n, p_f, p_l = float(probs[0]), float(probs[1]), float(probs[2])
            else:
                p_n = 1.0 if not last_detected else 0.8
                p_f, p_l = 0.0, 0.0

            max_p_fall = max(max_p_fall, p_f)
            max_p_lie = max(max_p_lie, p_l)
            min_p_norm = min(min_p_norm, p_n)

            fsm_out = fsm.update(t, p_n, p_f, p_l, last_detected)
            if fsm_out.event:
                events_triggered.append((frame_idx, round(t, 2), fsm_out.event, fsm_out.state.value))

    elapsed_wall = time.time() - t_eval_start
    final_state = fsm_out.state.value

    # Determine predictions
    event_names = [e[2] for e in events_triggered]
    pred_strict = 1 if ("ALERT_TRIGGERED" in event_names or final_state == "FALL_CONFIRMED") else 0
    pred_suspected = 1 if (
        "FALL_SUSPECTED" in event_names
        or "GROUND_CHECK_ENTERED" in event_names
        or pred_strict == 1
    ) else 0

    return {
        "frames_processed": frame_idx,
        "video_frames": total_video_frames,
        "fps": round(fps, 1),
        "duration_sec": round(total_video_frames / fps, 2),
        "elapsed_sec": round(elapsed_wall, 2),
        "pred_strict": pred_strict,
        "pred_suspected": pred_suspected,
        "final_state": final_state,
        "events": events_triggered,
        "max_p_fall": round(max_p_fall, 3),
        "max_p_lie": round(max_p_lie, 3),
        "min_p_norm": round(min_p_norm, 3)
    }


def compute_metrics(y_true: List[int], y_pred: List[int]) -> Dict[str, Any]:
    """Calculates TP, FP, TN, FN, Accuracy, Recall, Specificity, Precision, F1."""
    tp = sum(1 for yt, yp in zip(y_true, y_pred) if yt == 1 and yp == 1)
    tn = sum(1 for yt, yp in zip(y_true, y_pred) if yt == 0 and yp == 0)
    fp = sum(1 for yt, yp in zip(y_true, y_pred) if yt == 0 and yp == 1)
    fn = sum(1 for yt, yp in zip(y_true, y_pred) if yt == 1 and yp == 0)

    total = len(y_true)
    accuracy = (tp + tn) / total if total > 0 else 0.0
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0  # Recall (Fall detection rate)
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0  # ADL non-alert rate
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    f1 = (2 * precision * sensitivity) / (precision + sensitivity) if (precision + sensitivity) > 0 else 0.0

    return {
        "total": total,
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "accuracy": round(accuracy * 100, 2),
        "sensitivity_recall": round(sensitivity * 100, 2),
        "specificity": round(specificity * 100, 2),
        "precision": round(precision * 100, 2),
        "f1_score": round(f1 * 100, 2)
    }


def plot_confusion_matrices(
    metrics_strict: Dict[str, Any],
    metrics_suspect: Dict[str, Any],
    output_path: Path
):
    """Generates a side-by-side Confusion Matrix comparison chart."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    cms = [
        ("Strict Criteria (Confirmed Alert)", metrics_strict),
        ("Suspected Criteria (Fall / Ground Detected)", metrics_suspect)
    ]

    for ax, (title, m) in zip(axes, cms):
        cm = np.array([
            [m["tn"], m["fp"]],
            [m["fn"], m["tp"]]
        ])
        cax = ax.imshow(cm, cmap="Blues", interpolation="nearest")
        fig.colorbar(cax, ax=ax, fraction=0.046, pad=0.04)

        ax.set_title(f"{title}\nAcc: {m['accuracy']}% | Recall: {m['sensitivity_recall']}%", fontsize=11, fontweight="bold")
        ax.set_xlabel("Predicted Label", fontsize=10)
        ax.set_ylabel("True Label", fontsize=10)
        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xticklabels(["ADL (0)", "Fall (1)"])
        ax.set_yticklabels(["ADL (0)", "Fall (1)"])

        # Display text values
        thresh = cm.max() / 2.0
        for i in range(2):
            for j in range(2):
                val = cm[i, j]
                tag = "TN" if (i==0 and j==0) else "FP" if (i==0 and j==1) else "FN" if (i==1 and j==0) else "TP"
                ax.text(
                    j, i, f"{val}\n({tag})",
                    ha="center", va="center",
                    color="white" if val > thresh else "black",
                    fontweight="bold", fontsize=12
                )

    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


def print_evaluation_summary(
    metrics_strict: Dict[str, Any],
    metrics_suspect: Dict[str, Any],
    df_results: pd.DataFrame
):
    """Prints a clean CLI summary table."""
    print("\n" + "=" * 70)
    print("      FALLSAVE VIDEO DATASET ACCURACY EVALUATION RESULTS")
    print("=" * 70)
    print(f"Total Videos Evaluated: {metrics_strict['total']}")
    print(f"  - Fall Clips: {metrics_strict['tp'] + metrics_strict['fn']}")
    print(f"  - ADL Clips : {metrics_strict['tn'] + metrics_strict['fp']}")
    print("-" * 70)
    print(f"{'Metric':<28} | {'Strict (Confirmed)':<18} | {'Suspected (Event)':<18}")
    print("-" * 70)
    print(f"{'Overall Accuracy':<28} | {metrics_strict['accuracy']:>16.2f}% | {metrics_suspect['accuracy']:>16.2f}%")
    print(f"{'Fall Sensitivity (Recall)':<28} | {metrics_strict['sensitivity_recall']:>16.2f}% | {metrics_suspect['sensitivity_recall']:>16.2f}%")
    print(f"{'ADL Specificity':<28} | {metrics_strict['specificity']:>16.2f}% | {metrics_suspect['specificity']:>16.2f}%")
    print(f"{'Precision':<28} | {metrics_strict['precision']:>16.2f}% | {metrics_suspect['precision']:>16.2f}%")
    print(f"{'F1-Score':<28} | {metrics_strict['f1_score']:>16.2f}% | {metrics_suspect['f1_score']:>16.2f}%")
    print("-" * 70)
    print(f"Confusion Matrix (Strict):  TP={metrics_strict['tp']:<4} FP={metrics_strict['fp']:<4} TN={metrics_strict['tn']:<4} FN={metrics_strict['fn']:<4}")
    print(f"Confusion Matrix (Suspect): TP={metrics_suspect['tp']:<4} FP={metrics_suspect['fp']:<4} TN={metrics_suspect['tn']:<4} FN={metrics_suspect['fn']:<4}")
    print("=" * 70)

    # Per-action summary
    if "action_id" in df_results.columns and df_results["action_id"].notna().sum() > 0:
        print("\n>>> Per-Action Breakdown:")
        action_summary = df_results.groupby(["category", "action_id"]).agg(
            total=("video_path", "count"),
            pred_strict=("pred_strict", "sum"),
            pred_suspect=("pred_suspected", "sum"),
            avg_p_fall=("max_p_fall", "mean")
        ).reset_index()
        print(action_summary.to_string(index=False))
        print("=" * 70)


def main():
    parser = argparse.ArgumentParser(description="Evaluate FallSave pipeline on VideoDataset")
    parser.add_argument("--dataset-dir", type=str, default=str(DEFAULT_DATASET_DIR),
                        help="Path to VideoDataset directory")
    parser.add_argument("--sample", type=int, default=None,
                        help="Number of stratified sample videos to evaluate (e.g. 50 or 100)")
    parser.add_argument("--full", action="store_true",
                        help="Evaluate all videos in the dataset")
    parser.add_argument("--ground-confirm-time", type=float, default=2.0,
                        help="Seconds of ground contact to confirm fall (default: 2.0s)")
    parser.add_argument("--hold-time", type=float, default=2.0,
                        help="Seconds to hold final video frame to simulate continuous CCTV monitoring (default: 2.0s)")
    parser.add_argument("--output-dir", type=str, default=str(DEFAULT_OUTPUT_DIR),
                        help="Directory to store evaluation outputs and charts")
    parser.add_argument("--resume", action="store_true", default=False,
                        help="Resume evaluation from existing progress CSV if present")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for sampling (default: 42)")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "video_dataset_evaluation_details.csv"
    json_path = output_dir / "video_dataset_evaluation_summary.json"
    plot_path = output_dir / "confusion_matrix_video_dataset.png"

    print("=" * 70)
    print("   FALLSAVE: VIDEO DATASET ACCURACY BENCHMARK EVALUATION")
    print("=" * 70)

    # 1. Locate dataset
    dataset_root = locate_dataset_root(Path(args.dataset_dir))
    print(f"[1/5] Located dataset root: {dataset_root}")

    # 2. Gather videos
    records = gather_video_records(dataset_root)
    print(f"[2/5] Discovered {len(records)} total video clips ({sum(1 for r in records if r['true_label']==1)} Fall, {sum(1 for r in records if r['true_label']==0)} ADL)")

    if not records:
        print("[ERROR] No valid videos found. Please check dataset path.", file=sys.stderr)
        return

    # 3. Filter or Sample
    if args.sample is not None and not args.full:
        eval_records = sample_records(records, args.sample, random_seed=args.seed)
        print(f"[3/5] Selected Stratified Sample: {len(eval_records)} clips ({sum(1 for r in eval_records if r['true_label']==1)} Fall, {sum(1 for r in eval_records if r['true_label']==0)} ADL)")
    else:
        eval_records = records
        print(f"[3/5] Full Evaluation Mode: {len(eval_records)} clips")

    # Check for resume
    already_processed = set()
    results = []
    if args.resume and csv_path.exists():
        try:
            df_existing = pd.read_csv(csv_path)
            already_processed = set(df_existing["video_path"].tolist())
            results = df_existing.to_dict(orient="records")
            print(f"[Resume] Found existing progress: {len(already_processed)} videos already processed.")
        except Exception as e:
            print(f"[Resume] Could not read existing progress ({e}), starting fresh.")

    # 4. Load Models
    print(f"[4/5] Loading YOLO11n-Pose, Scaler, and Balanced Classifier...")
    yolo_model = YOLO(str(YOLO_MODEL_PATH))
    scaler = joblib.load(str(SCALER_PATH))
    classifier = joblib.load(str(MODEL_PATH))
    print("Models loaded successfully.")

    # 5. Evaluate Videos
    print(f"[5/5] Running evaluation pipeline (hold_time={args.hold_time}s, ground_confirm={args.ground_confirm_time}s)...")

    pending_records = [r for r in eval_records if r["video_path"] not in already_processed]
    print(f"Videos remaining to evaluate: {len(pending_records)}")

    progress_bar = tqdm(pending_records, desc="Evaluating Videos", unit="video")
    videos_since_save = 0

    try:
        for r in progress_bar:
            try:
                eval_info = evaluate_single_video(
                    video_path=r["video_path"],
                    yolo_model=yolo_model,
                    scaler=scaler,
                    classifier=classifier,
                    ground_confirm_time=args.ground_confirm_time,
                    hold_last_seconds=args.hold_time
                )
            except Exception as vid_err:
                eval_info = {
                    "error": str(vid_err),
                    "frames_processed": 0,
                    "pred_strict": 0,
                    "pred_suspected": 0,
                    "final_state": "ERROR",
                    "max_p_fall": 0.0,
                    "max_p_lie": 0.0,
                    "min_p_norm": 1.0,
                    "events": []
                }

            combined = {**r, **eval_info}
            results.append(combined)
            videos_since_save += 1

            progress_bar.set_postfix({
                "Cat": r["category"],
                "Act": r.get("action_id", "-"),
                "Strict": combined["pred_strict"],
                "MaxP": combined.get("max_p_fall", 0.0)
            })

            # Checkpoint save every 10 videos
            if videos_since_save >= 10:
                pd.DataFrame(results).to_csv(csv_path, index=False, encoding="utf-8")
                videos_since_save = 0

    except KeyboardInterrupt:
        print("\n[Interrupted] Evaluation paused by user. Saving current progress...")
    finally:
        # Save final or partial CSV
        df_res = pd.DataFrame(results)
        df_res.to_csv(csv_path, index=False, encoding="utf-8")
        print(f"\n[Saved] Detailed results exported to: {csv_path} ({len(df_res)} records)")

    if len(df_res) > 0:
        # Compute Metrics
        y_true = df_res["true_label"].tolist()
        y_pred_strict = df_res["pred_strict"].tolist()
        y_pred_suspect = df_res["pred_suspected"].tolist()

        metrics_strict = compute_metrics(y_true, y_pred_strict)
        metrics_suspect = compute_metrics(y_true, y_pred_suspect)

        # Print summary
        print_evaluation_summary(metrics_strict, metrics_suspect, df_res)

        # Save summary to JSON
        summary_data = {
            "dataset_root": str(dataset_root),
            "total_evaluated": len(df_res),
            "parameters": {
                "ground_confirm_time": args.ground_confirm_time,
                "hold_time": args.hold_time,
                "seed": args.seed
            },
            "metrics_strict": metrics_strict,
            "metrics_suspected": metrics_suspect
        }
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(summary_data, f, indent=4)
        print(f"[Saved] Metrics summary exported to: {json_path}")

        # Save Confusion Matrix Plot
        plot_confusion_matrices(metrics_strict, metrics_suspect, plot_path)
        print(f"[Saved] Confusion Matrix plot saved to: {plot_path}")


if __name__ == "__main__":
    main()
