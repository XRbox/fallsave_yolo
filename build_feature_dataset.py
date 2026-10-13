import json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

# ==============================================================================
# 1. CONFIGURATION & APPROVED SETTINGS
# ==============================================================================
KEYPOINTS_PATH = Path("output/keypoints.csv")
MANIFEST_PATH = Path("metadata/split_manifest.csv")
OUTPUT_DIR = Path("output")

WINDOW_LENGTH = 20          # 20 frames (~0.67s at 30 FPS)
TRAIN_STRIDE = 5            # 5 frames (~75% overlap for training augmentation)
VAL_TEST_STRIDE = 2         # 2 frames (~90% overlap for dense evaluation)
CONF_THRESHOLD = 0.20       # Keypoints with conf < 0.20 are masked to NaN
INTERP_LIMIT = 3            # Interpolate only short gaps <= 3 frames
MAX_MISSING_RATIO = 0.30    # Exclude windows with > 30% missing frames
FPS = 30.0                  # Frame rate for velocity calculation

# Audited event boundaries: (last_normal, fall_start, impact, lying_start or None)
EVENT_BOUNDARIES = {
    'fall-01': (88, 91, 110, 118),
    'fall-02': (44, 46, 72, 75),
    'fall-03': (158, 161, 175, 180),
    'fall-04': (11, 13, 39, 43),
    'fall-05': (75, 78, 105, 115),
    'fall-06': (27, 29, 45, 48),
    'fall-07': (94, 96, 120, 128),
    'fall-08': (25, 27, 50, 54),
    'fall-09': (126, 128, 150, 158),
    'fall-10': (35, 37, 58, 62),
    'fall-11': (54, 56, 75, 78),
    'fall-12': (25, 27, 48, 52),
    'fall-13': (58, 60, 70, None),     # Tracking failure stress case
    'fall-14': (12, 14, 41, None),     # Post-impact switches to edge object
    'fall-15': (35, 37, 49, 55),
    'fall-16': (10, 12, 35, 38),
    'fall-17': (64, 66, 77, 82),
    'fall-18': (13, 15, 38, None),     # Post-impact 22 missing frames
    'fall-19': (38, 40, 51, None),     # Post-impact 46 missing frames
    'fall-20': (41, 43, 55, 60),
    'fall-21': (36, 38, 45, None),     # Post-impact 9 missing frames
    'fall-22': (9, 11, 25, 28),
    'fall-23': (37, 39, 52, 56),
    'fall-24': (13, 15, 35, 40),
    'fall-25': (38, 41, 65, None),     # Post-impact 16 missing frames
    'fall-26': (5, 7, 30, 35),
    'fall-27': (52, 55, 72, None),     # Post-impact 16 missing frames
    'fall-28': (20, 22, 45, 48),
    'fall-29': (51, 54, 71, None),     # Post-impact 16 missing frames
    'fall-30': (19, 21, 47, 50),
}

FEATURE_COLUMNS = [
    'mean_torso_angle',
    'max_torso_angle',
    'min_torso_angle',
    'delta_torso_angle',
    'std_torso_angle',
    'max_downward_velocity',
    'mean_vertical_velocity',
    'delta_hip_y',
    'max_hip_y',
    'max_angular_velocity',
    'mean_angular_velocity',
    'mean_torso_span_y',
    'delta_torso_span_y',
    'mean_kp_aspect_ratio',
    'delta_kp_aspect_ratio',
    'missing_ratio',
    'mean_person_confidence'
]


# ==============================================================================
# 2. SEQUENCE PREPROCESSING & FEATURE EXTRACTION
# ==============================================================================
def process_sequence(seq_df, short_id, seq_type):
    """
    Sanitize keypoints, interpolate short micro-gaps (<= 3 frames),
    compute frame-level biomechanical signals, and assign frame-level labels.
    """
    s = seq_df.sort_values('frame_id').reset_index(drop=True)
    n = len(s)

    # 1. Mask low-confidence keypoints (< CONF_THRESHOLD) to NaN
    for i in range(17):
        mask = s[f'kp_{i}_confidence'] < CONF_THRESHOLD
        s.loc[mask, f'kp_{i}_x'] = np.nan
        s.loc[mask, f'kp_{i}_y'] = np.nan

    # 2. Linear interpolate micro-gaps <= INTERP_LIMIT within this sequence
    x_cols = [f'kp_{i}_x' for i in range(17)]
    y_cols = [f'kp_{i}_y' for i in range(17)]
    s[x_cols] = s[x_cols].interpolate(method='linear', limit=INTERP_LIMIT)
    s[y_cols] = s[y_cols].interpolate(method='linear', limit=INTERP_LIMIT)

    # 3. Core Torso Joints (Mid-Shoulder 5,6 and Mid-Hip 11,12)
    sh_x = s[['kp_5_x', 'kp_6_x']].mean(axis=1)
    sh_y = s[['kp_5_y', 'kp_6_y']].mean(axis=1)
    hip_x = s[['kp_11_x', 'kp_12_x']].mean(axis=1)
    hip_y = s[['kp_11_y', 'kp_12_y']].mean(axis=1)

    dx = sh_x - hip_x
    dy = sh_y - hip_y
    torso_len = np.sqrt(dx**2 + dy**2)

    # Scale factor: median torso length over valid standing frames, clamped to min 15px
    valid_lengths = torso_len.dropna()
    scale = float(valid_lengths.median()) if len(valid_lengths) > 0 else 50.0
    scale = max(scale, 15.0)

    # Torso Angle relative to vertical axis (0 deg = upright, 90 deg = horizontal)
    torso_angle = np.degrees(np.arctan2(np.abs(dx), np.abs(dy)))
    # If shoulder is below hip (inverted / head-first), angle > 90 deg
    inverted = sh_y > hip_y
    torso_angle = np.where(inverted, 180.0 - torso_angle, torso_angle)

    # Normalized Hip coordinates
    hip_y_norm = hip_y / scale
    hip_x_norm = hip_x / scale

    # Vertical Velocity of Hip (scale units per second)
    hip_v_y = hip_y_norm.diff() * FPS

    # Angular Velocity (degrees per second)
    ang_v = pd.Series(torso_angle).diff().abs() * FPS

    # Torso vertical span (|sh_y - hip_y| / scale)
    torso_span_y = np.abs(dy) / scale

    # Keypoint envelope aspect ratio (width / height of detected keypoints)
    kp_xs = s[x_cols]
    kp_ys = s[y_cols]
    kp_w = kp_xs.max(axis=1) - kp_xs.min(axis=1)
    kp_h = kp_ys.max(axis=1) - kp_ys.min(axis=1)
    kp_aspect_ratio = kp_w / np.maximum(kp_h, 15.0)

    # Track missing frames (where torso is missing)
    torso_missing = hip_y.isna() | sh_y.isna()

    # 4. Determine frame-level ground-truth labels (0: Normal, 1: Falling, 2: Lying, -1: Ambiguous)
    frame_labels = np.zeros(n, dtype=int)
    if seq_type == 'fall' and short_id in EVENT_BOUNDARIES:
        ln, fs, imp, ls = EVENT_BOUNDARIES[short_id]
        for f in range(n):
            if f <= ln:
                frame_labels[f] = 0   # Normal
            elif f < fs:
                frame_labels[f] = -1  # Ambiguous buffer before fall
            elif f <= min(imp + 3, n - 1):
                frame_labels[f] = 1   # Falling
            else:
                if ls is not None and f >= ls:
                    frame_labels[f] = 2   # Lying
                else:
                    frame_labels[f] = -1  # Ambiguous / uncertain lying
    else:
        # ADL sequences: all frames are Normal
        frame_labels[:] = 0

    return {
        'n': n,
        'torso_angle': pd.Series(torso_angle),
        'hip_y_norm': hip_y_norm,
        'hip_v_y': hip_v_y,
        'ang_v': ang_v,
        'torso_span_y': torso_span_y,
        'kp_aspect_ratio': kp_aspect_ratio,
        'torso_missing': torso_missing,
        'person_conf': s['person_confidence'].fillna(0.0),
        'frame_labels': frame_labels,
    }


# ==============================================================================
# 3. SLIDING WINDOW GENERATION & LABELING
# ==============================================================================
def generate_sequence_windows(seq_data, sid, short_id, seq_type, split, stride):
    """
    Generate sliding windows strictly within the sequence.
    Extract temporal features and label windows as Normal, Falling, Lying, or Exclude.
    """
    n = seq_data['n']
    windows = []

    for start_f in range(0, n - WINDOW_LENGTH + 1, stride):
        end_f = start_f + WINDOW_LENGTH - 1
        center_f = start_f + WINDOW_LENGTH // 2
        w_id = f"{short_id}_w{start_f:04d}_{end_f:04d}"

        # Window slices
        w_missing = seq_data['torso_missing'].iloc[start_f:end_f + 1]
        missing_count = int(w_missing.sum())
        missing_ratio = float(missing_count / WINDOW_LENGTH)

        w_flabels = seq_data['frame_labels'][start_f:end_f + 1]
        n_falling = int((w_flabels == 1).sum())
        n_lying = int((w_flabels == 2).sum())
        n_normal = int((w_flabels == 0).sum())

        # Determine window label & exclusion reasons
        label = -1
        label_name = 'excluded'
        exclusion_reason = None

        if short_id == 'fall-13':
            label = -1
            label_name = 'excluded'
            exclusion_reason = 'stress_case_tracking_failure'
        elif missing_ratio > MAX_MISSING_RATIO:
            label = -1
            label_name = 'excluded'
            exclusion_reason = 'high_missing_ratio'
        else:
            # Check for un-interpolated missing gaps > INTERP_LIMIT inside window
            runs = []
            cur_run = 0
            for m in w_missing:
                if m:
                    cur_run += 1
                else:
                    if cur_run > 0:
                        runs.append(cur_run)
                    cur_run = 0
            if cur_run > 0:
                runs.append(cur_run)

            if runs and max(runs) > INTERP_LIMIT:
                label = -1
                label_name = 'excluded'
                exclusion_reason = 'uninterpolated_gap'
            else:
                # Proportional voting on phases
                if n_falling / WINDOW_LENGTH >= 0.40:
                    label = 1
                    label_name = 'falling'
                elif n_lying / WINDOW_LENGTH >= 0.60:
                    label = 2
                    label_name = 'lying'
                elif n_normal / WINDOW_LENGTH >= 0.80:
                    label = 0
                    label_name = 'normal'
                else:
                    label = -1
                    label_name = 'excluded'
                    exclusion_reason = 'ambiguous_phase_mix'

        # Extract temporal features over the window
        w_angle = seq_data['torso_angle'].iloc[start_f:end_f + 1].dropna()
        w_hip_y = seq_data['hip_y_norm'].iloc[start_f:end_f + 1].dropna()
        w_hip_v = seq_data['hip_v_y'].iloc[start_f:end_f + 1].dropna()
        w_ang_v = seq_data['ang_v'].iloc[start_f:end_f + 1].dropna()
        w_span_y = seq_data['torso_span_y'].iloc[start_f:end_f + 1].dropna()
        w_ar = seq_data['kp_aspect_ratio'].iloc[start_f:end_f + 1].dropna()
        w_conf = seq_data['person_conf'].iloc[start_f:end_f + 1]

        # Calculate robust summary stats (handling potential missing gracefully)
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

        features = {
            'window_id': w_id,
            'sequence_id': sid,
            'short_id': short_id,
            'sequence_type': seq_type,
            'split': split,
            'start_frame': start_f,
            'end_frame': end_f,
            'center_frame': center_f,
            'label': label,
            'label_name': label_name,
            'exclusion_reason': exclusion_reason,
            # Feature columns:
            'mean_torso_angle': mean_angle,
            'max_torso_angle': max_angle,
            'min_torso_angle': min_angle,
            'delta_torso_angle': delta_angle,
            'std_torso_angle': std_angle,
            'max_downward_velocity': max_drop_v,
            'mean_vertical_velocity': mean_v,
            'delta_hip_y': delta_hip_y,
            'max_hip_y': max_hip_y,
            'max_angular_velocity': max_ang_v,
            'mean_angular_velocity': mean_ang_v,
            'mean_torso_span_y': mean_span_y,
            'delta_torso_span_y': delta_span_y,
            'mean_kp_aspect_ratio': mean_ar,
            'delta_kp_aspect_ratio': delta_ar,
            'missing_ratio': missing_ratio,
            'mean_person_confidence': mean_conf,
        }
        windows.append(features)

    return windows


# ==============================================================================
# 4. MAIN PIPELINE EXECUTION
# ==============================================================================
def main():
    print("=" * 60)
    print("   FALLSAVE STEP 2.3: FEATURE PIPELINE EXECUTION   ")
    print("=" * 60)

    # 1. Load data
    df = pd.read_csv(KEYPOINTS_PATH)
    manifest = pd.read_csv(MANIFEST_PATH)
    print(f"Loaded keypoints: {len(df):,} frames across {df['sequence_id'].nunique()} sequences")
    print(f"Loaded manifest: {len(manifest)} sequences")

    # 2. Process sequences and generate windows
    all_windows = []

    for idx, row in manifest.iterrows():
        sid = row['sequence_id']
        split = row['split']
        seq_type = row['sequence_type']
        short_id = f"{Path(sid).name.split('-')[0]}-{Path(sid).name.split('-')[1]}"

        # Stride setting
        stride = TRAIN_STRIDE if split == 'train' else VAL_TEST_STRIDE

        seq_df = df[df['sequence_id'] == sid]
        seq_data = process_sequence(seq_df, short_id, seq_type)
        win_list = generate_sequence_windows(seq_data, sid, short_id, seq_type, split, stride)
        all_windows.extend(win_list)

    win_df = pd.DataFrame(all_windows)
    print(f"\nGenerated total windows: {len(win_df):,}")

    # 3. Fit feature scaler on TRAINING data only (valid non-excluded windows)
    train_valid = win_df[(win_df['split'] == 'train') & (win_df['label'] != -1)]
    print(f"Training windows available for scaler fitting: {len(train_valid):,}")

    scaler = StandardScaler()
    scaler.fit(train_valid[FEATURE_COLUMNS])

    # Save fitted scaler
    scaler_path = OUTPUT_DIR / "feature_scaler.joblib"
    joblib.dump(scaler, scaler_path)
    print(f"Saved fitted StandardScaler to: {scaler_path}")

    # 4. Apply scaler transformation and save split datasets
    for split_name in ['train', 'validation', 'test']:
        split_subset = win_df[win_df['split'] == split_name].copy()

        # Add scaled feature columns
        scaled_mat = scaler.transform(split_subset[FEATURE_COLUMNS])
        for i, col in enumerate(FEATURE_COLUMNS):
            split_subset[f"norm_{col}"] = scaled_mat[:, i]

        out_path = OUTPUT_DIR / f"features_{split_name}.csv"
        split_subset.to_csv(out_path, index=False)
        print(f"Saved {split_name} features ({len(split_subset):,} windows) to: {out_path}")

    # 5. Generate and save audit summary report
    summary = {
        'total_windows': len(win_df),
        'by_split': win_df['split'].value_counts().to_dict(),
        'by_label': win_df['label_name'].value_counts().to_dict(),
        'cross_tab_split_label': pd.crosstab(win_df['split'], win_df['label_name']).to_dict(),
        'exclusion_reasons': win_df['exclusion_reason'].value_counts().dropna().to_dict(),
        'feature_columns': FEATURE_COLUMNS,
    }

    summary_path = OUTPUT_DIR / "feature_pipeline_summary.json"
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"Saved audit summary to: {summary_path}")

    print("\n" + "=" * 60)
    print("           FEATURE PIPELINE COMPLETED SUCCESSFULLY          ")
    print("=" * 60)

    # Print summary tables
    print("\n[Cross-Tabulation: Split vs. Window Class]")
    print(pd.crosstab(win_df['split'], win_df['label_name'], margins=True))

    print("\n[Exclusion Reasons Breakdown]")
    print(win_df['exclusion_reason'].value_counts(dropna=False))


if __name__ == "__main__":
    main()
