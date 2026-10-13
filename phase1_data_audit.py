import pandas as pd
import numpy as np
from pathlib import Path

# Mapping YOLO pose keypoints 0-16 to standard anatomical names
KEYPOINT_NAMES = [
    "nose",          # 0
    "left_eye",      # 1
    "right_eye",     # 2
    "left_ear",      # 3
    "right_ear",     # 4
    "left_shoulder", # 5
    "right_shoulder",# 6
    "left_elbow",    # 7
    "right_elbow",   # 8
    "left_wrist",    # 9
    "right_wrist",   # 10
    "left_hip",      # 11
    "right_hip",     # 12
    "left_knee",     # 13
    "right_knee",    # 14
    "left_ankle",    # 15
    "right_ankle"    # 16
]

def main():
    csv_path = Path("output/keypoints.csv")
    if not csv_path.exists():
        print(f"Error: {csv_path} not found.")
        return

    df = pd.read_csv(csv_path)

    print("==================================================")
    print("      PHASE 1: AUDIT รายงานการตรวจสอบข้อมูล       ")
    print("==================================================")

    # 1. โครงสร้างไฟล์และคอลัมน์
    print("\n--- 1. โครงสร้างไฟล์และคอลัมน์ ---")
    print(f"จำนวนแถวทั้งหมด (Frames): {len(df):,}")
    print(f"จำนวนคอลัมน์: {df.shape[1]}")
    meta_cols = ["sequence_id", "frame_id", "image_path", "label", "person_detected", "person_confidence"]
    kp_cols = [c for c in df.columns if c not in meta_cols]
    print(f"คอลัมน์ Metadata ({len(meta_cols)}): {meta_cols}")
    print(f"คอลัมน์ Keypoints ({len(kp_cols)}): {len(kp_cols)//3} จุด (x, y, confidence)")

    # 2. การตรวจพบคน (Person Detection) & Confidence
    print("\n--- 2. คุณภาพการตรวจจับคน (Person Detection) ---")
    total_frames = len(df)
    detected_frames = df[df["person_detected"] == 1]
    undetected_frames = df[df["person_detected"] == 0]

    print(f"เฟรมที่ตรวจพบคน (person_detected=1): {len(detected_frames):,} ({len(detected_frames)/total_frames*100:.2f}%)")
    print(f"เฟรมที่ไม่พบคน (person_detected=0):   {len(undetected_frames):,} ({len(undetected_frames)/total_frames*100:.2f}%)")

    conf = detected_frames["person_confidence"]
    print(f"Person BBox Confidence (เมื่อพบคน): Mean={conf.mean():.4f}, Median={conf.median():.4f}, Min={conf.min():.4f}, Max={conf.max():.4f}")

    # ตรวจสอบ Keypoints รายจุด (Missing & Confidence)
    print("\n--- 3. คุณภาพ Keypoints ทั้ง 17 จุด (เฉพาะเฟรมที่พบคน) ---")
    kp_stats = []
    for kid, name in enumerate(KEYPOINT_NAMES):
        x_col = f"kp_{kid}_x"
        y_col = f"kp_{kid}_y"
        c_col = f"kp_{kid}_confidence"

        nan_count = detected_frames[x_col].isna().sum()
        conf_vals = detected_frames[c_col].dropna()

        # นับจำนวนจุดที่ confidence ต่ำ (< 0.5)
        low_conf = (conf_vals < 0.5).sum()

        kp_stats.append({
            "ID": kid,
            "Keypoint": name,
            "NaN Count": nan_count,
            "Mean Conf": round(conf_vals.mean(), 4) if len(conf_vals) else 0,
            "Median Conf": round(conf_vals.median(), 4) if len(conf_vals) else 0,
            "Conf < 0.5 (%)": round(low_conf / len(conf_vals) * 100, 2) if len(conf_vals) else 100
        })

    kp_df = pd.DataFrame(kp_stats)
    print(kp_df.to_string(index=False))

    # 4. สถิติเฟรมที่หายไปติดต่อกัน (Consecutive Missing Frames)
    print("\n--- 4. การหายไปต่อเนื่องของบุคคล (Consecutive Undetected Frames) ---")
    consecutive_missing = []
    for seq_id, group in df.groupby("sequence_id"):
        group = group.sort_values("frame_id")
        det = group["person_detected"].values
        run = 0
        for val in det:
            if val == 0:
                run += 1
            else:
                if run > 0:
                    consecutive_missing.append((seq_id, run))
                    run = 0
        if run > 0:
            consecutive_missing.append((seq_id, run))

    if consecutive_missing:
        runs = [r[1] for r in consecutive_missing]
        print(f"จำนวนช่วงที่คนหายไปชั่วคราว: {len(runs)} ครั้ง")
        print(f"ความยาวช่วงที่คนหาย (เฟรม): Mean={np.mean(runs):.1f}, Median={np.median(runs):.1f}, Max={np.max(runs)}, Min={np.min(runs)}")
        longest_runs = sorted(consecutive_missing, key=lambda x: x[1], reverse=True)[:5]
        print("Sequence ที่คนหายติดต่อกันนานที่สุด 5 อันดับแรก:")
        for sid, length in longest_runs:
            print(f"  - {sid}: หายติดต่อกัน {length} เฟรม")
    else:
        print("ไม่มีช่วงที่คนหายเลย")

    # 5. การกระจายตัวของ Label และ Sequence
    print("\n--- 5. สถิติราย Label และ Sequence ---")
    seq_summary = df.groupby(["label", "sequence_id"]).agg(
        total_frames=("frame_id", "count"),
        detected_frames=("person_detected", "sum"),
        detection_rate=("person_detected", lambda x: round(x.mean() * 100, 2))
    ).reset_index()

    for label in ["fall", "normal"]:
        sub = seq_summary[seq_summary["label"] == label]
        print(f"\n[Label: {label.upper()}]")
        print(f"  จำนวน Sequences: {len(sub)}")
        print(f"  เฟรมทั้งหมด: {sub['total_frames'].sum():,}")
        print(f"  ความยาว Sequence: Min={sub['total_frames'].min()}, Mean={sub['total_frames'].mean():.1f}, Max={sub['total_frames'].max()}")
        print(f"  Detection Rate รวม: {sub['detected_frames'].sum() / sub['total_frames'].sum() * 100:.2f}%")
        print(f"  Sequence ที่มี Detection Rate ต่ำสุด 3 อันดับ:")
        lowest_det = sub.sort_values("detection_rate").head(3)
        for _, row in lowest_det.iterrows():
            print(f"    - {row['sequence_id']}: {row['detection_rate']}% (พบ {row['detected_frames']}/{row['total_frames']} เฟรม)")

    print("\n==================================================")
    print("               สิ้นสุดการตรวจสอบ                 ")
    print("==================================================")

if __name__ == "__main__":
    main()
