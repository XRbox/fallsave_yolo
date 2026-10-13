"""
FallSave dataset audit and sequence-level split generator.

Usage:
    python prepare_fallsave_data.py --csv output/keypoints.csv --out output/prepared

This script does NOT train a classifier. It audits the extracted keypoints and
creates a sequence-level split manifest to prevent adjacent frames from the
same video leaking into train/validation/test.
"""
from pathlib import Path
import argparse
import pandas as pd
from sklearn.model_selection import train_test_split

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", default="output/keypoints.csv")
    parser.add_argument("--out", default="output/prepared")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    csv_path = Path(args.csv)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(csv_path)

    required = {"sequence_id", "frame_id", "image_path", "label", "person_detected"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"CSV missing required columns: {sorted(missing)}")

    if df["sequence_id"].isna().any() or df["label"].isna().any():
        raise ValueError("sequence_id/label contains missing values.")
    labels = set(df["label"].astype(str).str.lower().unique())
    if not labels.issubset({"fall", "normal"}):
        raise ValueError(f"Unexpected labels: {sorted(labels)}")

    keypoint_cols = [c for c in df.columns if c.startswith("kp_")]
    if len(keypoint_cols) != 51:
        raise ValueError(f"Expected 51 keypoint columns (17 points x 3), found {len(keypoint_cols)}")

    dupes = int(df.duplicated(["sequence_id", "frame_id"]).sum())
    groups = (df.groupby(["sequence_id", "label"], dropna=False)
                .agg(frames=("frame_id", "size"),
                     detected=("person_detected", "sum"),
                     first_frame=("frame_id", "min"),
                     last_frame=("frame_id", "max"))
                .reset_index())
    groups["undetected"] = groups["frames"] - groups["detected"]
    groups["detected_rate"] = groups["detected"] / groups["frames"]

    # Split sequences, not frames. This keeps all frames from one recording together.
    train, temp = train_test_split(
        groups, test_size=0.30, random_state=args.seed, stratify=groups["label"]
    )
    validation, test = train_test_split(
        temp, test_size=0.50, random_state=args.seed, stratify=temp["label"]
    )
    split_by_sequence = {}
    for split_name, part in (("train", train), ("validation", validation), ("test", test)):
        for sequence_id in part["sequence_id"]:
            split_by_sequence[sequence_id] = split_name

    groups["split"] = groups["sequence_id"].map(split_by_sequence)
    groups.to_csv(out_dir / "sequence_summary.csv", index=False)
    groups[["sequence_id", "label", "frames", "split"]].to_csv(
        out_dir / "sequence_split_manifest.csv", index=False
    )

    not_detected = int((df["person_detected"] == 0).sum())
    report = [
        "FallSave dataset audit",
        "======================",
        f"Input: {csv_path}",
        f"Rows/frames: {len(df)}",
        f"Columns: {len(df.columns)}",
        f"Unique sequences: {df['sequence_id'].nunique()}",
        f"Duplicate (sequence_id, frame_id) pairs: {dupes}",
        f"Person detected: {int(df['person_detected'].sum())} ({df['person_detected'].mean()*100:.1f}%)",
        f"Person not detected: {not_detected} ({not_detected/len(df)*100:.1f}%)",
        "",
        "Sequence split counts:",
        groups.groupby(["split", "label"]).size().to_string(),
        "",
        "IMPORTANT: label is sequence-level. A 'fall' sequence may contain normal",
        "frames before the fall and recovery/lying frames after it. Do not treat",
        "every frame in a fall sequence as a confirmed fall frame without suitable",
        "per-frame annotations or an explicit weak-supervision strategy.",
        "",
        "No model has been trained by this script."
    ]
    (out_dir / "audit_report.txt").write_text("\n".join(report), encoding="utf-8")
    print("\n".join(report))
    print(f"\nWrote outputs to: {out_dir.resolve()}")

if __name__ == "__main__":
    main()
