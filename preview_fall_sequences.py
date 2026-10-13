from pathlib import Path
import cv2
import numpy as np
import pandas as pd

# ==========================================
# 1. CONFIGURATION
# ==========================================
CSV_PATH = Path("output/keypoints.csv")
OUTPUT_PATH = Path("output/fall_sequence_check.jpg")


def read_image(path):
    p = Path(path)
    if not p.exists():
        p = Path(__file__).resolve().parent / path
    if not p.exists():
        return None
    try:
        data = np.fromfile(str(p), dtype=np.uint8)
        return cv2.imdecode(data, cv2.IMREAD_COLOR)
    except Exception:
        return cv2.imread(str(p))


def save_image(path, img):
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    ext = p.suffix if p.suffix else ".jpg"
    is_success, buffer = cv2.imencode(ext, img)
    if is_success:
        buffer.tofile(str(p))
    else:
        cv2.imwrite(str(p), img)


def main():
    if not CSV_PATH.exists():
        print(f"ไม่พบไฟล์ CSV: {CSV_PATH.resolve()}")
        return

    df = pd.read_csv(CSV_PATH)
    fall_df = df[df["label"] == "fall"].copy()

    # จัดกลุ่มและหาความยาวของแต่ละ Sequence
    seq_lengths = fall_df.groupby("sequence_id").size().sort_values()

    # เลือก 3 Sequences: สั้น (Short), กลาง (Median), ยาว (Long)
    short_seq_id = seq_lengths.index[0]  # สั้นสุด
    median_seq_id = seq_lengths.index[len(seq_lengths) // 2]  # ค่ากลาง
    long_seq_id = seq_lengths.index[-1]  # ยาวสุด

    selected_targets = [
        ("Short", short_seq_id),
        ("Median", median_seq_id),
        ("Long", long_seq_id),
    ]

    print("=== ลำดับเหตุการณ์ที่เลือกสำหรับตรวจสอบ ===")
    for category, sid in selected_targets:
        print(f"[{category}] {sid}: {seq_lengths[sid]} เฟรม")

    tiles = []

    for category, target in selected_targets:
        seq = fall_df[fall_df["sequence_id"] == target].sort_values("frame_id")
        n_frames = len(seq)

        # เลือก 3 เฟรม: ต้น (~15%), กลาง (~50%), ท้าย (~85%)
        indices = [
            ("Early", int(round(0.15 * (n_frames - 1)))),
            ("Middle", int(round(0.50 * (n_frames - 1)))),
            ("Late", int(round(0.85 * (n_frames - 1)))),
        ]

        for stage, idx in indices:
            row = seq.iloc[idx]
            image_path = row["image_path"]
            image = read_image(image_path)

            if image is None:
                print(f"อ่านภาพไม่ได้: {image_path}")
                continue

            # ปรับขนาดภาพเป็น 480x270 (16:9)
            image = cv2.resize(image, (480, 270))

            seq_name = Path(target).name
            title = f"{category} ({stage}) | {seq_name} | frame={row['frame_id']} | label={row['label']}"

            # แถบสีดำด้านบนสำหรับใส่ข้อความกำกับ
            cv2.rectangle(image, (0, 0), (480, 30), (0, 0, 0), -1)
            cv2.putText(
                image,
                title[:65],
                (8, 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.40,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )

            tiles.append(image)
            print(f"{title} | {image_path}")

    if tiles:
        # เรียงภาพแถวละ 3 คอลัมน์ (3x3 grid)
        blank = [
            cv2.copyMakeBorder(
                img, 0, 0, 0, 0, cv2.BORDER_CONSTANT, value=(30, 30, 30)
            )
            for img in tiles
        ]
        while len(blank) < 9:
            blank.append(255 * np.ones((270, 480, 3), dtype="uint8"))

        rows = [cv2.hconcat(blank[i : i + 3]) for i in range(0, 9, 3)]
        contact_sheet = cv2.vconcat(rows)

        save_image(OUTPUT_PATH, contact_sheet)
        print(f"\nบันทึกภาพตรวจสอบที่: {OUTPUT_PATH.resolve()}")
    else:
        print("ไม่มีภาพที่อ่านได้ กรุณาตรวจสอบ image_path ใน CSV")


if __name__ == "__main__":
    main()
