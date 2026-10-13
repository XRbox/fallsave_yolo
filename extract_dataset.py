from pathlib import Path
import re
import sys

import cv2
import pandas as pd
from ultralytics import YOLO


# ==========================================
# 1. CONFIGURATION: แก้ Path ให้ตรงกับเครื่อง
# ==========================================

DATASET_DIR = Path("dataset")
OUTPUT_DIR = Path("output")

MODEL_PATH = "yolo11n-pose.pt"
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg"}

# บันทึกภาพตัวอย่างที่วาด Skeleton แล้ว
SAVE_PREVIEW = True

# กำหนด 3 ช่วงของแต่ละ Sequence (อัตราส่วนตำแหน่งเฟรม 0.0 - 1.0)
PREVIEW_STAGES = {
    "early": 0.15,    # 1. ช่วงต้น (Early): ดูว่า YOLO ตรวจจับคนในท่ายืนปกติได้หรือไม่ (~15%)
    "middle": 0.50,   # 2. ช่วงกลาง (Middle): ตรวจดูว่ามีท่าทางเปลี่ยนแปลงอย่างไรระหว่าง Sequence (~50%)
    "late": 0.85,     # 3. ช่วงท้าย (Late): ตรวจว่า YOLO ยังจับตำแหน่งร่างกายได้หรือไม่เมื่อคนนอนอยู่บนพื้น (~85%)
}


# ==========================================
# 2. HELPER FUNCTIONS
# ==========================================

def natural_sort_key(path):
    """เรียงชื่อไฟล์ตามตัวเลข เช่น 2.png ก่อน 10.png"""
    return [
        int(part) if part.isdigit() else part.lower()
        for part in re.split(r"(\d+)", str(path))
    ]


def get_label(image_path):
    """
    คาดหวังโครงสร้าง:
    dataset/fall/ชื่อเหตุการณ์/ภาพ.png
    dataset/normal/ชื่อเหตุการณ์/ภาพ.png
    """
    relative_path = image_path.relative_to(DATASET_DIR)
    category = relative_path.parts[0].lower()

    if category not in {"fall", "normal"}:
        raise ValueError(
            f"ไม่รู้จัก Label ของ {image_path}\n"
            "กรุณาแยกภาพไว้ใน dataset/fall "
            "หรือ dataset/normal"
        )

    return category


def get_preview_target_indices(num_frames, stages=PREVIEW_STAGES):
    """
    คำนวณตำแหน่ง frame index สำหรับแต่ละช่วง (early, middle, late)
    คืนค่าเป็น dict: {frame_index: stage_name}
    """
    if num_frames <= 0:
        return {}

    target_indices = {}
    for stage_name, ratio in stages.items():
        idx = int(round(ratio * (num_frames - 1)))
        idx = max(0, min(idx, num_frames - 1))
        target_indices[idx] = stage_name

    return target_indices


def make_columns():
    columns = [
        "sequence_id",
        "frame_id",
        "image_path",
        "label",
        "person_detected",
        "person_confidence",
    ]

    for keypoint_id in range(17):
        columns.extend([
            f"kp_{keypoint_id}_x",
            f"kp_{keypoint_id}_y",
            f"kp_{keypoint_id}_confidence",
        ])

    return columns


def generate_previews_only(sequences, model):
    """
    สร้างเฉพาะภาพ Preview 3 ช่วง (Early, Middle, Late) โดยไม่ต้องรันทุกเฟรม
    """
    total_saved = 0
    print(f"\nกำลังสร้างภาพ Preview 3 ช่วง (Early, Middle, Late) สำหรับ {len(sequences)} เหตุการณ์...")

    for sequence_index, ((label, sequence_id), paths) in enumerate(
        sorted(sequences.items()), start=1
    ):
        paths = sorted(paths, key=natural_sort_key)
        targets = get_preview_target_indices(len(paths))

        preview_dir = OUTPUT_DIR / "previews" / sequence_id
        preview_dir.mkdir(parents=True, exist_ok=True)

        for frame_index, stage_name in targets.items():
            image_path = paths[frame_index]
            image = cv2.imread(str(image_path))

            if image is None:
                print(f"อ่านภาพไม่ได้: {image_path}")
                continue

            result = model.predict(
                source=image,
                conf=0.25,
                verbose=False,
            )[0]

            preview = result.plot()
            preview_path = preview_dir / f"{stage_name}_frame_{frame_index:05d}.jpg"
            cv2.imwrite(str(preview_path), preview)
            total_saved += 1

        print(
            f"[{sequence_index}/{len(sequences)}] {label}/{sequence_id}: บันทึก Preview ครบ 3 ช่วงแล้ว",
            flush=True
        )

    print(f"\nบันทึกภาพ Preview สำเร็จทั้งหมด: {total_saved} ภาพ")
    print(f"บันทึกไว้ที่: {(OUTPUT_DIR / 'previews').resolve()}")


# ==========================================
# 3. MAIN PROCESS
# ==========================================

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if not DATASET_DIR.exists():
        raise FileNotFoundError(
            f"ไม่พบ Dataset: {DATASET_DIR.resolve()}"
        )

    image_paths = sorted(
        [
            path
            for path in DATASET_DIR.rglob("*")
            if path.is_file()
            and path.suffix.lower() in IMAGE_EXTENSIONS
        ],
        key=natural_sort_key,
    )

    if not image_paths:
        raise RuntimeError(
            f"ไม่พบภาพใน {DATASET_DIR.resolve()}"
        )

    # จัดกลุ่มตามโฟลเดอร์เหตุการณ์
    sequences = {}

    for image_path in image_paths:
        relative_path = image_path.relative_to(DATASET_DIR)

        if len(relative_path.parts) < 3:
            raise ValueError(
                f"ภาพอยู่ผิดตำแหน่ง: {image_path}\n"
                "ควรเก็บภาพใน dataset/fall/sequence/ "
                "หรือ dataset/normal/sequence/"
            )

        label = get_label(image_path)
        sequence_id = str(relative_path.parent)

        sequences.setdefault(
            (label, sequence_id), []
        ).append(image_path)

    print(f"พบภาพทั้งหมด: {len(image_paths)}")
    print(f"พบเหตุการณ์ทั้งหมด: {len(sequences)}")

    # โหลดโมเดลเพียงครั้งเดียว
    model = YOLO(MODEL_PATH)

    # โหมดสร้างเฉพาะภาพ Preview (หากใส่ Argument --previews-only)
    if "--previews-only" in sys.argv:
        generate_previews_only(sequences, model)
        return

    rows = []
    failed_images = []

    # ======================================
    # 4. PROCESS EVERY FRAME
    # ======================================

    for sequence_index, ((label, sequence_id), paths) in enumerate(
        sorted(sequences.items()), start=1
    ):
        paths = sorted(paths, key=natural_sort_key)
        preview_targets = (
            get_preview_target_indices(len(paths))
            if SAVE_PREVIEW
            else {}
        )

        print(
            f"\n[{sequence_index}/{len(sequences)}] "
            f"{label}/{sequence_id}: {len(paths)} frames"
        )

        for frame_index, image_path in enumerate(paths):
            image = cv2.imread(str(image_path))

            if image is None:
                print(f"อ่านภาพไม่ได้: {image_path}")
                failed_images.append(str(image_path))
                continue

            result = model.predict(
                source=image,
                conf=0.25,
                verbose=False,
            )[0]

            # บันทึกภาพตัวอย่าง 3 ช่วง (Early, Middle, Late)
            if SAVE_PREVIEW and frame_index in preview_targets:
                stage_name = preview_targets[frame_index]
                preview_dir = OUTPUT_DIR / "previews" / sequence_id
                preview_dir.mkdir(parents=True, exist_ok=True)

                preview = result.plot()
                preview_path = (
                    preview_dir / f"{stage_name}_frame_{frame_index:05d}.jpg"
                )
                cv2.imwrite(str(preview_path), preview)

            # ค่าเริ่มต้นกรณีไม่พบคน
            row = {
                "sequence_id": sequence_id,
                "frame_id": frame_index,
                "image_path": str(image_path),
                "label": label,
                "person_detected": 0,
                "person_confidence": float("nan"),
            }

            for keypoint_id in range(17):
                row[f"kp_{keypoint_id}_x"] = float("nan")
                row[f"kp_{keypoint_id}_y"] = float("nan")
                row[f"kp_{keypoint_id}_confidence"] = float("nan")

            if (
                result.keypoints is not None
                and result.keypoints.data.shape[0] > 0
            ):
                all_keypoints = (
                    result.keypoints.data.cpu().numpy()
                )

                # เลือกคนที่มี Bounding Box ใหญ่ที่สุด
                # เหมาะกับการทดลองแบบมีผู้สูงอายุหนึ่งคน
                if result.boxes is not None:
                    boxes = result.boxes.xyxy.cpu().numpy()
                    areas = (
                        (boxes[:, 2] - boxes[:, 0])
                        * (boxes[:, 3] - boxes[:, 1])
                    )
                    person_index = int(areas.argmax())
                else:
                    person_index = 0

                points = all_keypoints[person_index]

                row["person_detected"] = 1

                if result.boxes is not None:
                    row["person_confidence"] = float(
                        result.boxes.conf[person_index].item()
                    )

                for keypoint_id, point in enumerate(points):
                    x, y, confidence = point

                    row[f"kp_{keypoint_id}_x"] = float(x)
                    row[f"kp_{keypoint_id}_y"] = float(y)
                    row[
                        f"kp_{keypoint_id}_confidence"
                    ] = float(confidence)

            rows.append(row)

    # ======================================
    # 5. SAVE CSV
    # ======================================

    columns = make_columns()
    df = pd.DataFrame(rows, columns=columns)

    csv_path = OUTPUT_DIR / "keypoints.csv"
    df.to_csv(csv_path, index=False)

    # สรุปคุณภาพข้อมูล
    summary = (
        df.groupby(["label", "person_detected"])
        .size()
        .reset_index(name="frames")
    )
    summary_path = OUTPUT_DIR / "summary.csv"
    summary.to_csv(summary_path, index=False)

    print("\n========== FINISHED ==========")
    print(f"ภาพที่ประมวลผลได้: {len(df)}")
    print(f"ตรวจพบคน: {int(df['person_detected'].sum())}")
    print(f"ไม่พบคน: {int((df['person_detected'] == 0).sum())}")
    print(f"อ่านภาพไม่สำเร็จ: {len(failed_images)}")
    print(f"Keypoints CSV: {csv_path.resolve()}")
    print(f"Summary CSV: {summary_path.resolve()}")

    if failed_images:
        print("\nรายชื่อภาพที่อ่านไม่ได้:")
        for path in failed_images:
            print(path)


if __name__ == "__main__":
    main()
