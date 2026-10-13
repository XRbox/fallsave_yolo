from pathlib import Path
import cv2
import pandas as pd
from ultralytics import YOLO

# 1. ระบุภาพ PNG ที่ต้องการทดสอบ
image_path = Path("dataset/fall/fall-01-cam0-rgb/fall-01-cam0-rgb-001.png")

# 2. ตรวจสอบว่ามีไฟล์จริง
if not image_path.exists():
    raise FileNotFoundError(
        f"ไม่พบไฟล์ภาพ: {image_path.resolve()}"
    )

# 3. โหลด YOLO11n-Pose
# ครั้งแรกโปรแกรมจะดาวน์โหลดโมเดลให้อัตโนมัติ
print("กำลังโหลดโมเดล YOLO11n-Pose...")
model = YOLO("yolo11n-pose.pt")

# 4. ตรวจจับคนและ Keypoints
print(f"กำลังประมวลผลภาพ: {image_path}...")
results = model.predict(
    source=str(image_path),
    conf=0.25,
    verbose=False
)
result = results[0]

# 5. บันทึกภาพที่วาด Skeleton แล้ว
output_dir = Path("output")
output_dir.mkdir(parents=True, exist_ok=True)
annotated_image = result.plot()
cv2.imwrite(
    str(output_dir / "pose_result.jpg"),
    annotated_image
)

# 6. แปลง Keypoints เป็นตาราง
rows = []
if result.keypoints is not None:
    keypoints = result.keypoints.data.cpu().numpy()
    for person_id, person in enumerate(keypoints):
        for keypoint_id, point in enumerate(person):
            x, y, confidence = point
            rows.append({
                "person_id": person_id,
                "keypoint_id": keypoint_id,
                "x": float(x),
                "y": float(y),
                "confidence": float(confidence)
            })

# 7. บันทึกเป็น CSV
columns = [
    "person_id", "keypoint_id",
    "x", "y", "confidence"
]
df = pd.DataFrame(rows, columns=columns)
csv_path = output_dir / "keypoints.csv"
df.to_csv(csv_path, index=False)

print("สำเร็จเรียบร้อย!")
print(f"1. ภาพ Skeleton บันทึกที่: {output_dir / 'pose_result.jpg'}")
print(f"2. ไฟล์ตาราง Keypoints บันทึกที่: {csv_path}")
print(f"ตรวจพบคนทั้งหมด: {len(result.boxes) if result.boxes is not None else 0} คน")
print(f"จำนวน Keypoint rows: {len(rows)} แถว")
