import pandas as pd

# อ่านไฟล์ CSV ที่อยู่ในโฟลเดอร์ output
df = pd.read_csv("output/keypoints.csv")

print("Shape:", df.shape)

print("\nColumns:")
print(df.columns.tolist())

print("\nLabel counts:")
print(df["label"].value_counts())

print("\nSequences per label:")
print(df.groupby("label")['sequence_id'].nunique())

print("\nFrames per sequence:")
print(
    df.groupby(["label", "sequence_id"]).size()
      .groupby("label")
      .describe()
)

print("\nDetection rate per label (% of frames with person detected):")
print(
    df.groupby("label")["person_detected"]
      .mean()
      .mul(100)
      .round(2)
)
