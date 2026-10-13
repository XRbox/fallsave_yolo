import pandas as pd

CSV_PATH = "output/keypoints.csv"

df = pd.read_csv(CSV_PATH)
fall = df[df["label"] == "fall"].copy()

# ความยาวของแต่ละ Fall sequence
seq_lengths = (
    fall.groupby("sequence_id")
    .size()
    .sort_values()
)

print("=== Fall sequences: shortest ===")
print(seq_lengths.head(5))

print("\n=== Fall sequences: median ===")
mid = len(seq_lengths) // 2
print(seq_lengths.iloc[max(0, mid - 2):mid + 3])

print("\n=== Fall sequences: longest ===")
print(seq_lengths.tail(5))
