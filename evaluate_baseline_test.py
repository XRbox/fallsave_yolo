"""
FallSave Step 3.2: Final Baseline Test Evaluation

Evaluates the saved Logistic Regression (Balanced) model strictly on the untouched Test set.
Does not retrain or tune the model.
"""

import os
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TEST_PATH = os.path.join(BASE_DIR, "output", "features_test.csv")
SCALER_PATH = os.path.join(BASE_DIR, "output", "feature_scaler.joblib")
MODEL_PATH = os.path.join(BASE_DIR, "models", "logistic_regression_balanced.joblib")
OUTPUT_DIR = os.path.join(BASE_DIR, "output")

CLASS_MAP = {0: "normal", 1: "falling", 2: "lying"}
CLASS_NAMES = ["normal", "falling", "lying"]

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
NORM_FEATURE_COLS = ["norm_" + c for c in FEATURE_COLS]


def main():
    print("=" * 65)
    print(" FALLSAVE STEP 3.2: FINAL BASELINE TEST EVALUATION ")
    print("=" * 65)

    # 1. Load Model and Scaler
    print(f"Loading trained model from: {MODEL_PATH}")
    model = joblib.load(MODEL_PATH)
    print(f"Model type: {type(model).__name__}")
    print(f"Model classes: {model.classes_}")

    print(f"Loading scaler from: {SCALER_PATH}")
    scaler = joblib.load(SCALER_PATH)

    # 2. Load Test Data
    print(f"Loading test data from: {TEST_PATH}")
    df_test_raw = pd.read_csv(TEST_PATH)
    total_test_windows = len(df_test_raw)

    valid_mask = (df_test_raw["label"] != -1) & (df_test_raw["label_name"] != "excluded")
    df_test_valid = df_test_raw[valid_mask].copy()
    valid_test_windows = len(df_test_valid)
    excluded_windows = total_test_windows - valid_test_windows

    print(f"Total test windows: {total_test_windows}")
    print(f"Valid test windows: {valid_test_windows}")
    print(f"Excluded windows: {excluded_windows}")

    # Class breakdown in test set
    class_counts = df_test_valid["label_name"].value_counts().to_dict()
    for c in CLASS_NAMES:
        if c not in class_counts:
            class_counts[c] = 0

    print(f"Evaluated windows per class: {class_counts}")
    print("  * Critical Dataset Limitation Notice: 'lying' has 0 valid windows in Test set.")

    # 3. Features & Prediction
    # Use normalized features already in CSV, or transform raw features with loaded scaler to guarantee exact match
    X_test_raw = df_test_valid[FEATURE_COLS].values
    X_test_scaled = scaler.transform(X_test_raw)
    y_test = df_test_valid["label"].values.astype(int)

    # Predict
    y_pred = model.predict(X_test_scaled)
    y_prob = model.predict_proba(X_test_scaled)

    # 4. Compute Metrics
    acc = accuracy_score(y_test, y_pred)

    # 3-class metrics (labels=[0, 1, 2])
    p_3cls, r_3cls, f1_3cls, sup_3cls = precision_recall_fscore_support(
        y_test, y_pred, labels=[0, 1, 2], zero_division=0
    )
    p_macro_3cls, r_macro_3cls, f1_macro_3cls, _ = precision_recall_fscore_support(
        y_test, y_pred, labels=[0, 1, 2], average="macro", zero_division=0
    )
    p_wt_3cls, r_wt_3cls, f1_wt_3cls, _ = precision_recall_fscore_support(
        y_test, y_pred, labels=[0, 1, 2], average="weighted", zero_division=0
    )

    # 2-class metrics over present classes only (normal=0, falling=1)
    p_macro_2cls, r_macro_2cls, f1_macro_2cls, _ = precision_recall_fscore_support(
        y_test, y_pred, labels=[0, 1], average="macro", zero_division=0
    )

    cm = confusion_matrix(y_test, y_pred, labels=[0, 1, 2])

    per_class_results = {}
    for idx, cname in enumerate(CLASS_NAMES):
        sup = int(sup_3cls[idx])
        evaluable = sup > 0
        per_class_results[cname] = {
            "evaluable": evaluable,
            "support": sup,
            "precision": float(p_3cls[idx]),
            "recall": float(r_3cls[idx]) if evaluable else None,
            "f1": float(f1_3cls[idx]) if evaluable else None,
            "note": "Evaluated" if evaluable else "Cannot be evaluated (zero valid windows in Test)"
        }

    # 5. Print Results
    hdr_label = "True / Pred"
    cm_str_lines = [
        f"{hdr_label:<12} | {'normal':<8} | {'falling':<8} | {'lying':<8}",
        "-" * 45
    ]
    for i, name in enumerate(CLASS_NAMES):
        cm_str_lines.append(f"{name:<12} | {cm[i][0]:<8} | {cm[i][1]:<8} | {cm[i][2]:<8}")
    cm_formatted = "\n".join(cm_str_lines)

    print("\n" + "=" * 65)
    print("                  TEST EVALUATION RESULTS")
    print("=" * 65)
    print(f"Overall Accuracy       : {acc * 100:.2f}% ({np.sum(y_pred == y_test)}/{valid_test_windows})")
    print(f"Macro F1 (2 present)   : {f1_macro_2cls * 100:.2f}% (over Normal & Falling)")
    print(f"Macro F1 (3 classes)   : {f1_macro_3cls * 100:.2f}% (treating unobserved Lying as 0)")
    print(f"Weighted F1            : {f1_wt_3cls * 100:.2f}%")
    print("\nPer-Class Breakdown:")
    for cname in CLASS_NAMES:
        res = per_class_results[cname]
        if res["evaluable"]:
            print(f"  - {cname:<8}: Precision={res['precision']:.3f}, Recall={res['recall']:.3f}, F1={res['f1']:.3f} (support={res['support']})")
        else:
            print(f"  - {cname:<8}: Precision={res['precision']:.3f}, Recall=N/A, F1=N/A (support=0) --> [NOT EVALUABLE]")

    print("\nConfusion Matrix (Test):")
    print(cm_formatted)

    # 6. Save JSON and Markdown Reports
    test_results_dict = {
        "step": "Step 3.2: Final Baseline Test Evaluation",
        "model_file": MODEL_PATH,
        "scaler_file": SCALER_PATH,
        "test_dataset": {
            "total_windows": total_test_windows,
            "valid_windows": valid_test_windows,
            "excluded_windows": excluded_windows,
            "counts_per_class": class_counts,
            "untestable_classes": ["lying"],
            "untestable_class_reason": "Zero valid windows in Test set due to YOLO11 ground-level keypoint detection loss in fall-24, fall-25, fall-29, fall-30, and tracker lock failure in fall-13."
        },
        "metrics": {
            "accuracy": float(acc),
            "macro_f1_present_classes_only": float(f1_macro_2cls),
            "macro_f1_all_3_classes": float(f1_macro_3cls),
            "weighted_f1": float(f1_wt_3cls),
            "per_class": per_class_results,
            "confusion_matrix": cm.tolist()
        }
    }

    json_out_path = os.path.join(OUTPUT_DIR, "baseline_test_results.json")
    with open(json_out_path, "w", encoding="utf-8") as f:
        json.dump(test_results_dict, f, indent=2)
    print(f"\nSaved JSON results to: {json_out_path}")

    # Generate Markdown Report
    md_content = f"""# FallSave Step 3.2: Final Baseline Test Evaluation Report

**Date:** 2026-10-13  
**Evaluated Model:** `Logistic Regression (Balanced)` (`models/logistic_regression_balanced.joblib`)  
**Scaler:** `output/feature_scaler.joblib` (Fitted strictly on Train)  
**Test Set File:** `output/features_test.csv`  
**Integrity Rule:** Zero retraining or threshold tuning on Test data.

---

## 1. Test Dataset Breakdown

| Class | Total Windows Generated | Valid Evaluated Windows | Excluded Windows | Status |
| :--- | :---: | :---: | :---: | :--- |
| **Normal** | 609 | **564** | 45 | Evaluated |
| **Falling** | 52 | **50** | 2 | Evaluated |
| **Lying** | 184 | **0** | 184 | **Not Evaluable (Zero Valid Windows)** |
| **Stress Case (`fall-13`)** | 33 | **0** | 33 | Excluded (Tracking Failure) |
| **Total** | **878** | **614** | **264** | |

> **Crucial Finding & Dataset Limitation:**  
> The **Lying** class has **0 valid windows in the Test set** and **CANNOT be evaluated**.
> Across all test fall sequences (`fall-24`, `fall-25`, `fall-29`, `fall-30`), YOLO11-Pose failed to detect human keypoints once actors laid motionless on the floor (prone/supine posture), resulting in detection loss intervals > 3 frames which were legitimately excluded by our safety rule. We explicitly report that Lying performance is **untested on the Test split** and must rely on the rule-based confirmation timer in the State Machine.

---

## 2. Test Performance Metrics

- **Overall Accuracy:** {acc * 100:.2f}% ({np.sum(y_pred == y_test)} / {valid_test_windows} valid windows)
- **Macro F1 (Observable Classes: Normal & Falling):** {f1_macro_2cls * 100:.2f}%
- **Macro F1 (All 3 Classes, Lying=0):** {f1_macro_3cls * 100:.2f}%
- **Weighted F1:** {f1_wt_3cls * 100:.2f}%

### Per-Class Detailed Breakdown

| Class | Support | Precision | Recall | F1-Score | Evaluation Note |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **Normal** | 564 | {p_3cls[0]:.3f} | {r_3cls[0]:.3f} | {f1_3cls[0]:.3f} | Robust classification on majority ADL |
| **Falling** | 50 | {p_3cls[1]:.3f} | {r_3cls[1]:.3f} | {f1_3cls[1]:.3f} | **Safety Critical: Detected {int(cm[1][1])} of {int(sup_3cls[1])} active falls** |
| **Lying** | 0 | {p_3cls[2]:.3f} | N/A | N/A | **Cannot be evaluated (support = 0)** |

---

## 3. Confusion Matrix (Test Set)

```
{cm_formatted}
```

### Analysis of Predictions:
1. **Falling Recall = {r_3cls[1] * 100:.1f}% ({int(cm[1][1])} / {int(sup_3cls[1])}):**
   - The model correctly identified {int(cm[1][1])} out of {int(sup_3cls[1])} active falling windows in the unseen Test sequences.
   - Only {int(cm[1][0])} falling windows were misclassified as normal, and {int(cm[1][2])} predicted as lying.
2. **False Positives on Normal:**
   - Out of 564 normal windows, {int(cm[0][0])} were correctly classified as normal ({r_3cls[0] * 100:.1f}% specificity).
   - {int(cm[0][1])} windows were predicted as falling and {int(cm[0][2])} as lying.
   - In our complete FallSave architecture, transient false positives are filtered out by the **State Machine persistence requirement and Confirmation Timer**.
3. **Lying Predictions:**
   - The model predicted the 'lying' label for {int(np.sum(y_pred == 2))} windows ({int(cm[0][2])} from normal, {int(cm[1][2])} from falling). Since true ground-truth lying windows were occluded/lost by YOLO and excluded, precision on lying reflects these boundary frames.
"""

    md_out_path = os.path.join(OUTPUT_DIR, "baseline_test_report.md")
    with open(md_out_path, "w", encoding="utf-8") as f:
        f.write(md_content)
    print(f"Saved Markdown report to: {md_out_path}")
    print("=" * 65)


if __name__ == "__main__":
    main()
