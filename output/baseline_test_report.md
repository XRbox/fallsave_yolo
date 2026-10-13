# FallSave Step 3.2: Final Baseline Test Evaluation Report

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

- **Overall Accuracy:** 93.32% (573 / 614 valid windows)
- **Macro F1 (Observable Classes: Normal & Falling):** 83.49%
- **Macro F1 (All 3 Classes, Lying=0):** 55.66%
- **Weighted F1:** 94.20%

### Per-Class Detailed Breakdown

| Class | Support | Precision | Recall | F1-Score | Evaluation Note |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **Normal** | 564 | 0.983 | 0.943 | 0.963 | Robust classification on majority ADL |
| **Falling** | 50 | 0.621 | 0.820 | 0.707 | **Safety Critical: Detected 41 of 50 active falls** |
| **Lying** | 0 | 0.000 | N/A | N/A | **Cannot be evaluated (support = 0)** |

---

## 3. Confusion Matrix (Test Set)

```
True / Pred  | normal   | falling  | lying   
---------------------------------------------
normal       | 532      | 25       | 7       
falling      | 9        | 41       | 0       
lying        | 0        | 0        | 0       
```

### Analysis of Predictions:
1. **Falling Recall = 82.0% (41 / 50):**
   - The model correctly identified 41 out of 50 active falling windows in the unseen Test sequences.
   - Only 9 falling windows were misclassified as normal, and 0 predicted as lying.
2. **False Positives on Normal:**
   - Out of 564 normal windows, 532 were correctly classified as normal (94.3% specificity).
   - 25 windows were predicted as falling and 7 as lying.
   - In our complete FallSave architecture, transient false positives are filtered out by the **State Machine persistence requirement and Confirmation Timer**.
3. **Lying Predictions:**
   - The model predicted the 'lying' label for 7 windows (7 from normal, 0 from falling). Since true ground-truth lying windows were occluded/lost by YOLO and excluded, precision on lying reflects these boundary frames.
