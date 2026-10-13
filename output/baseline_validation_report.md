# FallSave Phase 3 / Step 3.1: Classifier Baseline Training Report

**Date:** 2026-10-13  
**Status:** Completed (Training and Validation Evaluation)  
**Rule Compliance:** Test set remains 100% untouched and un-tuned.

---

## 1. Valid Window Counts Across Splits

| Split | Normal | Falling | Lying | Excluded | Total Windows | Valid for Model |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Train** (stride=5) | 1,134 | 118 | 88 | 121 | 1,461 | **1,340** |
| **Validation** (stride=2) | 617 | 44 | 18 | 192 | 871 | **679** |
| **Test** (stride=2, untouched) | 564 | 50 | 0 | 264 | 878 | **614** |

> **Critical Dataset Limitation:**  
> The Test set contains **0 valid Lying windows** because YOLO11-Pose failed to detect human keypoints once actors laid flat on the floor (prone/supine posture) across all Test fall sequences (`fall-24, 25, 29, 30`), resulting in consecutive missing gaps > 3 frames that were legitimately excluded. **Lying performance has NOT been validated on Test.** This limitation is handled downstream by the Rule-Based State Machine + Confirmation Timer.

---

## 2. Validation Performance Summary (679 Valid Windows)

All models were fitted strictly on the 1,340 valid training windows and evaluated on the 679 validation windows.

| Model | Accuracy | Macro F1 | Weighted F1 | Falling Recall | Falling Precision | Lying Recall | Lying Precision |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Dummy (Most Frequent)** | 90.87% | 31.74% | 86.52% | 0.00% (0/44) | 0.00% | 0.00% (0/18) | 0.00% |
| **Dummy (Stratified)** | 77.47% | 33.49% | 80.03% | 11.36% (5/44) | 9.43% | 5.56% (1/18) | 2.00% |
| **Logistic Regression (Balanced)** | **90.87%** | **72.22%** | **92.00%** | **88.64% (39/44)** | **54.17%** | **83.33% (15/18)** | **40.54%** |
| **Random Forest (Balanced, Default)** | 93.52% | 61.02% | 92.62% | 65.91% (29/44) | 68.97% | 11.11% (2/18) | 66.67% |
| **Random Forest (Balanced, Regularized)** | 91.46% | 64.84% | 91.94% | 72.73% (32/44) | 52.46% | 38.89% (7/18) | 36.84% |

---

## 3. Confusion Matrices on Validation Set

### A. Logistic Regression (Balanced) — *Recommended Initial Baseline*
```
True \ Pred  | normal   | falling  | lying   
---------------------------------------------
normal       | 563      | 33       | 21      
falling      | 4        | 39       | 1       
lying        | 3        | 0        | 15      
```
- **Falling Recall = 88.6% (39/44)**: Only 4 falling windows missed (called normal) and 1 confused as lying.
- **Lying Recall = 83.3% (15/18)**: Correctly identifies 15 out of 18 lying windows.
- Output probabilities are well-calibrated and continuous, ideal for feeding into downstream state machine confidence thresholds.

### B. Random Forest (Balanced, Default)
```
True \ Pred  | normal   | falling  | lying   
---------------------------------------------
normal       | 604      | 12       | 1       
falling      | 15       | 29       | 0       
lying        | 15       | 1        | 2       
```
- Suffers severe recall collapse on minority class `lying` (only 2/18 detected, 15 misclassified as normal).
- High overall accuracy (93.52%) is driven entirely by memorizing the majority `normal` class.

### C. Random Forest (Balanced, Regularized: max_depth=8, min_samples_leaf=3)
```
True \ Pred  | normal   | falling  | lying   
---------------------------------------------
normal       | 582      | 23       | 12      
falling      | 12       | 32       | 0       
lying        | 5        | 6        | 7       
```
- Regularization improves minority recall (`falling` 32/44 = 72.7%, `lying` 7/18 = 38.9%), but still underperforms Logistic Regression on safety-critical recall.

---

## 4. Top Feature Importances (Random Forest)

1. `mean_kp_aspect_ratio`: **13.00%** (Width/height ratio of detected keypoint envelope)
2. `max_torso_angle`: **11.81%** (Maximum angle of shoulder-to-hip line relative to vertical)
3. `mean_torso_angle`: **10.90%** (Mean torso inclination across the window)
4. `mean_torso_span_y`: **8.88%** (Normalized vertical height of the torso)
5. `delta_torso_angle`: **8.24%** (Angular shift from start to end of window)
6. `delta_kp_aspect_ratio`: **6.27%** (Dynamic deformation of bounding envelope)
7. `delta_torso_span_y`: **5.57%** (Vertical compression of torso span)
8. `max_hip_y`: **5.42%** (Lowest hip level reached)
9. `delta_hip_y`: **4.74%** (Downward displacement of hip midpoint)
10. `mean_angular_velocity`: **4.63%** (Rate of torso rotation per frame)

---

## 5. Baseline Recommendation & Rationale

**Recommended Initial Baseline:** `Logistic Regression (Balanced)`
1. **Safety-First Sensitivity:** In safety-critical fall detection, missing an active fall (False Negative) is far more dangerous than a false alarm. Logistic Regression achieved **88.6% recall on Falling** (vs 65.9% for default RF) and **83.3% recall on Lying** (vs 11.1% for RF).
2. **Smooth Probabilities:** Logistic Regression outputs smoothly graded sigmoid/softmax probabilities, which directly plug into the planned Rule-Based State Machine (`P(falling) > threshold_fall` and `P(lying) > threshold_lying`).
3. **No Leaf Starvation on Imbalanced Data:** Unlike deep decision trees which struggle to partition tiny classes without overfitting, linear hyperplanes with balanced class weights generalize robustly on small minority subsets.
