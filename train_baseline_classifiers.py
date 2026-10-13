"""
FallSave Phase 3 / Step 3.1: Classifier Baseline Training & Validation

Trains and evaluates baseline classifiers on the feature dataset:
1. Dummy Classifier (Most Frequent & Stratified)
2. Logistic Regression (balanced class weights, using normalized features)
3. Random Forest (balanced class weights, n_estimators=100)
4. Random Forest (regularized: max_depth=8, min_samples_leaf=3, balanced)

Evaluates strictly on Validation data. Test data is kept untouched.
"""

import os
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    classification_report,
    confusion_matrix
)

# -------------------------------------------------------------
# Configuration & Paths
# -------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TRAIN_PATH = os.path.join(BASE_DIR, "output", "features_train.csv")
VAL_PATH = os.path.join(BASE_DIR, "output", "features_validation.csv")
TEST_PATH = os.path.join(BASE_DIR, "output", "features_test.csv")
SCALER_PATH = os.path.join(BASE_DIR, "output", "feature_scaler.joblib")
MODELS_DIR = os.path.join(BASE_DIR, "models")
OUTPUT_DIR = os.path.join(BASE_DIR, "output")

os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

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


def load_and_prep_split(csv_path, split_name):
    """Loads CSV and extracts valid labeled windows, excluding rows labeled Excluded (-1)."""
    df = pd.read_csv(csv_path)
    total_rows = len(df)
    
    # Filter valid windows
    valid_mask = (df["label"] != -1) & (df["label_name"] != "excluded")
    df_valid = df[valid_mask].copy()
    
    class_counts = df_valid["label_name"].value_counts().to_dict()
    for c in CLASS_NAMES:
        if c not in class_counts:
            class_counts[c] = 0
            
    print(f"[{split_name.upper()} DATA]")
    print(f"  Total windows: {total_rows}")
    print(f"  Valid windows: {len(df_valid)}")
    print(f"  Excluded windows: {total_rows - len(df_valid)}")
    print(f"  Class breakdown: {class_counts}")
    
    return df_valid, class_counts


def evaluate_model(model, X_val, y_val, model_name):
    """Evaluates a model on validation data and returns structured metrics."""
    y_pred = model.predict(X_val)
    
    acc = accuracy_score(y_val, y_pred)
    
    # Macro metrics
    p_macro, r_macro, f1_macro, _ = precision_recall_fscore_support(
        y_val, y_pred, average="macro", zero_division=0
    )
    # Weighted metrics
    p_wt, r_wt, f1_wt, _ = precision_recall_fscore_support(
        y_val, y_pred, average="weighted", zero_division=0
    )
    # Per-class metrics
    p_per, r_per, f1_per, sup_per = precision_recall_fscore_support(
        y_val, y_pred, labels=[0, 1, 2], zero_division=0
    )
    
    cm = confusion_matrix(y_val, y_pred, labels=[0, 1, 2])
    
    per_class_dict = {}
    for idx, name in enumerate(CLASS_NAMES):
        per_class_dict[name] = {
            "precision": float(p_per[idx]),
            "recall": float(r_per[idx]),
            "f1": float(f1_per[idx]),
            "support": int(sup_per[idx])
        }
        
    metrics = {
        "model_name": model_name,
        "accuracy": float(acc),
        "macro_f1": float(f1_macro),
        "macro_precision": float(p_macro),
        "macro_recall": float(r_macro),
        "weighted_f1": float(f1_wt),
        "weighted_precision": float(p_wt),
        "weighted_recall": float(r_wt),
        "per_class": per_class_dict,
        "confusion_matrix": cm.tolist()
    }
    
    return metrics, y_pred


def format_cm_str(cm):
    """Formats confusion matrix for text output."""
    hdr_label = "True / Pred"
    header = f"{hdr_label:<12} | {'normal':<8} | {'falling':<8} | {'lying':<8}"
    sep = "-" * len(header)
    lines = [header, sep]
    for i, name in enumerate(CLASS_NAMES):
        lines.append(f"{name:<12} | {cm[i][0]:<8} | {cm[i][1]:<8} | {cm[i][2]:<8}")
    return "\n".join(lines)


def main():
    print("=" * 65)
    print(" FALLSAVE PHASE 3 / STEP 3.1: CLASSIFIER BASELINE TRAINING ")
    print("=" * 65)
    
    # 1. Load Data
    df_train, train_counts = load_and_prep_split(TRAIN_PATH, "train")
    df_val, val_counts = load_and_prep_split(VAL_PATH, "validation")
    
    # Check Test counts without touching features for modeling
    df_test_raw = pd.read_csv(TEST_PATH)
    test_valid = df_test_raw[df_test_raw["label"] != -1]
    test_counts = test_valid["label_name"].value_counts().to_dict()
    for c in CLASS_NAMES:
        if c not in test_counts:
            test_counts[c] = 0
            
    print(f"\n[TEST DATA (AUDITED BUT UNTOUCHED)]")
    print(f"  Total windows: {len(df_test_raw)}")
    print(f"  Valid windows: {len(test_valid)}")
    print(f"  Excluded windows: {len(df_test_raw) - len(test_valid)}")
    print(f"  Class breakdown: {test_counts}")
    print("  * Note: Lying class has 0 windows in Test set due to floor occlusion / YOLO FN.")
    
    # Features & Targets
    X_train_raw = df_train[FEATURE_COLS].values
    X_train_norm = df_train[NORM_FEATURE_COLS].values
    y_train = df_train["label"].values.astype(int)
    
    X_val_raw = df_val[FEATURE_COLS].values
    X_val_norm = df_val[NORM_FEATURE_COLS].values
    y_val = df_val["label"].values.astype(int)
    
    # 2. Define Baseline Models
    models = {
        "Dummy (Most Frequent)": {
            "model": DummyClassifier(strategy="most_frequent"),
            "features": "raw",
            "desc": "Predicts majority class (normal) unconditionally"
        },
        "Dummy (Stratified)": {
            "model": DummyClassifier(strategy="stratified", random_state=42),
            "features": "raw",
            "desc": "Predicts classes randomly matching training prior distribution"
        },
        "Logistic Regression (Balanced)": {
            "model": LogisticRegression(
                max_iter=1000,
                class_weight="balanced",
                random_state=42
            ),
            "features": "norm",
            "desc": "Linear multinomial baseline with balanced class weights on normalized features"
        },
        "Random Forest (Balanced, Default)": {
            "model": RandomForestClassifier(
                n_estimators=100,
                max_depth=None,
                min_samples_split=2,
                class_weight="balanced",
                random_state=42,
                n_jobs=-1
            ),
            "features": "raw",
            "desc": "Ensemble tree baseline with balanced class weights (unconstrained depth)"
        },
        "Random Forest (Balanced, Regularized)": {
            "model": RandomForestClassifier(
                n_estimators=100,
                max_depth=8,
                min_samples_leaf=3,
                class_weight="balanced",
                random_state=42,
                n_jobs=-1
            ),
            "features": "raw",
            "desc": "Regularized tree baseline (max_depth=8, min_samples_leaf=3) to prevent overfitting on minority classes"
        }
    }
    
    # 3. Train and Evaluate Each Model
    results = {}
    print("\n" + "=" * 65)
    print("             TRAINING AND VALIDATION EVALUATION")
    print("=" * 65)
    
    for name, m_info in models.items():
        print(f"\n>>> Model: {name}")
        print(f"    Description: {m_info['desc']}")
        
        # Select raw vs normalized
        X_tr = X_train_norm if m_info["features"] == "norm" else X_train_raw
        X_v = X_val_norm if m_info["features"] == "norm" else X_val_raw
        
        # Fit on Train ONLY
        m_info["model"].fit(X_tr, y_train)
        
        # Evaluate on Validation
        metrics, y_pred = evaluate_model(m_info["model"], X_v, y_val, name)
        results[name] = metrics
        
        print(f"    Validation Accuracy : {metrics['accuracy'] * 100:.2f}%")
        print(f"    Macro F1-Score      : {metrics['macro_f1'] * 100:.2f}%")
        print(f"    Weighted F1-Score   : {metrics['weighted_f1'] * 100:.2f}%")
        print("    Per-Class Performance:")
        for cname in CLASS_NAMES:
            c = metrics["per_class"][cname]
            print(f"      - {cname:<8}: P={c['precision']:.3f}, R={c['recall']:.3f}, F1={c['f1']:.3f} (support={c['support']})")
        
        print("    Confusion Matrix (Val):")
        print(format_cm_str(metrics["confusion_matrix"]))
        
        # Save model
        clean_name = name.lower().replace(" ", "_").replace("(", "").replace(")", "").replace(",", "")
        model_filename = os.path.join(MODELS_DIR, f"{clean_name}.joblib")
        joblib.dump(m_info["model"], model_filename)
        print(f"    Saved model checkpoint: {model_filename}")
        
    # 4. Feature Importance for Random Forest
    rf_model = models["Random Forest (Balanced, Default)"]["model"]
    importances = rf_model.feature_importances_
    sorted_idx = np.argsort(importances)[::-1]
    
    print("\n" + "=" * 65)
    print("      TOP 10 FEATURE IMPORTANCE (RANDOM FOREST DEFAULT)")
    print("=" * 65)
    feature_importance_list = []
    for rank, idx in enumerate(sorted_idx[:10], 1):
        feat_name = FEATURE_COLS[idx]
        imp_val = float(importances[idx])
        feature_importance_list.append({"rank": rank, "feature": feat_name, "importance": imp_val})
        print(f"  {rank:2d}. {feat_name:<25}: {imp_val * 100:.2f}%")
        
    # 5. Save Complete Results Summary
    summary_data = {
        "task": "FallSave Step 3.1: Classifier Baseline Training",
        "dataset_counts": {
            "train": {
                "total_valid": len(df_train),
                "by_class": train_counts
            },
            "validation": {
                "total_valid": len(df_val),
                "by_class": val_counts
            },
            "test_untouched": {
                "total_valid": len(test_valid),
                "by_class": test_counts,
                "limitation": "Zero valid Lying windows due to YOLO11 false negatives on ground-level poses."
            }
        },
        "feature_list": FEATURE_COLS,
        "models_evaluated": results,
        "rf_feature_importances": feature_importance_list
    }
    
    report_json_path = os.path.join(OUTPUT_DIR, "baseline_validation_results.json")
    with open(report_json_path, "w", encoding="utf-8") as f:
        json.dump(summary_data, f, indent=2)
    print(f"\nSaved structured validation report to: {report_json_path}")
    print("=" * 65)
    print(" Step 3.1 completed successfully. Test set remains untouched. ")
    print("=" * 65)


if __name__ == "__main__":
    main()
