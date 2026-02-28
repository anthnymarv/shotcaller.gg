"""
shot_caller_pipeline.py
-----------------------
Pipeline for training the Shot Caller objective prediction model.

Steps:
  1. Load & clean objective_training_data.csv
  2. Remove outliers (the "smooth world" filter)
  3. Standardize features with StandardScaler
  4. Select best features using Mutual Information
  5. Train & evaluate a Logistic Regression baseline
  6. Save the scaler, selected features, and model for live inference
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib

from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import mutual_info_classif
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import (
    classification_report,
    roc_auc_score,
    RocCurveDisplay,
    ConfusionMatrixDisplay,
)
from sklearn.pipeline import Pipeline

# ------------------------------------------------------------------
# Config
# ------------------------------------------------------------------

DATA_PATH    = os.path.join(os.path.dirname(__file__), 'data', 'objective_training_data.csv')
OUTPUT_DIR   = os.path.join(os.path.dirname(__file__), 'model')
TOP_K_FEATURES = 10          # how many MI-selected features to keep
OUTLIER_Z_THRESH = 3.5       # z-score threshold for "smooth world" outlier removal
CV_FOLDS = 5                 # cross-validation folds
RANDOM_STATE = 42

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ------------------------------------------------------------------
# Feature columns (everything except metadata / label)
# ------------------------------------------------------------------

FEATURE_COLS = [
    'level_diff',
    'ally_alive', 'enemy_alive',
    'game_time_minutes',
    'cs_per_min_diff', 'ally_cs_per_min', 'enemy_cs_per_min',
    'ally_completed_items', 'enemy_completed_items', 'item_advantage',
    'kills_last_2min_ally', 'kills_last_2min_enemy', 'kill_diff_recent',
    'ally_towers', 'enemy_towers', 'tower_diff',
    'time_since_last_fight',
    'is_infernal', 'is_mountain', 'is_ocean',
    'is_cloud', 'is_hextech', 'is_chemtech',
    'is_baron',
]

LABEL_COL = 'objective_secured'


# ==================================================================
# 1. Load data
# ==================================================================

def load_data(path: str) -> pd.DataFrame:
    print(f"Loading data from {path}...")
    df = pd.read_csv(path)
    print(f"  Raw shape: {df.shape}")
    print(f"  Label balance:\n{df[LABEL_COL].value_counts(normalize=True).round(3)}\n")
    return df


# ==================================================================
# 2. Clean data
# ==================================================================

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    print("Cleaning data...")
    original_len = len(df)

    # --- Drop rows with missing values in feature/label columns ---
    required_cols = FEATURE_COLS + [LABEL_COL]
    existing_cols = [c for c in required_cols if c in df.columns]
    df = df.dropna(subset=existing_cols)
    print(f"  After dropping NaNs: {len(df)} rows ({original_len - len(df)} removed)")

    # --- Drop rows from implausibly short games (laning phase only) ---
    if 'game_time_minutes' in df.columns:
        df = df[df['game_time_minutes'] >= 5]
        print(f"  After removing sub-5min rows: {len(df)} rows")

    # --- "Smooth world" outlier removal via z-score ---
    # Only applied to continuous diff/rate features, not one-hot or alive counts
    continuous_features = [
        'level_diff', 'cs_per_min_diff', 'item_advantage',
        'kill_diff_recent', 'tower_diff', 'time_since_last_fight',
        'ally_cs_per_min', 'enemy_cs_per_min',
    ]
    existing_continuous = [c for c in continuous_features if c in df.columns]

    z_scores = np.abs((df[existing_continuous] - df[existing_continuous].mean())
                      / df[existing_continuous].std())
    mask = (z_scores < OUTLIER_Z_THRESH).all(axis=1)
    removed = (~mask).sum()
    df = df[mask].reset_index(drop=True)
    print(f"  After z-score outlier removal (threshold={OUTLIER_Z_THRESH}): "
          f"{len(df)} rows ({removed} removed)\n")

    return df


# ==================================================================
# 3. Build feature matrix
# ==================================================================

def build_feature_matrix(df: pd.DataFrame):
    available = [c for c in FEATURE_COLS if c in df.columns]
    missing   = [c for c in FEATURE_COLS if c not in df.columns]
    if missing:
        print(f"  WARNING: missing feature columns (will be skipped): {missing}")

    X = df[available].values.astype(np.float32)
    y = df[LABEL_COL].values.astype(int)
    return X, y, available


# ==================================================================
# 4. Mutual Information feature selection
# ==================================================================

def select_features_mi(X: np.ndarray, y: np.ndarray,
                        feature_names: list, k: int) -> tuple:
    """
    Score all features by Mutual Information with the label,
    keep the top-k, and plot a bar chart for easy inspection.
    """
    print(f"Running Mutual Information feature selection (top {k} of {len(feature_names)})...")

    mi_scores = mutual_info_classif(X, y, random_state=RANDOM_STATE)
    mi_series = pd.Series(mi_scores, index=feature_names).sort_values(ascending=False)

    print("\n  MI scores (all features):")
    for feat, score in mi_series.items():
        bar = '█' * int(score * 200)
        print(f"    {feat:<30} {score:.4f}  {bar}")

    top_features = mi_series.head(k).index.tolist()
    top_indices  = [feature_names.index(f) for f in top_features]
    X_selected   = X[:, top_indices]

    print(f"\n  Selected features: {top_features}\n")

    # --- Plot ---
    fig, ax = plt.subplots(figsize=(9, 5))
    colors = ['#2ecc71' if f in top_features else '#bdc3c7' for f in mi_series.index]
    ax.barh(mi_series.index[::-1], mi_series.values[::-1], color=colors[::-1])
    ax.set_xlabel('Mutual Information Score')
    ax.set_title(f'Feature Importance (top {k} highlighted)')
    ax.axvline(mi_series.iloc[k - 1], color='#e74c3c', linestyle='--',
               label=f'Cut-off (top {k})')
    ax.legend()
    plt.tight_layout()
    plot_path = os.path.join(OUTPUT_DIR, 'mi_feature_importance.png')
    plt.savefig(plot_path, dpi=150)
    plt.close()
    print(f"  MI plot saved to {plot_path}")

    return X_selected, top_features


# ==================================================================
# 5. Train Logistic Regression baseline
# ==================================================================

def train_logistic_regression(X: np.ndarray, y: np.ndarray,
                               feature_names: list) -> Pipeline:
    """
    Trains a StandardScaler → LogisticRegression pipeline.

    - L2 regularization (C=1.0) to avoid overfitting to patch-specific data
    - StratifiedKFold CV to respect class imbalance
    - Reports accuracy, ROC-AUC, and a full classification report
    """
    print("Training Logistic Regression baseline...")

    model = Pipeline([
        ('scaler', StandardScaler()),
        ('clf',    LogisticRegression(
            penalty='l2',
            C=1.0,
            max_iter=1000,
            random_state=RANDOM_STATE,
            class_weight='balanced',   # handles slight class imbalance
        ))
    ])

    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)

    # --- Cross-validated metrics ---
    acc_scores = cross_val_score(model, X, y, cv=cv, scoring='accuracy')
    auc_scores = cross_val_score(model, X, y, cv=cv, scoring='roc_auc')

    print(f"\n  {CV_FOLDS}-Fold Cross-Validation Results:")
    print(f"    Accuracy : {acc_scores.mean():.3f} ± {acc_scores.std():.3f}")
    print(f"    ROC-AUC  : {auc_scores.mean():.3f} ± {auc_scores.std():.3f}")

    # --- Fit on full dataset for inspection & saving ---
    model.fit(X, y)

    # --- Coefficient table (interpretability) ---
    coefs = model.named_steps['clf'].coef_[0]
    coef_df = (pd.DataFrame({'feature': feature_names, 'coefficient': coefs})
               .sort_values('coefficient', key=abs, ascending=False))

    print("\n  Logistic Regression Coefficients (post-scaling = effect size):")
    for _, row in coef_df.iterrows():
        direction = '▲' if row['coefficient'] > 0 else '▼'
        print(f"    {direction} {row['feature']:<30} {row['coefficient']:+.4f}")

    # --- Full report on training data (sanity check, not generalization) ---
    y_pred = model.predict(X)
    y_prob = model.predict_proba(X)[:, 1]
    print(f"\n  Training set report (for sanity — use CV numbers for generalization):")
    print(classification_report(y, y_pred, target_names=['Failed', 'Secured']))
    print(f"  Training ROC-AUC: {roc_auc_score(y, y_prob):.3f}")

    return model


# ==================================================================
# 6. Plots
# ==================================================================

def plot_evaluation(model: Pipeline, X: np.ndarray, y: np.ndarray):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    RocCurveDisplay.from_estimator(model, X, y, ax=axes[0])
    axes[0].set_title('ROC Curve (training data)')
    axes[0].plot([0, 1], [0, 1], 'k--', alpha=0.4)

    ConfusionMatrixDisplay.from_estimator(
        model, X, y,
        display_labels=['Failed', 'Secured'],
        cmap='Blues', ax=axes[1]
    )
    axes[1].set_title('Confusion Matrix (training data)')

    plt.tight_layout()
    plot_path = os.path.join(OUTPUT_DIR, 'model_evaluation.png')
    plt.savefig(plot_path, dpi=150)
    plt.close()
    print(f"\n  Evaluation plots saved to {plot_path}")


# ==================================================================
# 7. Save artefacts
# ==================================================================

def save_artefacts(model: Pipeline, selected_features: list):
    model_path    = os.path.join(OUTPUT_DIR, 'shot_caller_lr.joblib')
    features_path = os.path.join(OUTPUT_DIR, 'selected_features.json')

    joblib.dump(model, model_path)
    print(f"  Model saved to {model_path}")

    import json
    with open(features_path, 'w') as f:
        json.dump(selected_features, f, indent=2)
    print(f"  Selected features saved to {features_path}")

    print("\n  To load and run inference:")
    print("    import joblib, json, numpy as np")
    print("    model    = joblib.load('model/shot_caller_lr.joblib')")
    print("    features = json.load(open('model/selected_features.json'))")
    print("    prob     = model.predict_proba(game_state_array)[0][1]")


# ==================================================================
# Main
# ==================================================================

def main():
    print("=" * 60)
    print("SHOT CALLER — Training Pipeline")
    print("=" * 60 + "\n")

    df = load_data(DATA_PATH)
    df = clean_data(df)

    X, y, feature_names = build_feature_matrix(df)
    print(f"Feature matrix: {X.shape[0]} samples × {X.shape[1]} features\n")

    X_selected, selected_features = select_features_mi(
        X, y, feature_names, k=TOP_K_FEATURES
    )

    model = train_logistic_regression(X_selected, y, selected_features)
    plot_evaluation(model, X_selected, y)
    save_artefacts(model, selected_features)

    print("\n" + "=" * 60)
    print("Pipeline complete.")
    print(f"Outputs written to: {OUTPUT_DIR}/")
    print("=" * 60)


if __name__ == '__main__':
    main()
