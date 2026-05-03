import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.ensemble import RandomForestClassifier


def load_processed_dataset(processed_path: str) -> pd.DataFrame:
    df = pd.read_csv(processed_path, parse_dates=["timestamp"])  # timestamp present in processed
    df.set_index("timestamp", inplace=True)
    return df


def build_feature_list(df: pd.DataFrame) -> list:
    # mirror the pipeline's interpretable features + top 5 PCs (if available)
    key_features = [
        'sump_flow_lpm', 'airborne_gamma_uSvph', 'particulate_cpm',
        'makeup_flow_lpm', 'letdown_flow_lpm', 'rcs_mass_balance_lpm',
        'pressurizer_level_pct', 'pressurizer_pressure_MPa',
        'rcs_hot_leg_temp_C', 'rcs_cold_leg_temp_C', 'temp_gradient_C',
        'containment_humidity_pct', 'pump_vibration_mm_s',
        'radiation_index', 'hydraulic_imbalance_index', 'thermal_stress_index'
    ]
    pca_cols = [c for c in df.columns if c.startswith('pc_')]
    pca_cols = sorted(pca_cols)[:5]  # top 5 if available
    return [c for c in key_features if c in df.columns] + pca_cols


def plot_and_save_roc(y_true: np.ndarray, y_score: np.ndarray, out_path: str, title: str = "ROC Curve") -> float:
    auc = roc_auc_score(y_true, y_score)
    fpr, tpr, _ = roc_curve(y_true, y_score)
    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, label=f"AUC = {auc:.3f}")
    plt.plot([0, 1], [0, 1], 'k--', alpha=0.5)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(title)
    plt.legend(loc='lower right')
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    return auc


def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    processed_csv = os.path.join(base_dir, 'data', 'processed', 'nuclear_sensors_processed.csv')
    output_dir = os.path.join(base_dir, 'data', 'mining_results')
    os.makedirs(output_dir, exist_ok=True)

    df = load_processed_dataset(processed_csv)

    # Binary target
    y_binary = (df['ground_truth_leak_flag'] == True).astype(int)

    # Features
    features = build_feature_list(df)
    X = df[features].fillna(df[features].median())

    # Train/test split (same as pipeline)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_binary, test_size=0.2, random_state=42, stratify=y_binary
    )

    # Use RandomForest as in the pipeline
    clf = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
    clf.fit(X_train, y_train)
    y_score = clf.predict_proba(X_test)[:, 1]

    # Plot ROC
    out_png = os.path.join(output_dir, 'roc_visualization_random_forest.png')
    auc = plot_and_save_roc(y_test.values, y_score, out_png, title='ROC - Random Forest (Binary Leak Detection)')

    # Print a short summary so users can see AUC in console
    print(f"ROC AUC (Random Forest): {auc:.3f}")
    print(f"ROC curve saved to: {out_png}")


if __name__ == '__main__':
    main()




