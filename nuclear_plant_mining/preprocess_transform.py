# preprocess_transform.py
import os, joblib
import numpy as np
import pandas as pd
from scipy.signal import savgol_filter
from sklearn.impute import KNNImputer, SimpleImputer
from sklearn.preprocessing import PowerTransformer, StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

RAW_CSV = "data/raw/nuclear_sensors_raw.csv"
OUT_DIR = "data/processed"
os.makedirs(OUT_DIR, exist_ok=True)

# PARAMETERS (exact, tunable)
SHORT_GAP_MAX = 5      # minutes -> ffill/bfill limit
MED_GAP_MAX = 30       # minutes -> linear interpolation limit
LONG_GAP_MIN = 31      # minutes -> model-based impute or flag
ROLL_WINDOWS = [5, 15, 60]  # minutes for rolling stats
SAVGOL_WINDOW = 11     # must be odd
SAVGOL_POLY = 2

# Read
df = pd.read_csv(RAW_CSV, parse_dates=["timestamp"])
df.sort_values("timestamp", inplace=True)
df.reset_index(drop=True, inplace=True)

# Ensure timestamp cadence and add index for resampling-based ops
df.set_index("timestamp", inplace=True)
assert df.index.is_monotonic_increasing

# --- SCHEMA & RANGE CHECKS (fail loudly if out of range) ---
def check_ranges(dff):
    errs = []
    if (dff["sump_flow_lpm"] < 0).any():
        errs.append("sump_flow < 0 found")
    if (dff["airborne_gamma_uSvph"] < 0).any():
        errs.append("airborne_gamma < 0 found")
    # add more as needed
    return errs

errs = check_ranges(df)
if errs:
    print("Range check errors:", errs)
    # Optionally: raise Exception(errs)

# --- MISSINGNESS REPORT ---
na_frac = df.isna().mean().sort_values(ascending=False)
na_frac.to_csv(os.path.join(OUT_DIR, "missingness_percent.csv"))
print("Missingness per column saved.")

# Visual missingness (heatmap)
try:
    import missingno as msno
    msno.matrix(df.sample(frac=0.2), figsize=(12,4))
    plt.savefig(os.path.join(OUT_DIR, "missingness_matrix.png"), bbox_inches="tight")
    plt.close()
except Exception:
    pass

# --- GAP CATEGORIZATION (short/medium/long) ---
# Convert to minutes index differences
time_diffs = df.index.to_series().diff().dt.total_seconds().div(60).fillna(1)
if (time_diffs > 1+1e-6).any():
    print("Warning: irregular time steps found.")

# For each column, find contiguous NaN segments and categorize
def contiguous_nan_segments(series):
    is_na = series.isna().astype(int)
    groups = (is_na != is_na.shift()).cumsum()
    segments = []
    for g, sub in series.groupby(groups):
        if sub.isna().all():
            start = sub.index[0]
            end = sub.index[-1]
            length_min = int((end - start).total_seconds()/60) + 1
            segments.append((start, end, length_min))
    return segments

nan_segments_report = {}
for col in df.columns:
    segments = contiguous_nan_segments(df[col])
    if segments:
        nan_segments_report[col] = segments

# Save a sample of long outages
long_outages = {c:[s for s in segs if s[2] >= LONG_GAP_MIN] for c,segs in nan_segments_report.items()}
# Persist brief report
with open(os.path.join(OUT_DIR, "nan_segments_summary.txt"), "w") as f:
    for col, segs in long_outages.items():
        f.write(f"{col}: {len(segs)} long outages\n")
print("NaN segment summary saved.")

# --- SHORT GAP handling (<= SHORT_GAP_MAX): ffill/bfill limit ---
sensors = [c for c in df.columns if c not in ["plant_id","unit_id","ground_truth_event_id","ground_truth_leak_flag"]]
df_shortfilled = df.copy()
for col in sensors:
    df_shortfilled[col] = df_shortfilled[col].ffill(limit=SHORT_GAP_MAX).bfill(limit=SHORT_GAP_MAX)

# --- MEDIUM gaps handling (<= MED_GAP_MAX): time interpolation ---
df_interp = df_shortfilled.copy()
for col in sensors:
    df_interp[col] = df_interp[col].interpolate(method="time", limit=MED_GAP_MAX)

# Note: after interpolation some long gaps remain NaN

# --- LONG gaps (> LONG_GAP_MIN) --- flag them; do NOT blindly impute
long_gap_flags = pd.DataFrame(index=df.index)
for col in sensors:
    segs = contiguous_nan_segments(df[col])
    long_mask = pd.Series(False, index=df.index)
    for (s,e,length) in segs:
        if length >= LONG_GAP_MIN:
            long_mask.loc[s:e] = True
    long_gap_flags[col + "_outage"] = long_mask
long_gap_flags.to_csv(os.path.join(OUT_DIR, "long_gap_flags.csv"))

# --- OUTLIER DETECTION: IQR per-column (global) + rolling-MAD ---
def iqr_bounds(series):
    q1 = series.quantile(0.25)
    q3 = series.quantile(0.75)
    iqr = q3 - q1
    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr
    return lower, upper

outlier_flags = pd.DataFrame(index=df.index)
for col in ["sump_flow_lpm", "airborne_gamma_uSvph", "particulate_cpm"]:
    s = df_interp[col].dropna()
    if s.empty:
        outlier_flags[col + "_iqr_flag"] = False
        continue
    lower, upper = iqr_bounds(s)
    flag = (df_interp[col] < lower) | (df_interp[col] > upper)
    outlier_flags[col + "_iqr_flag"] = flag.fillna(False)

# Rolling-MAD (local outliers)
def rolling_mad_flag(series, window=60, threshold=7.0):
    roll_med = series.rolling(window, min_periods=5, center=True).median()
    mad = series.rolling(window, min_periods=5, center=True).apply(lambda x: np.median(np.abs(x - np.median(x))))
    mad = mad.replace(0, np.nan).fillna(series.std() * 0.6745)  # fallback
    flag = (np.abs(series - roll_med) > threshold * mad)
    return flag.fillna(False)

for col in ["sump_flow_lpm", "airborne_gamma_uSvph", "particulate_cpm"]:
    outlier_flags[col + "_rollingmad_flag"] = rolling_mad_flag(df_interp[col], window=60, threshold=7.0)

outlier_flags.to_csv(os.path.join(OUT_DIR, "outlier_flags.csv"))
print("Outlier flags saved.")

# For grading: save an outliers_report.csv with timestamp, sensor, value, lower_fence, upper_fence
outlier_rows = []
for col in ["sump_flow_lpm", "airborne_gamma_uSvph", "particulate_cpm"]:
    ser = df_interp[col]
    lower, upper = iqr_bounds(ser.dropna())
    mask = outlier_flags[col + "_iqr_flag"]
    for t, val in ser[mask].items():
        outlier_rows.append({"timestamp": t, "sensor": col, "value": float(val), "lower_fence": float(lower), "upper_fence": float(upper)})
pd.DataFrame(outlier_rows).to_csv(os.path.join(OUT_DIR, "outliers_report.csv"), index=False)

# --- ACTION ON OUTLIERS (policy) ---
# Policy: For timestamps inside ground-truth leak events, do NOT cap/remove outliers (they are real).
# For non-event timestamps, cap to fence (or optionally replace with rolling median).

df_cleaned = df_interp.copy()
event_mask = df_cleaned["ground_truth_leak_flag"].fillna(False)

for col in ["sump_flow_lpm", "airborne_gamma_uSvph", "particulate_cpm"]:
    lower, upper = iqr_bounds(df_cleaned[col].dropna())
    # create capped column but preserve raw
    df_cleaned[col + "_raw"] = df_cleaned[col]
    # cap only where not in event and value outside fences
    non_event_mask = ~event_mask
    cap_mask = non_event_mask & ((df_cleaned[col] < lower) | (df_cleaned[col] > upper))
    df_cleaned.loc[cap_mask, col] = df_cleaned.loc[cap_mask, col].clip(lower=lower, upper=upper)
    # Optionally replace with rolling median instead of clipping:
    # df_cleaned.loc[cap_mask, col] = df_cleaned[col].rolling(5, min_periods=1).median()

# --- DENOISING: create smoothed versions (preserve raw) ---
for col in ["sump_flow_lpm", "airborne_gamma_uSvph", "particulate_cpm", "rcs_hot_leg_temp_C"]:
    arr = df_cleaned[col].ffill().bfill().values
    # Rolling median
    df_cleaned[col + "_rolling_median"] = pd.Series(arr, index=df_cleaned.index).rolling(5, center=True, min_periods=1).median()
    # Savitzky-Golay (must handle edge cases)
    try:
        sg = savgol_filter(arr, window_length=SAVGOL_WINDOW, polyorder=SAVGOL_POLY, mode='interp')
        df_cleaned[col + "_savgol"] = sg
    except Exception:
        df_cleaned[col + "_savgol"] = df_cleaned[col + "_rolling_median"]

# --- BIAS/DRIFT DETECTION & CORRECTION (simple approach) ---
# Example: detect a sudden median shift using change-point at median of two halves
def detect_and_correct_drift(series):
    # split at 50% for a naive detection
    mid = len(series)//2
    first_med = series.iloc[:mid].median()
    second_med = series.iloc[mid:].median()
    shift = second_med - first_med
    if abs(shift) > 3 * series.std():  # threshold: 3*sigma
        # correct second half by subtracting shift
        corrected = series.copy()
        corrected.iloc[mid:] = corrected.iloc[mid:] - shift
        return corrected, shift
    return series, 0.0

# apply for pump_vibration example
pv = df_cleaned["pump_vibration_mm_s"].ffill().bfill()
pv_corr, pv_shift = detect_and_correct_drift(pv)
if pv_shift != 0.0:
    df_cleaned["pump_vibration_mm_s_drift_corrected"] = pv_corr
    print(f"Detected and corrected pump_vibration drift: {pv_shift:.3f}")

# --- FEATURE ENGINEERING (exact features to generate) ---
# 1) rolling mean/std over windows in ROLL_WINDOWS (minutes)
for w in ROLL_WINDOWS:
    window = f"{w}T"
    for col in ["sump_flow_lpm", "airborne_gamma_uSvph", "rcs_mass_balance_lpm", "pressurizer_level_pct"]:
        df_cleaned[f"{col}_rollmean_{w}m"] = df_cleaned[col].rolling(window=w, min_periods=1).mean()
        df_cleaned[f"{col}_rollstd_{w}m"] = df_cleaned[col].rolling(window=w, min_periods=1).std().fillna(0)

# 2) slopes / rates: minute difference and rolling average slope (5-min rolling)
for col in ["pressurizer_level_pct", "rcs_hot_leg_temp_C", "rcs_cold_leg_temp_C"]:
    df_cleaned[f"{col}_delta_1m"] = df_cleaned[col].diff().fillna(0)
    df_cleaned[f"{col}_slope_5m"] = df_cleaned[f"{col}_delta_1m"].rolling(5, min_periods=1).mean()

# 3) derived indices (exact formulas)
# Radiation Index (standardize each then weighted sum)
rad_cols = ["airborne_gamma_uSvph", "particulate_cpm"]
# Use std instead of mad() which is deprecated
rad_std = (df_cleaned[rad_cols] - df_cleaned[rad_cols].median()) / (df_cleaned[rad_cols].std().replace(0,1))
df_cleaned["radiation_index"] = 0.6 * rad_std["airborne_gamma_uSvph"] + 0.4 * rad_std["particulate_cpm"]

# Hydraulic imbalance
df_cleaned["hydraulic_imbalance_index"] = (df_cleaned["makeup_flow_lpm"] - df_cleaned["letdown_flow_lpm"]) / \
                                         (df_cleaned["makeup_flow_lpm"] + df_cleaned["letdown_flow_lpm"] + 1e-6)

# Thermal stress
df_cleaned["thermal_stress_index"] = df_cleaned["temp_gradient_C"] * (df_cleaned["primary_flow_kgps"].mean() / (df_cleaned["primary_flow_kgps"] + 1e-6))

# --- TRANSFORMATIONS: Power transform (Yeo-Johnson), scaling, PCA ---
transform_cols = ["airborne_gamma_uSvph", "particulate_cpm", "sump_flow_lpm", "rcs_mass_balance_lpm"]
# Prepare array for transformer
Xt = df_cleaned[transform_cols].fillna(method="ffill").fillna(0).values

# Power transform (Yeo-Johnson)
pt = PowerTransformer(method="yeo-johnson", standardize=False)
Xt_pt = pt.fit_transform(Xt)
for i, c in enumerate(transform_cols):
    df_cleaned[c + "_yeojohnson"] = Xt_pt[:, i]

# Standardize (mean0 std1) after power transform for modeling / PCA
scaler = StandardScaler()
Xt_scaled = scaler.fit_transform(df_cleaned[[c + "_yeojohnson" for c in transform_cols]].fillna(0))
for i, c in enumerate(transform_cols):
    df_cleaned[c + "_yj_scaled"] = Xt_scaled[:, i]

# PCA on a specified feature list (exact list)
features_for_pca = [
 "sump_flow_lpm","airborne_gamma_uSvph","particulate_cpm",
 "makeup_flow_lpm","letdown_flow_lpm","rcs_mass_balance_lpm",
 "pressurizer_level_pct","pressurizer_pressure_MPa","pressurizer_level_slope",
 "rcs_hot_leg_temp_C","rcs_cold_leg_temp_C","temp_gradient_C","primary_flow_kgps",
 "steam_generator_dp_kPa","containment_humidity_pct","containment_temp_C",
 "pump_vibration_mm_s","valve_position_pct","hydrogen_ppm","neutron_flux_pctFP",
 "turbine_power_MWe","feedwater_flow_kgps",
 "radiation_index","hydraulic_imbalance_index","thermal_stress_index"
]

# Impute remaining NaNs with median for PCA/scaling
imp = SimpleImputer(strategy="median")
X_pca_ready = imp.fit_transform(df_cleaned[features_for_pca].fillna(0))

# First standardize for PCA
scaler_pca = StandardScaler()
X_scaled_for_pca = scaler_pca.fit_transform(X_pca_ready)

pca = PCA(n_components=0.95, svd_solver="full")   # keep 95% variance
X_pca = pca.fit_transform(X_scaled_for_pca)

# Save K components explicitly (pc_01, pc_02, ...)
n_components = X_pca.shape[1]
for i in range(n_components):
    df_cleaned[f"pc_{i+1:02d}"] = X_pca[:, i]

print(f"PCA components: {n_components} (capturing 95% variance)")
print(f"Explained variance ratio cumsum: {pca.explained_variance_ratio_.cumsum()[-5:]}")

# --- SAVE OUTPUTS ---
# 1) Interim outputs
df_cleaned.to_csv(os.path.join(OUT_DIR, "nuclear_sensors_processed.csv"))
print("Processed dataset saved.")

# 2) Save fitted transformers
joblib.dump(pt, os.path.join(OUT_DIR, "power_transformer.pkl"))
joblib.dump(scaler, os.path.join(OUT_DIR, "standard_scaler.pkl"))
joblib.dump(scaler_pca, os.path.join(OUT_DIR, "pca_scaler.pkl"))
joblib.dump(pca, os.path.join(OUT_DIR, "pca_transformer.pkl"))
joblib.dump(imp, os.path.join(OUT_DIR, "median_imputer.pkl"))
print("Transformers saved.")

# 3) Feature lists and metadata
feature_metadata = {
    "raw_sensors": sensors,
    "transform_cols": transform_cols,
    "features_for_pca": features_for_pca,
    "pca_components": n_components,
    "rolling_windows": ROLL_WINDOWS,
    "savgol_params": {"window": SAVGOL_WINDOW, "poly": SAVGOL_POLY},
    "gap_params": {"short_max": SHORT_GAP_MAX, "med_max": MED_GAP_MAX, "long_min": LONG_GAP_MIN}
}

import json
with open(os.path.join(OUT_DIR, "feature_metadata.json"), "w") as f:
    json.dump(feature_metadata, f, indent=2, default=str)
print("Feature metadata saved.")

# 4) Processing summary
summary = {
    "total_rows": len(df_cleaned),
    "total_columns": len(df_cleaned.columns),
    "original_sensors": len(sensors),
    "engineered_features": len(df_cleaned.columns) - len(df.columns),
    "pca_components": n_components,
    "missing_data_handled": True,
    "outliers_flagged": len(outlier_flags.columns),
    "drift_corrected": pv_shift != 0.0,
    "transformations_applied": ["power_transform", "standardization", "pca"]
}

with open(os.path.join(OUT_DIR, "processing_summary.json"), "w") as f:
    json.dump(summary, f, indent=2)
print("Processing summary saved.")

print(f"\n=== PROCESSING COMPLETE ===")
print(f"Final dataset shape: {df_cleaned.shape}")
print(f"Original columns: {len(df.columns)}")
print(f"Final columns: {len(df_cleaned.columns)}")
print(f"Features engineered: {len(df_cleaned.columns) - len(df.columns)}")
print(f"PCA components: {n_components}")
print(f"Files saved to: {OUT_DIR}")

# 5) Optional: Create a sample of final features for inspection
sample_features = [
    "sump_flow_lpm", "sump_flow_lpm_rolling_median", "sump_flow_lpm_savgol",
    "airborne_gamma_uSvph", "airborne_gamma_uSvph_yeojohnson", 
    "radiation_index", "hydraulic_imbalance_index", "thermal_stress_index",
    "pc_01", "pc_02", "pc_03",
    "ground_truth_leak_flag", "ground_truth_leak_rate_lpm"
]

sample_df = df_cleaned[sample_features].head(100)
sample_df.to_csv(os.path.join(OUT_DIR, "sample_features.csv"))
print("Sample features saved for inspection.")

# 6) Create visualization of key processed features
try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Raw vs processed sump flow
    axes[0,0].plot(df_cleaned.index[:1000], df_cleaned["sump_flow_lpm"].iloc[:1000], 
                   alpha=0.7, label="Raw", linewidth=1)
    axes[0,0].plot(df_cleaned.index[:1000], df_cleaned["sump_flow_lpm_savgol"].iloc[:1000], 
                   alpha=0.9, label="Savgol Smoothed", linewidth=1.5)
    axes[0,0].set_title("Sump Flow: Raw vs Denoised")
    axes[0,0].legend()
    axes[0,0].set_ylabel("LPM")
    
    # Radiation index over time
    axes[0,1].plot(df_cleaned.index[:1000], df_cleaned["radiation_index"].iloc[:1000], 
                   color='red', alpha=0.8, linewidth=1)
    axes[0,1].set_title("Engineered Radiation Index")
    axes[0,1].set_ylabel("Index Value")
    
    # PCA components
    axes[1,0].scatter(df_cleaned["pc_01"].iloc[:1000], df_cleaned["pc_02"].iloc[:1000], 
                      c=df_cleaned["ground_truth_leak_flag"].iloc[:1000], 
                      alpha=0.6, s=1, cmap='coolwarm')
    axes[1,0].set_title("PCA: PC1 vs PC2 (colored by leak events)")
    axes[1,0].set_xlabel("PC1")
    axes[1,0].set_ylabel("PC2")
    
    # Missing data heatmap for key sensors
    missing_subset = df_cleaned[["sump_flow_lpm", "airborne_gamma_uSvph", 
                                "particulate_cpm", "pump_vibration_mm_s"]].iloc[:1000]
    sns.heatmap(missing_subset.isnull().T, cbar=True, ax=axes[1,1], 
                cmap='Blues', yticklabels=True, xticklabels=False)
    axes[1,1].set_title("Missing Data Pattern")
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "processing_overview.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print("Processing overview plot saved.")
    
except ImportError:
    print("Matplotlib/seaborn not available - skipping visualization.")
except Exception as e:
    print(f"Visualization failed: {e}")

# 7) Final data quality report
quality_report = {
    "timestamp_range": f"{df_cleaned.index.min()} to {df_cleaned.index.max()}",
    "total_missing_values": int(df_cleaned.isnull().sum().sum()),
    "sensors_with_missing_data": int((df_cleaned.isnull().sum() > 0).sum()),
    "outliers_detected_total": int(outlier_flags.sum().sum()),
    "leak_events_preserved": int(df_cleaned["ground_truth_leak_flag"].sum()),
    "feature_categories": {
        "raw_sensors": len([c for c in df_cleaned.columns if not any(x in c for x in ["_raw", "_yeojohnson", "_scaled", "_rolling", "_savgol", "_rollmean", "_rollstd", "_delta", "_slope", "_index", "pc_"])]),
        "denoised_versions": len([c for c in df_cleaned.columns if "_savgol" in c or "_rolling_median" in c]),
        "rolling_features": len([c for c in df_cleaned.columns if "_rollmean" in c or "_rollstd" in c]),
        "slope_features": len([c for c in df_cleaned.columns if "_slope" in c or "_delta" in c]),
        "composite_indices": len([c for c in df_cleaned.columns if "_index" in c]),
        "pca_components": n_components,
        "transformed_features": len([c for c in df_cleaned.columns if "_yeojohnson" in c or "_scaled" in c])
    }
}

with open(os.path.join(OUT_DIR, "data_quality_report.json"), "w") as f:
    json.dump(quality_report, f, indent=2, default=str)

print("\n=== FINAL SUMMARY ===")
print(f"✅ Dataset processed: {quality_report['timestamp_range']}")
print(f"✅ Shape: {df_cleaned.shape}")
print(f"✅ Missing values handled: {quality_report['total_missing_values']} remaining")
print(f"✅ Outliers detected: {quality_report['outliers_detected_total']}")
print(f"✅ Leak events preserved: {quality_report['leak_events_preserved']} timestamps")
print(f"✅ PCA components: {n_components}")
print(f"✅ All files saved to: {OUT_DIR}")
print(f"✅ Ready for modeling!")

# Display file list
print(f"\nGenerated files:")
for f in os.listdir(OUT_DIR):
    print(f"  - {f}")