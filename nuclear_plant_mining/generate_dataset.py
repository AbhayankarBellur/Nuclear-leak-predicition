# generate_dataset.py
import os
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

RNG_SEED = 42
np.random.seed(RNG_SEED)

OUT_DIR = "data"
RAW_DIR = os.path.join(OUT_DIR, "raw")
os.makedirs(RAW_DIR, exist_ok=True)

# CONFIG
DAYS = 14
FREQ = "1min"
START = "2025-08-01T00:00:00"   # change if you want different start
periods = DAYS * 24 * 60

# TIME INDEX
ts = pd.date_range(start=START, periods=periods, freq=FREQ)

# BASELINES (realistic-ish ranges)
n = len(ts)
def baseline(mean, sd, daily_amp=0.0):
    t = np.arange(n)
    arr = np.random.normal(loc=mean, scale=sd, size=n)
    if daily_amp:
        arr += daily_amp * np.sin(2*np.pi*t/(24*60))  # gentle diurnal
    return arr

df = pd.DataFrame({
    "timestamp": ts,
    "plant_id": "Plant-A",
    "unit_id": "Unit-1",
    "sump_flow_lpm": np.random.exponential(scale=0.02, size=n),        # mostly near 0
    "airborne_gamma_uSvph": baseline(0.12, 0.02, daily_amp=0.02),
    "particulate_cpm": np.random.poisson(lam=18, size=n).astype(float),
    "makeup_flow_lpm": baseline(150.0, 2.5),
    "letdown_flow_lpm": baseline(148.0, 3.0),
    "pressurizer_level_pct": baseline(50.0, 0.6),
    "pressurizer_pressure_MPa": baseline(15.5, 0.08),
    "rcs_hot_leg_temp_C": baseline(290.0, 0.6),
    "rcs_cold_leg_temp_C": baseline(271.0, 0.6),
    "primary_flow_kgps": baseline(5000.0, 40.0, daily_amp=10.0),
    "steam_generator_dp_kPa": baseline(100.0, 3.0),
    "containment_humidity_pct": baseline(45.0, 4.0, daily_amp=3.0),
    "containment_temp_C": baseline(30.0, 0.6, daily_amp=1.0),
    "pump_vibration_mm_s": baseline(0.5, 0.08),
    "valve_position_pct": np.clip(baseline(50.0, 5.0), 0, 100),
    "hydrogen_ppm": baseline(1.1, 0.15),
    "neutron_flux_pctFP": baseline(80.0, 1.5),
    "turbine_power_MWe": baseline(1000.0, 8.0, daily_amp=20.0),
    "feedwater_flow_kgps": baseline(3000.0, 25.0)
})

# DERIVED columns (initial)
df["rcs_mass_balance_lpm"] = df["makeup_flow_lpm"] - df["letdown_flow_lpm"]
df["temp_gradient_C"] = df["rcs_hot_leg_temp_C"] - df["rcs_cold_leg_temp_C"]

# --- INJECT LEAK EVENTS (8 events randomized) ---
events = []
num_events = 8
min_dur_min = 30         # minutes
max_dur_min = 6*60      # 6 hours
available_idx = np.arange(0, n - max_dur_min - 1)
starts = np.random.choice(available_idx, size=num_events, replace=False)

for i, s in enumerate(starts):
    dur = np.random.randint(min_dur_min, max_dur_min)
    e = min(s + dur, n-1)
    leak_type = np.random.choice(["gradual", "rapid"])
    if leak_type == "gradual":
        leak_size = np.random.uniform(3.0, 40.0)
        ramp = np.linspace(0, leak_size, e - s + 1)
    else:
        leak_size = np.random.uniform(5.0, 80.0)
        ramp = np.ones(e - s + 1) * leak_size

    # apply
    df.loc[s:e, "sump_flow_lpm"] += ramp
    df.loc[s:e, "airborne_gamma_uSvph"] += (np.sqrt(leak_size) * np.linspace(0.2,1.2, e-s+1) +
                                           np.random.normal(0, 0.05, e-s+1)).clip(min=0)
    df.loc[s:e, "particulate_cpm"] += (np.random.poisson(lam=50, size=e-s+1) * (leak_size/10.0))
    df.loc[s:e, "makeup_flow_lpm"] += ramp
    df.loc[s:e, "pressurizer_level_pct"] -= np.linspace(0, leak_size*0.15, e-s+1)
    df.loc[s:e, "pressurizer_pressure_MPa"] -= np.linspace(0, leak_size*0.01, e-s+1)
    df.loc[s:e, "rcs_hot_leg_temp_C"] += np.linspace(0, leak_size*0.03, e-s+1) * 0.6
    df.loc[s:e, "rcs_cold_leg_temp_C"] -= np.linspace(0, leak_size*0.03, e-s+1) * 0.4

    events.append({
        "event_id": f"leak_{i+1}",
        "start_idx": int(s),
        "end_idx": int(e),
        "start_time": df.loc[s, "timestamp"],
        "end_time": df.loc[e, "timestamp"],
        "leak_type": leak_type,
        "leak_rate_lpm": float(leak_size)
    })

# false positives: rad blips & humidity spikes
false_events = []
for j in range(3):
    s = np.random.randint(0, n-200)
    dur = np.random.randint(5, 120)
    e = s + dur
    df.loc[s:e, "airborne_gamma_uSvph"] += np.random.uniform(0.5, 5.0, size=dur+1)
    false_events.append({"event_id":f"false_rad_{j+1}", "start_idx":s, "end_idx":e})

for j in range(2):
    s = np.random.randint(0, n-200)
    dur = np.random.randint(30, 300)
    e = s + dur
    df.loc[s:e, "containment_humidity_pct"] += np.random.uniform(10,30,size=dur+1)
    false_events.append({"event_id":f"false_hum_{j+1}", "start_idx":s, "end_idx":e})

# ground truth flags
df["ground_truth_event_id"] = pd.NA
df["ground_truth_leak_flag"] = False
df["ground_truth_leak_rate_lpm"] = 0.0
for ev in events:
    mask = (df.index >= ev["start_idx"]) & (df.index <= ev["end_idx"])
    df.loc[mask, "ground_truth_event_id"] = ev["event_id"]
    df.loc[mask, "ground_truth_leak_flag"] = True
    df.loc[mask, "ground_truth_leak_rate_lpm"] = ev["leak_rate_lpm"]
for ev in false_events:
    mask = (df.index >= ev["start_idx"]) & (df.index <= ev["end_idx"])
    df.loc[mask, "ground_truth_event_id"] = df.loc[mask, "ground_truth_event_id"].fillna(ev["event_id"])

# --- MISSINGNESS & DRIFT ---
# MCAR per-cell ~2% for key sensors
for col in ["sump_flow_lpm", "airborne_gamma_uSvph", "particulate_cpm", "makeup_flow_lpm", "letdown_flow_lpm"]:
    mask = np.random.rand(n) < 0.02
    df.loc[mask, col] = np.nan

# contiguous outages for some sensors
for sensor in ["pump_vibration_mm_s", "airborne_gamma_uSvph", "particulate_cpm"]:
    for _ in range(2):
        s = np.random.randint(0, n-1000)
        dur = np.random.randint(60, 6*60)  # 1 hour to 6 hours
        e = min(s + dur, n-1)
        df.loc[s:e, sensor] = np.nan

# drift: add bias to pump vibration at a random time
drift_sensor = "pump_vibration_mm_s"
drift_start = np.random.randint(int(n*0.2), int(n*0.8))
df.loc[drift_start:, drift_sensor] += 0.9

# recompute derived features after injection
df["rcs_mass_balance_lpm"] = df["makeup_flow_lpm"] - df["letdown_flow_lpm"]
df["temp_gradient_C"] = df["rcs_hot_leg_temp_C"] - df["rcs_cold_leg_temp_C"]
df["pressurizer_level_slope"] = df["pressurizer_level_pct"].diff().fillna(0).rolling(5, min_periods=1).mean()

# Save CSVs
raw_csv = os.path.join(RAW_DIR, "nuclear_sensors_raw.csv")
events_csv = os.path.join(RAW_DIR, "nuclear_events_table.csv")
df.to_csv(raw_csv, index=False)
pd.DataFrame(events + false_events).to_csv(events_csv, index=False)

print("Saved:", raw_csv)
print("Saved:", events_csv)
print(f"Dataset shape: {df.shape}")
print(f"True leak events: {len(events)}")
print(f"False events: {len(false_events)}")