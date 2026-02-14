"""
config.py — Central configuration for the Energy Demand Prediction project.
All paths, constants, hyperparameters, and feature definitions live here.
"""

import os
from pathlib import Path

# ──────────────────────────────────────────────
# PATHS
# ──────────────────────────────────────────────
PROJECT_ROOT = Path(os.path.dirname(os.path.abspath(__file__)))
DATA_RAW = PROJECT_ROOT / "data" / "raw"
DATA_PROCESSED = PROJECT_ROOT / "data" / "processed"
MODELS_DIR = PROJECT_ROOT / "models_saved"
RESULTS_DIR = PROJECT_ROOT / "results"
FIGURES_DIR = RESULTS_DIR / "figures"
METRICS_DIR = RESULTS_DIR / "metrics"

# Create directories if they don't exist
for d in [DATA_PROCESSED, MODELS_DIR, FIGURES_DIR, METRICS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# ──────────────────────────────────────────────
# DATASET
# ──────────────────────────────────────────────
PRIMARY_DATASET = "PJME_hourly.csv"          # PJM East region
TARGET_COLUMN = "PJME_MW"                     # Megawatt demand
DATETIME_COLUMN = "Datetime"

# ──────────────────────────────────────────────
# FORECAST HORIZONS
# ──────────────────────────────────────────────
HORIZON_24H = 24       # hours
HORIZON_7D = 24 * 7    # 168 hours
HORIZON_30D = 24 * 30  # 720 hours

# ──────────────────────────────────────────────
# FEATURE ENGINEERING
# ──────────────────────────────────────────────
LAG_FEATURES = [1, 2, 3, 24, 48, 168]                    # t-1 … t-168
ROLLING_WINDOWS = [24, 168]                                # 24h and 7-day
CALENDAR_FEATURES = ["hour", "dayofweek", "month",
                     "day", "quarter", "is_weekend"]
WEATHER_FEATURES = ["temperature", "humidity",
                    "wind_speed", "dew_point"]

# ──────────────────────────────────────────────
# TRAIN / VAL / TEST SPLIT  (chronological)
# ──────────────────────────────────────────────
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15

# ──────────────────────────────────────────────
# SLIDING WINDOW (for DL models)
# ──────────────────────────────────────────────
WINDOW_SIZE = 168          # 7 days of history → predict next step(s)
BATCH_SIZE = 64
EPOCHS = 50
LEARNING_RATE = 1e-3
EARLY_STOP_PATIENCE = 7

# ──────────────────────────────────────────────
# MODEL HYPERPARAMETERS
# ──────────────────────────────────────────────
RF_PARAMS = {
    "n_estimators": 300,
    "max_depth": 20,
    "min_samples_split": 5,
    "n_jobs": -1,
    "random_state": 42,
}

XGB_PARAMS = {
    "n_estimators": 500,
    "max_depth": 8,
    "learning_rate": 0.05,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "random_state": 42,
}

LSTM_UNITS = 64
GRU_UNITS = 64
CNN_FILTERS = 64
CNN_KERNEL_SIZE = 3

# ──────────────────────────────────────────────
# SMART GRID OPTIMIZATION
# ──────────────────────────────────────────────
PEAK_THRESHOLD_PERCENTILE = 90   # MW above this → peak alert
BATTERY_CAPACITY_MWH = 500       # Total battery storage
BATTERY_MAX_RATE_MW = 100        # Max charge/discharge rate
RENEWABLE_CAPACITY_MW = 800      # Combined solar + wind nameplate

RANDOM_SEED = 42
