"""
data_ingestion.py — Load PJM energy data and weather data, merge, and save.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys, os

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config


def load_pjm_data(filename: str = config.PRIMARY_DATASET) -> pd.DataFrame:
    """Load PJM hourly energy consumption data."""
    filepath = config.DATA_RAW / filename
    if not filepath.exists():
        raise FileNotFoundError(f"PJM data not found at {filepath}")

    df = pd.read_csv(filepath)
    df[config.DATETIME_COLUMN] = pd.to_datetime(df[config.DATETIME_COLUMN])
    df = df.sort_values(config.DATETIME_COLUMN).reset_index(drop=True)
    df = df.set_index(config.DATETIME_COLUMN)

    print(f"[INFO] Loaded PJM data: {df.shape[0]:,} rows  "
          f"({df.index.min()} → {df.index.max()})")
    return df


def generate_synthetic_weather(index: pd.DatetimeIndex,
                               seed: int = config.RANDOM_SEED) -> pd.DataFrame:
    """
    Generate realistic synthetic weather features aligned to the energy index.

    The synthetic data follows these realistic patterns:
    - Temperature: seasonal (cold winters, hot summers) + diurnal cycle + noise
    - Humidity: inverse correlation with temperature + noise
    - Wind speed: log-normal distribution with seasonal variation
    - Dew point: derived from temperature and humidity
    """
    rng = np.random.RandomState(seed)
    n = len(index)

    day_of_year = index.dayofyear.values.astype(float)
    hour = index.hour.values.astype(float)

    # Temperature (°F): seasonal + diurnal + noise
    seasonal = 55 + 25 * np.sin(2 * np.pi * (day_of_year - 80) / 365)
    diurnal = 5 * np.sin(2 * np.pi * (hour - 6) / 24)
    temperature = seasonal + diurnal + rng.normal(0, 4, n)

    # Humidity (%): inversely related to temperature
    humidity = 80 - 0.4 * (temperature - 40) + rng.normal(0, 8, n)
    humidity = np.clip(humidity, 15, 100)

    # Wind speed (mph): log-normal with seasonal variation
    wind_base = 6 + 2 * np.sin(2 * np.pi * (day_of_year - 30) / 365)
    wind_speed = wind_base + rng.exponential(2.5, n)
    wind_speed = np.clip(wind_speed, 0, 45)

    # Dew point (°F): derived from temperature & humidity (Magnus formula approx.)
    dew_point = temperature - ((100 - humidity) / 5.0)

    weather_df = pd.DataFrame({
        "temperature": np.round(temperature, 1),
        "humidity": np.round(humidity, 1),
        "wind_speed": np.round(wind_speed, 1),
        "dew_point": np.round(dew_point, 1),
    }, index=index)

    print(f"[INFO] Generated synthetic weather data: {n:,} rows")
    return weather_df


def load_weather_data(index: pd.DatetimeIndex) -> pd.DataFrame:
    """
    Attempt to load real weather data from data/raw/weather/.
    Falls back to synthetic generation if not available.
    """
    weather_dir = config.DATA_RAW / "weather"
    if weather_dir.exists():
        csv_files = list(weather_dir.glob("*.csv"))
        if csv_files:
            frames = [pd.read_csv(f, parse_dates=["datetime"],
                                  index_col="datetime") for f in csv_files]
            weather = pd.concat(frames).sort_index()
            weather = weather.reindex(index, method="nearest")
            print(f"[INFO] Loaded real weather data: {weather.shape[0]:,} rows")
            return weather

    return generate_synthetic_weather(index)


def ingest_data() -> pd.DataFrame:
    """Main ingestion pipeline: load PJM + weather, merge, and save."""
    # Load energy data
    energy = load_pjm_data()

    # Load or generate weather data
    weather = load_weather_data(energy.index)

    # Merge
    merged = energy.join(weather, how="left")

    # Ensure no timezone issues
    if merged.index.tz is not None:
        merged.index = merged.index.tz_localize(None)

    # Save processed
    output_path = config.DATA_PROCESSED / "merged_data.csv"
    merged.to_csv(output_path)
    print(f"[INFO] Merged data saved to {output_path}  "
          f"({merged.shape[0]:,} rows, {merged.shape[1]} columns)")

    return merged


if __name__ == "__main__":
    df = ingest_data()
    print(df.head())
    print(df.describe())
