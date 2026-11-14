"""Synthetic sample dataset for practice. Not real measurements."""

import numpy as np
import pandas as pd


SAMPLE_FILE_NAME = "synthetic_air_quality_2024.csv"


SAMPLE_STATIONS = {   # fictional station: typical PM2.5 level
    "Riverside": 55, "Old Market": 95, "Industrial Estate": 110,
    "University Campus": 45, "Green Park": 35,
}


def make_sample_dataset(seed: int = 42) -> pd.DataFrame:
    """Synthetic daily air-quality readings for five fictional stations in 2024.

    Not real measurements. Deliberately includes problems to practise on:
    missing values, inconsistent station names, dates stored as text and a few
    duplicate rows.
    """
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2024-01-01", "2024-12-31", freq="D")
    day = dates.dayofyear.to_numpy()
    winter = 1 + 0.8 * np.cos(2 * np.pi * (day - 5) / 366)        # peaks in early January
    frames = []
    for station, level in SAMPLE_STATIONS.items():
        n = len(dates)
        wind = np.clip(rng.gamma(2.0, 1.2, n), 0.2, None)
        pm25 = level * winter * rng.lognormal(0, 0.25, n) / np.sqrt(wind / 2.4)
        frames.append(pd.DataFrame({
            "date": dates.strftime("%Y-%m-%d"),
            "station": station,
            "pm25_ugm3": np.round(pm25, 1),
            "pm10_ugm3": np.round(pm25 * rng.uniform(1.5, 2.2, n), 1),
            "temperature_c": np.round(25 - 9 * np.cos(2 * np.pi * (day - 15) / 366) + rng.normal(0, 1.5, n), 1),
            "relative_humidity_pct": np.round(np.clip(60 + 20 * np.sin(2 * np.pi * (day - 120) / 366)
                                                      + rng.normal(0, 8, n), 10, 100), 0),
            "wind_speed_ms": np.round(wind, 1),
        }))
    df = pd.concat(frames, ignore_index=True)

    # Problems to practise on.
    for col, share in (("pm25_ugm3", 0.04), ("pm10_ugm3", 0.06), ("relative_humidity_pct", 0.02)):
        df.loc[rng.random(len(df)) < share, col] = np.nan
    messy = rng.random(len(df)) < 0.02
    df.loc[messy, "station"] = df.loc[messy, "station"].str.lower() + " "
    duplicates = df.sample(5, random_state=seed)
    return pd.concat([df, duplicates], ignore_index=True)
