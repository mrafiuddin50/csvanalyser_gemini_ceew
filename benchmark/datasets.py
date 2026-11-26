"""Synthetic datasets for the benchmark. None of this is real data."""
import numpy as np
import pandas as pd

from csvanalyser.sample import make_sample_dataset


def make_survey_dataset(seed: int = 7) -> pd.DataFrame:
    """A made-up household energy survey: 600 households in four fictional districts.

    Includes inconsistent fuel spellings, missing electricity readings and
    dates written day/month/year as text.
    """
    rng = np.random.default_rng(seed)
    n = 600
    districts = np.array(["Northfield", "Lakeside", "Hillcrest", "Riverbend"])
    district = rng.choice(districts, n, p=[0.30, 0.25, 0.25, 0.20])
    income = rng.choice(["Low", "Middle", "High"], n, p=[0.45, 0.40, 0.15])

    fuels = ["LPG", "Firewood", "Kerosene", "Electric"]
    fuel_probs = {"Low": [0.45, 0.40, 0.10, 0.05], "Middle": [0.75, 0.15, 0.03, 0.07], "High": [0.80, 0.02, 0.0, 0.18]}
    fuel = np.array([rng.choice(fuels, p=fuel_probs[band]) for band in income], dtype=object)

    size = rng.integers(1, 9, n) + (district == "Hillcrest").astype(int)
    size = np.clip(size, 1, 10)
    base_kwh = pd.Series(income).map({"Low": 60, "Middle": 140, "High": 320}).to_numpy()
    kwh = np.round(base_kwh * (1 + 0.08 * size) * rng.lognormal(0, 0.3, n), 1)
    kwh[rng.random(n) < 0.05] = np.nan

    messy = rng.random(n) < 0.04
    fuel[messy & (fuel == "LPG")] = " lpg"
    dates = pd.Timestamp("2025-01-06") + pd.to_timedelta(rng.integers(0, 80, n), unit="D")

    return pd.DataFrame({
        "household_id": [f"HH-{i:04d}" for i in range(1, n + 1)],
        "district": district,
        "income_band": income,
        "cooking_fuel": fuel,
        "household_size": size,
        "monthly_kwh": kwh,
        "survey_date": dates.strftime("%d/%m/%Y"),
    })


def make_hourly_dataset(seed: int = 11) -> pd.DataFrame:
    """Made-up hourly PM2.5 at four fictional monitors over two years, in the shape of CPCB/OpenAQ exports.

    Times are stored as text, monitor names are long, and readings follow a daily cycle (high at night,
    low in the afternoon) and a winter peak. About 30% of hours are missing.
    """
    rng = np.random.default_rng(seed)
    times = pd.date_range("2022-01-01", "2023-12-31 23:00", freq="h")
    stations = {"Riverside Colony, Northfield - SPCB": 1.0, "Old Market Square, Northfield - CPCB": 1.25,
                "Industrial Area Phase II, Northfield - CPCB": 1.45, "University Campus, Northfield - SPCB": 0.8}
    frames = []
    for station, level in stations.items():
        keep = rng.random(len(times)) > 0.3
        t = times[keep]
        hour, day = t.hour.to_numpy(), t.dayofyear.to_numpy()
        daily = 1 + 0.35 * np.cos(2 * np.pi * (hour - 23) / 24)
        winter = 1 + 0.7 * np.cos(2 * np.pi * (day - 5) / 366)
        value = np.round(60 * level * daily * winter * rng.lognormal(0, 0.35, len(t)), 2)
        frames.append(pd.DataFrame({"parameter": "PM25", "observation_hour_start": t.strftime("%Y-%m-%d %H:%M:%S"),
                                    "location_name": station, "average_value_in_hour": value,
                                    "maximum_value_in_hour": np.round(value * rng.uniform(1.0, 1.3, len(t)), 2)}))
    return pd.concat(frames, ignore_index=True)


DATASETS = {
    "air_quality": make_sample_dataset,
    "household_survey": make_survey_dataset,
    "hourly_air": make_hourly_dataset,
}
