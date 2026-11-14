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


DATASETS = {
    "air_quality": make_sample_dataset,
    "household_survey": make_survey_dataset,
}
