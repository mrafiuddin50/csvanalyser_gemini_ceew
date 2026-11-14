"""Data profile and the synthetic sample dataset."""


import pandas as pd

from csvanalyser import profiling, sample


def make_df():
    return pd.DataFrame({
        "site_name": ["Anand Vihar", "Punjabi Bagh", "Okhla Phase-2", "Anand Vihar"],
        "pm25": [123.45, 678.91, None, 123.45],
        "date": ["2023-01-05", "2023-02-17", "2023-03-09", "2023-01-05"],
    })


def fake_llm(code):
    return lambda prompt: code


# Data profile

def test_profile_summary_and_notes():
    summary, table = profiling.profile_dataset(make_df())
    assert summary == {"rows": 4, "columns": 3, "duplicate rows": 1, "missing cells": 1}
    by_col = table.set_index("column")
    assert by_col.loc["pm25", "missing"] == 1 and by_col.loc["pm25", "max"] == 678.91
    assert by_col.loc["date", "note"] == "Looks like dates stored as text"
    assert by_col.loc["site_name", "note"] == ""
    assert by_col.loc["site_name", "most common"].startswith("Anand Vihar (2)")


def test_profile_edge_cases():
    df = pd.DataFrame({"empty": [None, None], "flag": [True, False], "num_text": ["1", "2"]})
    _, table = profiling.profile_dataset(df)
    by_col = table.set_index("column")
    assert by_col.loc["empty", "note"] == "Entirely empty"
    assert pd.isna(by_col.loc["flag", "min"])
    summary, _ = profiling.profile_dataset(pd.DataFrame())
    assert summary["rows"] == 0


# Synthetic sample

def test_sample_is_deterministic_and_messy():
    a, b = sample.make_sample_dataset(), sample.make_sample_dataset()
    pd.testing.assert_frame_equal(a, b)
    assert list(a.columns) == ["date", "station", "pm25_ugm3", "pm10_ugm3", "temperature_c",
                               "relative_humidity_pct", "wind_speed_ms"]
    assert len(a) == 5 * 366 + 5
    summary, table = profiling.profile_dataset(a)
    assert summary["duplicate rows"] == 5
    assert table.set_index("column").loc["date", "note"] == "Looks like dates stored as text"
    assert a["pm25_ugm3"].isna().any()
    assert a["station"].nunique() > 5          # inconsistent spellings to clean
    jan = a[a["date"].str.startswith("2024-01")]["pm25_ugm3"].mean()
    jul = a[a["date"].str.startswith("2024-07")]["pm25_ugm3"].mean()
    assert jan > jul                           # winter peak
