"""Data profile, computed locally and never sent."""

import pandas as pd


def _looks_like_dates(series: pd.Series) -> bool:
    import warnings
    sample = series.dropna().astype(str).str.strip()
    sample = sample[sample.str.contains(r"\d", regex=True)].head(200)
    if len(sample) < 3 or len(sample) < 0.5 * min(len(series.dropna()), 200):
        return False
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            parsed = pd.to_datetime(sample, errors="coerce", format="mixed")
        except (TypeError, ValueError):
            parsed = pd.to_datetime(sample, errors="coerce")
    return parsed.notna().mean() >= 0.8


def profile_dataset(df: pd.DataFrame):
    """Returns (summary dict, per-column table) describing data quality."""
    rows = []
    for col in df.columns:
        series = df[col]
        missing = int(series.isna().sum())
        row = {
            "column": str(col),
            "type": str(series.dtype),
            "missing": missing,
            "missing %": round(100 * missing / len(series), 1) if len(series) else 0.0,
            "distinct": int(series.nunique(dropna=True)),
            "min": None, "median": None, "mean": None, "max": None,
            "most common": "",
            "note": "",
        }
        if pd.api.types.is_numeric_dtype(series) and not pd.api.types.is_bool_dtype(series):
            values = series.dropna()
            if len(values):
                row.update({
                    "min": float(values.min()), "median": float(values.median()),
                    "mean": round(float(values.mean()), 3), "max": float(values.max()),
                })
        else:
            counts = series.dropna().astype(str).value_counts().head(3)
            row["most common"] = ", ".join(f"{value} ({count})" for value, count in counts.items())
            if not pd.api.types.is_datetime64_any_dtype(series) and _looks_like_dates(series):
                row["note"] = "Looks like dates stored as text"
        if missing == len(series) and len(series):
            row["note"] = "Entirely empty"
        rows.append(row)

    summary = {
        "rows": len(df),
        "columns": df.shape[1],
        "duplicate rows": int(df.duplicated().sum()) if len(df.columns) else 0,
        "missing cells": int(df.isna().sum().sum()),
    }
    return summary, pd.DataFrame(rows)
