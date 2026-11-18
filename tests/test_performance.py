"""Speed and memory safeguards: fast date reading, light copies, released working data, capped results.

All data here is made up.
"""
import pathlib
import time

import numpy as np
import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest

from csvanalyser import audit
from csvanalyser.profiling import profile_dataset
from csvanalyser.sandbox import run_generated_code

APP = str(pathlib.Path(__file__).resolve().parents[1] / "stlite_app.py")


# -- dates -------------------------------------------------------------------

@pytest.mark.parametrize("values, dayfirst", [
    (pd.date_range("2023-01-01", periods=300, freq="7h").strftime("%d/%m/%Y %H:%M"), True),
    (pd.date_range("2023-01-01", periods=300, freq="D").strftime("%m/%d/%Y"), False),
    (pd.date_range("2023-01-01", periods=300, freq="D").strftime("%Y-%m-%d"), False),
    (list(pd.date_range("2023-01-01", periods=50).strftime("%Y-%m-%d"))
     + list(pd.date_range("2023-03-01", periods=50).strftime("%d %b %Y"))
     + ["not a date 9", "31/02/2024", "2024-13-01", "25/01/2024"], False),
])
def test_fast_date_reading_matches_reading_each_value(values, dayfirst):
    values = pd.Series(list(values) + [None])
    fast = audit._parse_dates(values, dayfirst)
    uniques = pd.Series(values.dropna().unique())
    slow = values.map(dict(zip(uniques, audit._parse_each(uniques, dayfirst))))
    assert fast.equals(slow)


def test_large_text_date_column_is_read_quickly():
    # Used to take about 18 s for 100,000 distinct dates (each parsed one by one); now about 2 s.
    dates = pd.date_range("2015-01-01", periods=100_000, freq="h").strftime("%d/%m/%Y %H:%M")
    df = pd.DataFrame({"date": dates, "pm25": np.arange(100_000) % 300})
    start = time.perf_counter()
    report = audit.run_audit(df, "big.csv")
    seconds = time.perf_counter() - start
    info = {c.name: c for c in report.column_info}
    assert info["date"].semantic == "datetime (stored as text)" and info["date"].day_month == "dayfirst"
    assert seconds < 15, f"audit of 100,000 rows took {seconds:.1f} s"


def test_codes_with_digits_are_not_read_as_dates():
    df = pd.DataFrame({"plant_code": [f"PL-{i % 400:04d}X" for i in range(5000)], "v": range(5000)})
    info = {c.name: c for c in audit.run_audit(df).column_info}
    assert info["plant_code"].semantic != "datetime (stored as text)"


# -- correlation -------------------------------------------------------------

def test_correlations_match_pair_by_pair_calculation():
    rng = np.random.default_rng(3)
    n = 400
    df = pd.DataFrame({"a": rng.normal(size=n), "b": rng.normal(size=n), "c": rng.normal(size=n)})
    df["b"] = df["a"] * 0.8 + df["b"] * 0.2
    df.loc[rng.random(n) < 0.2, "c"] = np.nan                       # a gappy column
    table = audit.run_audit(df).correlations.set_index(["column A", "column B"])
    for x, y in [("a", "b"), ("a", "c"), ("b", "c")]:
        pair = df[[x, y]].dropna()
        ranks = pair.rank()
        row = table.loc[(x, y)] if (x, y) in table.index else table.loc[(y, x)]
        assert row["rows used"] == len(pair)
        assert row["Pearson r"] == round(pair[x].corr(pair[y]), 3)
        assert row["Spearman rho"] == round(ranks[x].corr(ranks[y]), 3)


# -- memory ------------------------------------------------------------------

def test_audit_report_does_not_keep_full_length_copies():
    df = pd.DataFrame({"date": pd.date_range("2024-01-01", periods=60).strftime("%Y-%m-%d"),
                       "site": ["A", "B", "C"] * 20, "value": range(60), "amount": ["1,200"] * 60})
    report = audit.run_audit(df)
    assert all(i.numeric is None and i.dates is None and i.labels is None for i in report.column_info)
    assert report.temporal == ["date"]                     # still worked out before release


def test_column_details_reuse_the_audit_and_match():
    df = pd.DataFrame({"site": ["A", "B", None, "A"] * 15, "value": [1.0, None, 3.0, 1.0] * 15,
                       "when": ["2024-01-0" + str(i % 9 + 1) for i in range(60)]})
    df = pd.concat([df, df.head(5)], ignore_index=True)
    report = audit.run_audit(df)
    assert report.duplicate_rows == int(df.duplicated().sum())
    plain, reused = profile_dataset(df), profile_dataset(df, report)
    assert plain[0] == reused[0]
    assert plain[1].equals(reused[1])


@pytest.mark.parametrize("code", [
    "df['a'] = 0\nresult = 1",
    "df.loc[df['a'] > 2, 'b'] = 'x'\nresult = 1",
    "df.sort_values('a', inplace=True)\ndf.dropna(inplace=True)\nresult = 1",
    "df.columns = ['x', 'y', 'z']\ndf.index.name = 'i'\nresult = 1",
    "df.drop(columns=['c'], inplace=True)\ndf['new'] = 1\nresult = 1",
])
def test_generated_code_cannot_change_the_loaded_data(code):
    df = pd.DataFrame({"a": [1, 2, 3, 4], "b": list("pqrs"), "c": [1.0, np.nan, 3.0, np.nan]})
    before = df.copy(deep=True)
    run_generated_code(code, df, {})
    assert df.equals(before) and list(df.columns) == ["a", "b", "c"] and df.index.name is None


def _turn(i, result):
    return {"id": i, "question": f"question {i}", "sent_question": f"question {i}", "anonymise": False,
            "sent_prompts": ["prompt"], "code": f"result = df.head({i})", "attempts": 1,
            "outputs": [("caption", (f"output {i}",), {})], "result": result, "error": None}


def test_only_the_latest_answers_keep_their_results():
    data = pd.DataFrame({"x": range(30)})
    at = AppTest.from_file(APP, default_timeout=120)
    at.session_state["df"] = data
    at.session_state["file_name"] = "test.csv"
    at.session_state["history"] = [_turn(i, data.head(i)) for i in range(1, 12)]
    at.session_state["next_turn_id"] = 12
    at.session_state["main_tabs"] = "② Ask questions"
    at.session_state["card_tabs_11"] = "Code"              # where the "Run my version" button is
    at.run()
    at.button(key="run_11").click().run()                 # "Run my version" adds answer 12
    assert not at.exception, at.exception
    history = at.session_state["history"]
    assert len(history) == 12
    dropped = [t["id"] for t in history if t.get("result_dropped")]
    assert dropped == [1, 2]                              # 12 answers, latest 10 keep results
    assert all(t["result"] is None and t["outputs"] == [] and t["csv"] is None for t in history[:2])
    assert all(t["result"] is not None for t in history[2:])
    # Each kept result's CSV is written once and reused on later redraws
    csv_before = history[5]["csv"]
    assert csv_before.startswith(b",x")
    at.run()
    assert at.session_state["history"][5]["csv"] is csv_before
    assert all(t["code"] for t in history)                # code and questions are kept for export
    assert sum("only the latest 10 answers keep their results" in i.value for i in at.info) == 2
