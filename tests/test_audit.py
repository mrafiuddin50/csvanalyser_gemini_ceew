"""Data quality audit: each check, the verdicts, the 'what you can analyse' table and the report."""
import datetime as dt

import numpy as np
import pandas as pd
import pytest

from benchmark.datasets import make_survey_dataset
from csvanalyser import audit, sample

TODAY = dt.date(2026, 11, 17)


def run(df, name="test.csv"):
    return audit.run_audit(df, name, today=TODAY)


def issues(report, severity=None):
    return [f.issue for f in report.findings if severity is None or f.severity == severity]


def find(report, text, severity=None):
    matches = [f for f in report.findings if text in f.issue and (severity is None or f.severity == severity)]
    assert matches, f"no finding containing {text!r} in {issues(report)}"
    return matches[0]


def capability(report, start):
    return next(c for c in report.capabilities if c["Analysis"].startswith(start))


def clean_df(n=60):
    rng = np.random.default_rng(1)
    return pd.DataFrame({
        "record_id": np.arange(1, n + 1),
        "site": rng.choice(["North", "South", "East"], n),
        "temperature_c": np.round(rng.normal(25, 2, n), 1),
        "humidity": np.round(rng.normal(60, 5, n), 1),
    })


# Verdicts on known datasets

def test_clean_data_is_good_and_ready():
    report = run(clean_df())
    assert report.assessment == "Good" and report.readiness == audit.READY
    assert not report.by_severity(audit.CRITICAL) and not report.by_severity(audit.WARNING)
    assert report.identifiers == ["record_id"]
    assert capability(report, "Joining")["Status"] == audit.CAN


def test_synthetic_sample_needs_attention():
    report = run(sample.make_sample_dataset(), "synthetic.csv")
    assert report.assessment == "Needs Attention" and report.readiness == audit.CAUTION
    dup = find(report, "exact copies", audit.WARNING)
    assert dup.affected == 5 and dup.total == 1835
    labels = find(report, "'station' has 5 label(s) written in more than one way", audit.WARNING)
    assert labels.affected > 0
    find(report, "Dates in 'date' are stored as text", audit.OBSERVATION)
    find(report, "Very strong correlations", audit.OBSERVATION)
    assert capability(report, "Trends over time")["Status"] == audit.CAREFUL
    assert capability(report, "Comparisons between groups")["Status"] == audit.CAREFUL
    assert capability(report, "Maps")["Status"] == audit.CANNOT
    assert "'station' + 'date'" in capability(report, "Joining")["Why"]


def test_survey_spots_day_first_dates_and_is_not_treated_as_time_series():
    report = run(make_survey_dataset(), "survey.csv")
    assert report.readiness == audit.CAUTION
    find(report, "written day/month/year", audit.WARNING)
    assert not any("share the same" in i for i in issues(report))
    assert any("does not look like a time series" in s for s in report.checks_skipped)
    assert capability(report, "Joining")["Status"] == audit.CAN          # household_id
    assert capability(report, "Maps")["Status"] == audit.CAREFUL         # district names only


def test_empty_file_is_not_ready():
    report = run(pd.DataFrame({"a": [], "b": []}))
    assert report.assessment == "Poor" and report.readiness == audit.NOT_READY
    find(report, "no data rows", audit.CRITICAL)


# Individual checks

def test_heavy_duplicates_are_critical():
    df = pd.concat([clean_df(40)] * 2, ignore_index=True)
    report = run(df)
    assert find(report, "exact copies", audit.CRITICAL).affected == 40
    assert report.readiness == audit.NOT_READY


def test_missing_values_and_placeholders():
    df = clean_df(50)
    df.loc[:19, "humidity"] = np.nan                     # 40% missing
    df.loc[:2, "temperature_c"] = np.nan                 # 6% missing
    df["notes"] = ["-"] * 5 + ["unknown"] * 3 + ["ok"] * 42
    report = run(df)
    assert find(report, "40.0% of values are missing in 'humidity'", audit.WARNING).affected == 20
    find(report, "Some missing values in 'temperature_c'", audit.OBSERVATION)
    assert find(report, "Placeholder text", audit.WARNING).affected == 8


def test_mostly_empty_table_is_critical():
    df = pd.DataFrame({"a": [1, None, None, None], "b": [None, None, None, 2]})
    find(run(df), "Most of the table is empty", audit.CRITICAL)


def test_numbers_stored_as_text_and_mixed_types():
    n = 40
    df = pd.DataFrame({
        "income": [f"{1000 + i:,}" for i in range(n)],           # '1,000' style
        "reading": [str(i) if i % 3 else "low" for i in range(n)],
    })
    report = run(df)
    find(report, "Numbers stored as text in 'income'", audit.WARNING)
    mixed = find(report, "'reading' mixes numbers and text", audit.WARNING)
    assert mixed.affected == 14
    info = {c.name: c for c in report.column_info}
    assert info["income"].semantic == "quantitative (stored as text)"
    assert report.stats.set_index("column").loc["income", "max"] == 1039


@pytest.mark.parametrize("dates, expected", [
    (["25/01/2025", "05/02/2025", "13/03/2025"] * 5, "written day/month/year"),
    (["05/02/2025", "06/03/2025", "07/04/2025"] * 5, "Day/month order cannot be told apart"),
    (["25/01/2025", "01/25/2025", "05/02/2025"] * 5, "mixes day/month and month/day"),
])
def test_day_month_order(dates, expected):
    report = run(pd.DataFrame({"visit_date": dates, "value": range(len(dates))}))
    find(report, expected, audit.WARNING)


def test_unreadable_and_implausible_dates():
    df = pd.DataFrame({"date": ["2024-01-01", "2024-01-02", "not a date 9", "2030-01-01", "1850-01-01"] * 4,
                       "value": range(20)})
    report = run(df)
    assert find(report, "could not be read as dates", audit.WARNING).affected == 4
    find(report, "in the future", audit.OBSERVATION)
    find(report, "before 1900", audit.OBSERVATION)


def test_value_rules():
    n = 30
    df = pd.DataFrame({
        "latitude": [28.6] * (n - 2) + [95.0, -91.0],
        "longitude": [77.2] * (n - 1) + [200.0],
        "share_pct": [50.0] * (n - 1) + [120.0],
        "pm25": [40.0] * (n - 3) + [-999.0] * 3,
        "temperature": [-5.0] * n,                                     # negatives fine: no rule
        "start_date": ["2024-01-10"] * (n - 2) + ["2024-03-01"] * 2,
        "end_date": ["2024-02-10"] * n,
    })
    report = run(df)
    assert find(report, "latitudes in 'latitude' are outside", audit.WARNING).affected == 2
    assert find(report, "longitudes in 'longitude' are outside", audit.WARNING).affected == 1
    assert find(report, "outside 0-100", audit.WARNING).kind == "domain"
    assert find(report, "negative values in 'pm25'", audit.WARNING).affected == 3
    assert not any("'temperature'" in i and "negative" in i for i in issues(report))
    assert find(report, "rows have 'start_date' after 'end_date'", audit.WARNING).affected == 2
    assert capability(report, "Maps")["Status"] == audit.CAREFUL


def test_conflicting_records_in_time_series():
    dates = pd.date_range("2024-01-01", periods=20).strftime("%Y-%m-%d").tolist()
    df = pd.DataFrame({"station": ["A"] * 20 + ["B"] * 20, "date": dates * 2, "pm25": range(40)})
    clash = pd.DataFrame({"station": ["A", "A"], "date": [dates[3], dates[4]], "pm25": [999, 998]})
    report = run(pd.concat([df, clash], ignore_index=True))
    assert find(report, "share the same 'station' and 'date'", audit.WARNING).affected == 4


def test_reporting_gaps():
    dates = pd.date_range("2024-01-01", periods=40)
    dates = dates.delete(list(range(10, 15)) + list(range(30, 33)))
    df = pd.DataFrame({"date": dates.strftime("%Y-%m-%d"), "value": range(len(dates))})
    report = run(df)
    gap = find(report, "gaps in reporting", audit.OBSERVATION)
    assert gap.affected == 2 and "1 day" in gap.issue


def test_unnamed_and_repeated_columns():
    df = pd.DataFrame([[1, 2, 3, 4]] * 5, columns=["a", "Unnamed: 1", "b", "b.1"])
    report = run(df)
    find(report, "have no header name", audit.WARNING)
    find(report, "appear more than once", audit.WARNING)


def test_rare_and_dominant_categories():
    df = pd.DataFrame({"fuel": ["LPG"] * 96 + ["Kerosene"] * 2 + ["Biogas"] * 2, "x": range(100)})
    report = run(df)
    find(report, "Rare categories in 'fuel'", audit.OBSERVATION)
    find(report, "dominated by one category", audit.OBSERVATION)


def test_outliers_skew_and_constant_columns():
    values = [10.0 + 0.01 * i for i in range(80)] + [500.5, 600.5]
    df = pd.DataFrame({"load_kw": values, "flag_const": [7.5] * 82})
    report = run(df)
    out = find(report, "Possible outliers (IQR rule) in 'load_kw'", audit.OBSERVATION)
    assert out.affected == 2 and out.columns == ("load_kw",)
    find(report, "Strongly skewed", audit.OBSERVATION)
    find(report, "single value", audit.OBSERVATION)


def test_identifier_columns_are_not_treated_as_measurements():
    report = run(clean_df())
    assert "record_id" not in report.stats["column"].tolist()
    assert not any("record_id" in i for i in issues(report))


def test_counts_are_quantitative_not_codes():
    df = pd.DataFrame({"household_size": [1, 2, 3, 4, 5, 6] * 10, "grade_code": [1, 2, 3, 4, 5, 6] * 10})
    info = {c.name: c for c in run(df).column_info}
    assert info["household_size"].semantic == "quantitative"
    assert info["grade_code"].semantic == "numeric (possibly a category code)"


# Personal data: detected, never shown

PERSONAL = {
    "respondent_name": ["Asha Verma", "Ravi Kumar", "Meena Iyer", "John Paul"] * 5,
    "contact": ["asha.verma@example.org", "ravi99@example.com", "meena@example.net", "jp@example.org"] * 5,
    "mobile": ["9876543210", "+91 91234 56789", "8123456789", "7012345678"] * 5,
    "id_doc": ["1234 5678 9012", "234567890123", "ABCDE1234F", "BCDEF2345G"] * 5,
    "phone_num": [9876543210, 9123456780, 8012345678, 7098765432] * 5,
}
SECRETS = ["Asha", "Verma", "Ravi", "Meena", "@example", "9876543210", "91234", "1234 5678", "ABCDE1234F",
           "9123456780"]


def test_personal_information_is_flagged():
    report = run(pd.DataFrame(PERSONAL))
    for col in PERSONAL:
        find(report, f"'{col}' may contain personal information", audit.WARNING)
    assert "20 email addresses" in find(report, "'contact' may").issue
    assert "20 phone numbers" in find(report, "'mobile' may").issue
    assert "Aadhaar-like" in find(report, "'id_doc' may").issue and "PAN-like" in find(report, "'id_doc' may").issue
    assert "phone numbers" in find(report, "'phone_num' may").issue
    assert capability(report, "Sharing")["Status"] == audit.CAREFUL


def test_personal_values_never_appear_in_report_or_html():
    report = run(pd.DataFrame(PERSONAL))
    text = " ".join(f.issue + f.impact + f.action for f in report.findings)
    text += audit.column_summary(report).to_string() + str(report.capabilities)
    text += audit.build_html_report(report)
    for secret in SECRETS:
        assert secret not in text, secret


def test_station_names_are_not_mistaken_for_personal_names():
    df = pd.DataFrame({"station_name": ["Riverside", "Old Market"] * 30, "value": range(60)})
    assert not any("personal" in i for i in issues(run(df)))


# Report

def test_html_report_has_all_sections_and_escapes_text():
    df = sample.make_sample_dataset()
    df["<b>odd</b>"] = 1
    html = audit.build_html_report(run(df, "my <file>.csv"))
    for heading in ["1. Dataset overview", "2. Overall assessment", "3. Critical issues", "4. Warnings",
                    "5. Observations", "6. Column-level quality summary", "7. Recommended actions",
                    "8. Analysis readiness", "What you can analyse", "Methodology and limitations"]:
        assert heading in html
    assert "No critical issues identified by the checks performed." in html
    assert "<b>odd</b>" not in html and "&lt;b&gt;odd&lt;/b&gt;" in html
    assert "my &lt;file&gt;.csv" in html
    assert "UTC" in html


def test_every_column_in_summary_and_actions_capped():
    report = run(sample.make_sample_dataset())
    summary = audit.column_summary(report)
    assert summary["Column"].tolist() == list(sample.make_sample_dataset().columns)
    assert 1 <= len(report.actions) <= 7


def test_numeric_column_names_from_excel_do_not_break_audit():
    df = pd.DataFrame({2023: [1.0, 2.0, 3.0], "site": ["a", "b", "c"]})
    report = run(df)
    assert {c.name for c in report.column_info} == {"2023", "site"}


def test_audit_works_without_scipy(monkeypatch):
    # The browser (Pyodide via stlite) has no scipy; pandas' Spearman needs it.
    import sys
    monkeypatch.setitem(sys.modules, "scipy", None)
    monkeypatch.setitem(sys.modules, "scipy.stats", None)
    report = run(sample.make_sample_dataset())
    assert not report.correlations.empty
    row = report.correlations.iloc[0]
    assert {row["column A"], row["column B"]} == {"pm25_ugm3", "pm10_ugm3"} and row["Spearman rho"] > 0.9


def test_spearman_matches_pandas_when_scipy_is_available():
    pytest.importorskip("scipy")
    report = run(sample.make_sample_dataset())
    df = sample.make_sample_dataset()
    for _, row in report.correlations.iterrows():
        pair = df[[row["column A"], row["column B"]]].dropna()
        expected = pair.iloc[:, 0].corr(pair.iloc[:, 1], method="spearman")
        assert row["Spearman rho"] == pytest.approx(expected, abs=1e-3)


def test_audit_does_not_change_the_data():
    df = sample.make_sample_dataset()
    before = df.copy()
    run(df)
    pd.testing.assert_frame_equal(df, before)
