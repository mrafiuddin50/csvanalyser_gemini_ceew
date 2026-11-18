"""Example questions."""
import pandas as pd

from benchmark.datasets import make_survey_dataset
from csvanalyser import audit, sample
from csvanalyser.suggestions import suggest_questions


def columns(df):
    return audit.run_audit(df, "x.csv").column_info


def test_examples_for_sample_dataset():
    questions = suggest_questions(columns(sample.make_sample_dataset()))
    assert questions == [
        "Average pm25_ugm3 for each station",
        "Monthly average pm25_ugm3 over time, as a line chart",
        "Number of rows for each station, as a bar chart",
        "The 10 rows with the highest pm25_ugm3",
        "Is pm25_ugm3 related to pm10_ugm3? Show a scatter plot",
    ]
    assert all(len(q) <= 60 for q in questions)    # short enough to fit on a button


def test_examples_use_column_names_only_never_values():
    df = sample.make_sample_dataset()
    text = " ".join(suggest_questions(columns(df), limit=10))
    for value in df["station"].dropna().unique():
        assert str(value).strip() not in text
    assert str(df["pm25_ugm3"].dropna().iloc[0]) not in text


def test_examples_skip_possible_personal_data_columns():
    df = pd.DataFrame({"respondent_name": ["Asha Verma", "Ravi Kumar"] * 30, "score": range(60)})
    assert all("respondent_name" not in q for q in suggest_questions(columns(df)))


def test_examples_for_other_shapes():
    survey = suggest_questions(columns(make_survey_dataset()))
    assert any("for each district" in q or "for each income_band" in q for q in survey)
    only_text = suggest_questions(columns(pd.DataFrame({"note": [f"free text number {i}" for i in range(40)]})))
    assert only_text == ["A short summary of each column"]
    assert len(suggest_questions(columns(sample.make_sample_dataset()), limit=2)) == 2


