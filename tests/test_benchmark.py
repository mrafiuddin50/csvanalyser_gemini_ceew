"""The benchmark itself: datasets, reference answers and scoring (offline, no key)."""
import numpy as np
import pandas as pd
import pytest

from benchmark import run_benchmark
from benchmark.datasets import make_survey_dataset
from benchmark.questions import QUESTIONS, Question


def test_question_ids_unique_and_kinds_valid():
    ids = [q.id for q in QUESTIONS]
    assert len(ids) == len(set(ids))
    assert {q.kind for q in QUESTIONS} <= {"number", "label"}


def test_survey_dataset_is_deterministic_and_messy():
    a, b = make_survey_dataset(), make_survey_dataset()
    pd.testing.assert_frame_equal(a, b)
    assert len(a) == 600 and a["monthly_kwh"].isna().any()
    assert (a["cooking_fuel"] == " lpg").any()


def test_reference_answers_are_single_values():
    answers = run_benchmark.reference_answers(QUESTIONS)
    for q in QUESTIONS:
        value = answers[q.id]
        assert isinstance(value, (int, float, str)), (q.id, value)
        if q.kind == "number":
            assert not np.isnan(float(value)), q.id


def test_self_check_scores_full_marks():
    summary = run_benchmark.main(["--self-check"])
    assert summary.loc["self-check", "accuracy_pct"] == 100.0
    assert summary.loc["self-check", "mean_attempts"] == 1.0


NUM = Question("x", "air_quality", "q", "number", "result = 1", rel_tol=0.01)
LABEL = Question("y", "air_quality", "q", "label", "result = 'a'")


@pytest.mark.parametrize("question, answer, expected, ok", [
    (NUM, 100.5, 100.0, True),
    (NUM, 102.0, 100.0, False),
    (NUM, pd.DataFrame({"v": [100.2]}), 100.0, True),
    (NUM, pd.Series([100.2]), 100.0, True),
    (NUM, np.float64(100.0), 100.0, True),
    (NUM, pd.Series([1.0, 2.0]), 100.0, False),
    (NUM, "not a number", 100.0, False),
    (NUM, float("nan"), 100.0, False),
    (NUM, None, 100.0, False),
    (LABEL, " Industrial Estate ", "industrial estate", True),
    (LABEL, "Green Park", "industrial estate", False),
])
def test_scoring(question, answer, expected, ok):
    assert run_benchmark.is_correct(question, answer, expected) is ok
