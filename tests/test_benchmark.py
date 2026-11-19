"""The benchmark itself: datasets, reference answers and scoring (offline, no key)."""
import numpy as np
import pandas as pd
import pytest

from benchmark import run_benchmark
from benchmark.datasets import make_survey_dataset
from benchmark.questions import CHART_QUESTIONS, QUESTIONS, Question


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
    assert list(summary["suite"]) == ["answers", "charts"]
    assert (summary["accuracy_pct"] == 100.0).all() and (summary["mean_attempts"] == 1.0).all()
    assert list(summary["questions"]) == [len(QUESTIONS), len(CHART_QUESTIONS)]


def test_every_chart_question_has_a_reference_chart():
    charts = run_benchmark.reference_charts(CHART_QUESTIONS)
    for q in CHART_QUESTIONS:
        if q.charts:
            assert charts[q.id] is not None and len(run_benchmark.chart_numbers(charts[q.id])), q.id
            assert not [lvl for lvl, _ in charts[q.id].notes if lvl == "fallback"], q.id


def _turn(code, outputs=(), error=None):
    return {"code": code, "outputs": list(outputs), "error": error}


def test_chart_scoring_catches_broken_rules():
    data = make_survey_dataset()
    question = next(q for q in CHART_QUESTIONS if q.id == "ch12")
    reference = run_benchmark.reference_charts([question])["ch12"]
    good_outputs = run_benchmark.run_generated_code(question.answer_code, data, {})[1]
    assert run_benchmark.score_chart(question, _turn(question.answer_code, good_outputs), reference)["correct"]

    titled = question.answer_code + "\nplt.title('LPG use')"
    assert "wrote a title" in run_benchmark.score_chart(question, _turn(titled, good_outputs), reference)["problems"]
    no_highlight = question.answer_code.replace(", highlight=['Lakeside']", "")
    outputs = run_benchmark.run_generated_code(no_highlight, data, {})[1]
    assert "highlighted" in run_benchmark.score_chart(question, _turn(no_highlight, outputs), reference)["problems"]
    wrong = question.answer_code.replace("* 100", "* 10")
    outputs = run_benchmark.run_generated_code(wrong, data, {})[1]
    assert "numbers differ" in run_benchmark.score_chart(question, _turn(wrong, outputs), reference)["problems"]
    scatter = "ceew.scatter(df, 'household_size', 'monthly_kwh')\nresult = df"
    outputs = run_benchmark.run_generated_code(scatter, data, {})[1]
    assert "chart kind: scatter" in run_benchmark.score_chart(question, _turn(scatter, outputs), reference)["problems"]


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
