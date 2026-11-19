"""Runs the benchmark questions through the app's own pipeline and scores the answers.

Usage (from the repository root):

    export GEMINI_API_KEY=...            # never commit or paste your key into files
    python -m benchmark.run_benchmark --models gemini-2.5-flash,gemini-2.5-pro

    python -m benchmark.run_benchmark --suite charts   # only the chart questions

    python -m benchmark.run_benchmark --self-check   # offline: no key, no network

Two suites: "answers" (questions with one known number or label) and "charts" (chart
requests scored on the CEEW data viz skill's rules: the right kind of chart, the right
numbers, no AI-written titles, no colours or second axes, highlights only when asked).

Only the synthetic datasets in benchmark/datasets.py are used, and the app's
privacy rules apply: Gemini sees the questions and column names/types, never values.
Each question costs one to three API requests per model.
"""
import argparse
import csv
import datetime
import math
import os
import pathlib
import re
import sys
import time

import numpy as np
import pandas as pd

from benchmark.datasets import DATASETS
from benchmark.questions import CHART_QUESTIONS, QUESTIONS, ChartQuestion, Question
from csvanalyser.analysis import analyse
from csvanalyser.gemini import GeminiError, generate_python_code
from csvanalyser.sandbox import run_generated_code

BENCH_CONTEXT = (
    "Give a single answer: assign `result` to one number or one text label, not a table. "
    "Do not round numbers."
)
# The app's default instructions, as a researcher asking for a chart would send them.
CHART_CONTEXT = (
    "Return the final answer strictly as a pandas dataframe "
    "(e.g. result = summary_df). Streamlit will render it automatically."
)

# Things the chart rules forbid, found in the generated code.
RULE_BREAKS = {
    "wrote a title": r"\.set_title\(|plt\.title\(|suptitle\(|\bst\.title\(",
    "used a second axis": r"twinx\(|twiny\(|secondary_y",
    "chose colours": r"\b(colors?|cmap|palette)\s*=",      # matplotlib's spelling; ceew.scatter's colour= is fine
    "drew without the ceew functions": r"plt\.subplots\(|plt\.figure\(|\.plot\(|st\.(bar|line|area|scatter)_chart\(",
}


def extract_answer(value):
    """Reduces a result to a single number or label where that is unambiguous."""
    if isinstance(value, pd.DataFrame) and value.size == 1:
        value = value.iat[0, 0]
    elif isinstance(value, pd.Series) and len(value) == 1:
        value = value.iloc[0]
    if isinstance(value, np.generic):
        value = value.item()
    return value


def is_correct(question: Question, answer, expected) -> bool:
    answer = extract_answer(answer)
    if answer is None:
        return False
    if question.kind == "label":
        return str(answer).strip().lower() == str(expected).strip().lower()
    try:
        answer = float(answer)
    except (TypeError, ValueError):
        return False
    if math.isnan(answer):
        return False
    return abs(answer - float(expected)) <= max(question.abs_tol, question.rel_tol * abs(float(expected)))


def reference_answers(questions) -> dict:
    data = {name: make() for name, make in DATASETS.items()}
    answers = {}
    for q in questions:
        result, _ = run_generated_code(q.answer_code, data[q.dataset], {})
        answers[q.id] = extract_answer(result)
    return answers


def run(models, questions, delay: float = 0.0, api_key: str = "", self_check: bool = False):
    data = {name: make() for name, make in DATASETS.items()}
    expected = reference_answers(questions)
    rows = []
    for model in models:
        for q in questions:
            if self_check:
                llm = (lambda code: (lambda prompt: code))(q.answer_code)
            else:
                llm = (lambda m: (lambda prompt: generate_python_code(prompt, m, api_key)))(model)
            start = time.perf_counter()
            turn = analyse(q.question, data[q.dataset], BENCH_CONTEXT, [], False, llm)
            seconds = time.perf_counter() - start
            answer = extract_answer(turn["result"]) if not turn["error"] else None
            correct = not turn["error"] and is_correct(q, answer, expected[q.id])
            rows.append({
                "model": model, "question": q.id, "correct": correct, "attempts": turn["attempts"],
                "error": turn["error"] or "", "answer": answer, "expected": expected[q.id],
                "seconds": round(seconds, 2),
            })
            mark = "ok " if correct else "XX "
            print(f"{mark} {model:<28} {q.id}  attempts={turn['attempts']}  answer={answer!r:.40}  "
                  f"expected={expected[q.id]!r:.40}" + (f"  error={turn['error'][:80]}" if turn["error"] else ""))
            if delay and not self_check:
                time.sleep(delay)
    return rows


def chart_specs(outputs) -> list:
    return [args[0] for name, args, _ in outputs if name == "chart"]


VALUE_PARAMS = ("value", "values", "ys", "parts", "start", "end")


def chart_numbers(spec) -> np.ndarray:
    """The plotted numbers of a chart (not its category or time labels), sorted."""
    p = spec.params
    if spec.kind == "histogram":
        cols = ["count"]
    elif spec.kind == "scatter":
        cols = [p["x"], p["y"]]
    elif spec.kind == "boxplot":
        cols = ["median"]
    else:
        cols = []
        for key in VALUE_PARAMS:
            value = p.get(key)
            cols += list(value) if isinstance(value, (list, tuple)) else ([value] if value is not None else [])
    cols = [c for c in cols if c in spec.data.columns]
    values = spec.data[cols].to_numpy(dtype=float).ravel() if cols else np.array([])
    return np.sort(values[np.isfinite(values)])


def score_chart(question: ChartQuestion, turn: dict, reference) -> dict:
    """Scores one chart answer. Returns the checks and a list of problems."""
    problems = []
    specs = chart_specs(turn["outputs"]) if not turn["error"] else []
    if turn["error"]:
        problems.append("error")
    if question.charts:
        matching = [s for s in specs if s.kind in question.charts]
        if not matching:
            problems.append("chart kind: " + (", ".join(s.kind for s in specs) or "no ceew chart"))
        elif reference is not None:
            ours, theirs = chart_numbers(matching[0]), chart_numbers(reference)
            if len(ours) != len(theirs) or not np.allclose(ours, theirs, rtol=question.rel_tol, atol=1e-6):
                problems.append("numbers differ")
    code = turn.get("code") or ""
    for rule, pattern in RULE_BREAKS.items():
        if rule.startswith("drew without") and not question.charts:
            continue
        if re.search(pattern, code):
            problems.append(rule)
    wanted = {h.lower() for h in question.highlight}
    for spec in specs:
        chosen = {str(h).lower() for h in spec.params.get("highlight") or []}
        if chosen != wanted:
            problems.append(f"highlighted {sorted(chosen)} instead of {sorted(wanted)}")
            break
    return {"correct": not problems, "problems": "; ".join(problems)}


def reference_charts(questions) -> dict:
    data = {name: make() for name, make in DATASETS.items()}
    charts = {}
    for q in questions:
        _, outputs = run_generated_code(q.answer_code, data[q.dataset], {})
        specs = [s for s in chart_specs(outputs) if s.kind in q.charts] if q.charts else []
        charts[q.id] = specs[0] if specs else None
    return charts


def run_charts(models, questions, delay: float = 0.0, api_key: str = "", self_check: bool = False):
    data = {name: make() for name, make in DATASETS.items()}
    reference = reference_charts(questions)
    rows = []
    for model in models:
        for q in questions:
            if self_check:
                llm = (lambda code: (lambda prompt: code))(q.answer_code)
            else:
                llm = (lambda m: (lambda prompt: generate_python_code(prompt, m, api_key)))(model)
            start = time.perf_counter()
            turn = analyse(q.question, data[q.dataset], CHART_CONTEXT, [], False, llm)
            seconds = time.perf_counter() - start
            score = score_chart(q, turn, reference[q.id])
            kinds = ",".join(s.kind for s in chart_specs(turn["outputs"])) or "-"
            rows.append({
                "model": model, "question": q.id, "correct": score["correct"], "attempts": turn["attempts"],
                "error": turn["error"] or "", "answer": kinds, "expected": ",".join(q.charts) or "any",
                "problems": score["problems"], "seconds": round(seconds, 2),
            })
            mark = "ok " if score["correct"] else "XX "
            print(f"{mark} {model:<28} {q.id}  attempts={turn['attempts']}  chart={kinds}"
                  + (f"  problems={score['problems'][:120]}" if score["problems"] else ""))
            if delay and not self_check:
                time.sleep(delay)
    return rows


def summarise(rows) -> pd.DataFrame:
    df = pd.DataFrame(rows)
    return df.groupby("model").agg(
        questions=("question", "count"),
        correct=("correct", "sum"),
        accuracy_pct=("correct", lambda s: round(100 * s.mean(), 1)),
        mean_attempts=("attempts", lambda s: round(s.mean(), 2)),
        errors=("error", lambda s: int((s != "").sum())),
        mean_seconds=("seconds", lambda s: round(s.mean(), 1)),
    )


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--models", default="gemini-2.5-flash", help="Comma-separated model names.")
    parser.add_argument("--questions", default="", help="Comma-separated question ids (default: all).")
    parser.add_argument("--delay", type=float, default=4.0, help="Seconds between questions (free-tier limits).")
    parser.add_argument("--out", default="benchmark/results", help="Folder for the results CSV.")
    parser.add_argument("--self-check", action="store_true", help="Offline check using the reference code.")
    parser.add_argument("--suite", choices=["all", "answers", "charts"], default="all",
                        help="Which questions to ask (default: all).")
    args = parser.parse_args(argv)

    wanted = set(args.questions.split(",")) if args.questions else None
    answer_questions = [q for q in QUESTIONS if wanted is None or q.id in wanted] \
        if args.suite in ("all", "answers") else []
    chart_questions = [q for q in CHART_QUESTIONS if wanted is None or q.id in wanted] \
        if args.suite in ("all", "charts") else []

    if args.self_check:
        models, api_key = ["self-check"], ""
    else:
        api_key = os.environ.get("GEMINI_API_KEY", "").strip()
        if not api_key:
            sys.exit("Set the GEMINI_API_KEY environment variable first (or use --self-check).")
        models = [m.strip() for m in args.models.split(",") if m.strip()]

    try:
        rows = run(models, answer_questions, args.delay, api_key, args.self_check) if answer_questions else []
        chart_rows = run_charts(models, chart_questions, args.delay, api_key, args.self_check) \
            if chart_questions else []
    except GeminiError as exc:
        sys.exit(f"Gemini error: {exc}")

    summaries = []
    for name, suite_rows in (("answers", rows), ("charts", chart_rows)):
        if suite_rows:
            summary = summarise(suite_rows)
            summary.insert(0, "suite", name)
            summaries.append(summary)
    summary = pd.concat(summaries)
    print("\n" + summary.to_string())
    if not args.self_check:
        out = pathlib.Path(args.out)
        out.mkdir(parents=True, exist_ok=True)
        stamp = f"{datetime.datetime.now():%Y%m%d_%H%M%S}"
        for name, suite_rows in (("results", rows), ("chart_results", chart_rows)):
            if not suite_rows:
                continue
            path = out / f"{name}_{stamp}.csv"
            with path.open("w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=list(suite_rows[0]))
                writer.writeheader()
                writer.writerows(suite_rows)
            print(f"Detailed results: {path}")
    return summary


if __name__ == "__main__":
    main()
