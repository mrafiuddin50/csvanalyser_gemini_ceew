"""Runs the benchmark questions through the app's own pipeline and scores the answers.

Usage (from the repository root):

    export GEMINI_API_KEY=...            # never commit or paste your key into files
    python -m benchmark.run_benchmark --models gemini-2.5-flash,gemini-2.5-pro

    python -m benchmark.run_benchmark --self-check   # offline: no key, no network

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
import sys
import time

import numpy as np
import pandas as pd

from benchmark.datasets import DATASETS
from benchmark.questions import QUESTIONS, Question
from csvanalyser.analysis import analyse
from csvanalyser.gemini import GeminiError, generate_python_code
from csvanalyser.sandbox import run_generated_code

BENCH_CONTEXT = (
    "Give a single answer: assign `result` to one number or one text label, not a table. "
    "Do not round numbers."
)


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
    args = parser.parse_args(argv)

    questions = QUESTIONS
    if args.questions:
        wanted = set(args.questions.split(","))
        questions = [q for q in QUESTIONS if q.id in wanted]

    if args.self_check:
        models, api_key = ["self-check"], ""
    else:
        api_key = os.environ.get("GEMINI_API_KEY", "").strip()
        if not api_key:
            sys.exit("Set the GEMINI_API_KEY environment variable first (or use --self-check).")
        models = [m.strip() for m in args.models.split(",") if m.strip()]

    try:
        rows = run(models, questions, args.delay, api_key, args.self_check)
    except GeminiError as exc:
        sys.exit(f"Gemini error: {exc}")

    summary = summarise(rows)
    print("\n" + summary.to_string())
    if not args.self_check:
        out = pathlib.Path(args.out)
        out.mkdir(parents=True, exist_ok=True)
        path = out / f"results_{datetime.datetime.now():%Y%m%d_%H%M%S}.csv"
        with path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        print(f"\nDetailed results: {path}")
    return summary


if __name__ == "__main__":
    main()
