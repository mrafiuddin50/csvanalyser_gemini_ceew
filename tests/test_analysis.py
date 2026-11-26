"""Prompts, self-correcting retries, follow-ups and anonymised column names."""

import re

import pandas as pd
import pytest

from csvanalyser import analysis, prompts

SECRET_TEXT = ["Anand Vihar", "Punjabi Bagh", "Okhla Phase-2", "R-0042", "R-0077", "R-0913", "2023-01-05", "2023-02-17", "notadate-xyz"]
SECRET_NUMS = ["123.45", "678.91", "246.8"]


def make_df():
    return pd.DataFrame({
        "site_name": ["Anand Vihar", "Punjabi Bagh", "Okhla Phase-2"],
        "respondent": ["R-0042", "R-0077", "R-0913"],
        "date": ["2023-01-05", "2023-02-17", "notadate-xyz"],
        "pm25": [123.45, 678.91, 246.8],
    })


def fake_llm(codes):
    prompts = []

    def llm(prompt):
        prompts.append(prompt)
        return codes[min(len(prompts) - 1, len(codes) - 1)]
    return llm, prompts


def assert_no_leak(texts):
    for text in texts:
        for secret in SECRET_TEXT + SECRET_NUMS:
            assert secret not in text, f"leaked {secret!r}:\n{text}"


def test_schema_has_no_values():
    schema = prompts.describe_schema(make_df())
    assert "'site_name'" in schema and "pm25" in schema
    assert_no_leak([schema])


@pytest.mark.parametrize("bad_code", [
    "x = float(df['site_name'].iloc[0])",                         # quoted value in message
    "x = pd.to_datetime(df['date'], format='%Y-%m-%d')",            # unquoted date/text values
    "raise ValueError('bad ' + str(df['pm25'].iloc[0]))",           # number in message
    "raise ValueError(df['respondent'].iloc[1] + ' missing')",       # unquoted ID
    "raise KeyError(df['site_name'].iloc[2])",                       # KeyError repr
    "x = df.loc[df['site_name'] == 'zzz', 'pm25'].iloc[0]",          # IndexError
    "result_typo = df",                                              # no result
])
def test_errors_are_sanitised_and_retried(bad_code):
    llm, prompts = fake_llm([bad_code, "result = df['pm25'].mean()"])
    turn = analysis.analyse("average pm25?", make_df(), "ctx", [], False, llm)
    assert turn["error"] is None, turn["error"]
    assert turn["attempts"] == 2
    assert turn["result"] == pytest.approx((123.45 + 678.91 + 246.8) / 3)
    assert len(prompts) == 2 and "Your previous attempt failed" in prompts[1]
    assert_no_leak(prompts)
    print("\nSENT ERROR:", next(l for l in prompts[1].splitlines() if l.startswith("Error: ")))


def test_gives_up_after_max_attempts():
    llm, prompts = fake_llm(["x = float(df['site_name'].iloc[0])"])
    turn = analysis.analyse("q", make_df(), "", [], False, llm)
    assert turn["attempts"] == analysis.MAX_ATTEMPTS and turn["error"]
    assert "Anand Vihar" in turn["error"]          # full error shown locally only
    assert_no_leak(prompts)


def test_api_failure_not_retried():
    def llm(prompt):
        raise Exception("HTTP Error 400: bad key")
    turn = analysis.analyse("q", make_df(), "", [], False, llm)
    assert turn["attempts"] == 1 and "Gemini request failed" in turn["error"]


def test_recorder_outputs_and_streamlit_import():
    code = (
        "import streamlit as st2\n"
        "from streamlit import markdown\n"
        "st2.write('### Metadata')\n"
        "markdown('hello')\n"
        "st.dataframe(df.head(2))\n"
        "fig, ax = plt.subplots(); ax.plot([1, 2]); st.pyplot(fig)\n"
        "fig2, ax2 = plt.subplots(); ax2.plot([3, 4])\n"      # never shown explicitly
        "print('printed line')\n"
        "result = 'Plotted'\n"
    )
    llm, _ = fake_llm([code])
    turn = analysis.analyse("q", make_df(), "", [], False, llm)
    assert turn["error"] is None, turn["error"]
    kinds = [o[0] for o in turn["outputs"]]
    assert kinds == ["write", "markdown", "dataframe", "image", "image", "text"], kinds


def test_layout_calls_trigger_retry():
    llm, prompts = fake_llm(["c1, c2 = st.columns(2)\nresult = 1", "result = 1"])
    turn = analysis.analyse("q", make_df(), "", [], False, llm)
    assert turn["attempts"] == 2 and "st.columns is not available" in prompts[1]


def test_anonymise():
    df = make_df()
    llm, prompts = fake_llm(["result = df.groupby('col_1')['col_4'].mean()"])
    turn = analysis.analyse("mean pm25 by Site_Name", df, "use pm25", [], True, llm)
    assert turn["error"] is None, turn["error"]
    for name in df.columns:
        # As a word: the fixed instructions may name pandas functions such as `pd.to_datetime`.
        assert not re.search(rf"\b{re.escape(name)}\b", prompts[0].lower().replace("col_", "#")), name
    assert "mean col_4 by col_1" in prompts[0]
    assert turn["result"].index.name == "site_name"
    assert_no_leak(prompts)


def test_follow_up_sends_code_not_results():
    llm, prompts = fake_llm(["result = df[df['pm25'] > 200]"])
    first = analysis.analyse("sites above 200", make_df(), "", [], False, llm)
    llm2, prompts2 = fake_llm(["result = len(df)"])
    analysis.analyse("how many of those?", make_df(), "", [first], False, llm2)
    assert "Q: sites above 200" in prompts2[0] and "df['pm25'] > 200" in prompts2[0]
    assert_no_leak(prompts2)


def test_input_df_not_mutated():
    df = make_df()
    llm, _ = fake_llm(["df['pm25'] = 0\nresult = 1"])
    analysis.analyse("q", df, "", [], False, llm)
    assert df["pm25"].iloc[0] == 123.45


# Progress messages shown in the page's status box

def test_progress_messages_for_retry_then_success():
    steps = []
    llm, _ = fake_llm(["x = float(df['site_name'].iloc[0])", "result = 1"])
    turn = analysis.analyse("q", make_df(), "", [], False, llm, progress=steps.append)
    assert turn["error"] is None
    assert steps == [
        "Asking Gemini to write the code (attempt 1 of 3)…",
        "Running the code on your computer…",
        "The code did not work (ValueError). Retrying…",
        "Asking Gemini to fix the code (attempt 2 of 3)…",
        "Running the code on your computer…",
        "Done.",
    ]
    assert_no_leak(steps)


def test_progress_messages_when_all_attempts_fail_or_api_fails():
    steps = []
    llm, _ = fake_llm(["result = df['nope']"])
    analysis.analyse("q", make_df(), "", [], False, llm, progress=steps.append)
    assert steps[-1] == "The code did not work (KeyError). No attempts left."

    steps = []

    def broken(prompt):
        raise RuntimeError("HTTP 500")
    analysis.analyse("q", make_df(), "", [], False, broken, progress=steps.append)
    assert steps == ["Asking Gemini to write the code (attempt 1 of 3)…", "Gemini request failed."]


def test_progress_is_optional():
    llm, _ = fake_llm(["result = 1"])
    assert analysis.analyse("q", make_df(), "", [], False, llm)["result"] == 1
