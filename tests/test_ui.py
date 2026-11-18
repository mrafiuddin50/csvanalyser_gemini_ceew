"""The Streamlit page, driven with Streamlit's AppTest and a fake Gemini."""
import io
import json
import pathlib
from unittest import mock

import pandas as pd
from streamlit.testing.v1 import AppTest

APP = str(pathlib.Path(__file__).resolve().parents[1] / "stlite_app.py")
TIMEOUT = 120   # seconds per script run; the first run imports pandas, matplotlib, etc.
SENT = []
REPLIES = []


class FakeResponse(io.BytesIO):
    status = 200

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


REQUESTS = []


def fake_urlopen(req, *a, **k):
    REQUESTS.append(req)
    assert "key=" not in req.full_url
    assert req.get_header("X-goog-api-key") == "fake-key"
    if req.get_method() == "GET":
        models = {"models": [
            {"name": "models/gemini-2.5-flash", "supportedGenerationMethods": ["generateContent"]},
            {"name": "models/zz-other-model", "supportedGenerationMethods": ["generateContent"]},
            {"name": "models/text-embedding-004", "supportedGenerationMethods": ["embedContent"]},
            {"name": "models/gemini-2.5-flash-preview-tts", "supportedGenerationMethods": ["generateContent"]},
        ]}
        return FakeResponse(json.dumps(models).encode())
    SENT.append(json.loads(req.data)["contents"][0]["parts"][0]["text"])
    code = REPLIES.pop(0)
    body = {"candidates": [{"content": {"parts": [{"text": f"```python\n{code}\n```"}]}}]}
    return FakeResponse(json.dumps(body).encode())


def df():
    return pd.DataFrame({"site_name": ["Anand Vihar", "Punjabi Bagh"], "pm25": [123.45, 678.91]})


def test_no_file_shows_info():
    at = AppTest.from_file(APP, default_timeout=TIMEOUT).run()
    assert not at.exception
    # Welcome screen with the three steps, sample and tour buttons
    assert any(s.value == "Get started in three steps" for s in at.subheader)
    assert at.button(key="sample_welcome").label == "Try the synthetic sample"
    assert at.button(key="tour_welcome").label == "🧭 Take the tour"
    assert at.button(key="tour_header").label == "🧭 Take the tour"
    assert any("[Printable guide](guide.html)" in c.value for c in at.caption)      # footer
    # Status line and an empty activity log are always present
    assert any("**Status:** ready. Upload a file" in c.value for c in at.caption)
    assert any(e.label == "Activity log (0)" for e in at.sidebar.expander)


def _status_line(at):
    return next(c.value for c in at.caption if c.value.startswith("**Status:**"))


def test_status_line_and_activity_log_track_each_action():
    SENT.clear()
    REPLIES[:] = ["x = float(df['station'].iloc[0])", "result = df['pm25_ugm3'].mean()"]
    at = AppTest.from_file(APP, default_timeout=TIMEOUT).run()
    at.button(key="sample_sidebar").click().run()
    assert not at.exception, at.exception
    assert "Loaded the synthetic sample: 1,835 rows; data quality: Needs Attention" in _status_line(at)

    with mock.patch("urllib.request.urlopen", fake_urlopen):
        at.sidebar.text_input[0].input("fake-key").run()
        assert "API key works: 2 models available" in _status_line(at)
        at.text_input[0].input("average pm25").run()
        [b for b in at.button if b.label == "Analyse"][0].click().run()
        assert not at.exception, at.exception
    assert "Answered the question after 2 attempts" in _status_line(at)

    log = at.session_state["activity"]
    assert [e["message"] for e in log] == [
        "Loaded the synthetic sample: 1,835 rows; data quality: Needs Attention, readiness: READY WITH CAUTION",
        "API key works: 2 models available",
        "Answered the question after 2 attempts",
    ]
    assert all(e["ok"] and e["seconds"] >= 0 for e in log)
    assert any(e.label == "Activity log (3)" for e in at.sidebar.expander)
    assert at.session_state["pending"] is None


def test_failures_are_reported_in_the_status_line():
    at = AppTest.from_file(APP, default_timeout=TIMEOUT).run()
    at.button(key="sample_sidebar").click().run()
    at.text_input[0].input("average pm25").run()
    [b for b in at.button if b.label == "Analyse"][0].click().run()          # no key entered
    assert "⚠️ Analysis not started: no API key entered" in _status_line(at)
    with mock.patch("urllib.request.urlopen", side_effect=OSError("offline")):
        at.sidebar.text_input[0].input("some-key").run()
    assert "⚠️ API key check failed" in _status_line(at)
    assert not at.session_state["activity"][-1]["ok"]


def test_full_turn_with_retry_and_history():
    SENT.clear()
    REPLIES[:] = [
        "x = float(df['site_name'].iloc[0])",                                   # fails
        "st.subheader('Means')\nfig, ax = plt.subplots(); ax.bar([0, 1], df['pm25']); st.pyplot(fig)\nresult = df.groupby('site_name')['pm25'].mean()",
        "result = len(df)",
    ]
    at = AppTest.from_file(APP, default_timeout=TIMEOUT)
    at.session_state["df"] = df()
    at.session_state["file_name"] = "test.csv"
    with mock.patch("urllib.request.urlopen", fake_urlopen):
        at.run()
        at.sidebar.text_input[0].input("fake-key")
        at.text_input[0].input("mean pm25 by site")
        at.run()
        # Key check fetched the model list; non-text models filtered out
        assert at.sidebar.selectbox[0].options == ["gemini-2.5-flash", "zz-other-model"]
        assert "Key works. 2 models" in at.sidebar.success[0].value
        assert not any(b.label == "Enter" for b in at.sidebar.button)
        # Preview expander shows the prompt before sending
        assert any("What will be sent" in e.label for e in at.expander)
        [b for b in at.button if b.label == "Analyse"][0].click().run()
        assert not at.exception, at.exception
        assert len(SENT) == 2
        assert at.session_state["history"][0]["attempts"] == 2
        # Rerun (e.g. any click) keeps the result on screen
        at.run()
        # The answer's own subheader (the audit section adds its own headings too)
        assert [s.value for s in at.main.subheader].count("Means") == 1
        # AppTest calls this element "imgs" before Streamlit 1.65 and "image" from then on.
        assert len(at.main.get("image")) + len(at.main.get("imgs")) == 1
        assert len(at.dataframe) >= 2  # preview + result
        # Follow-up question includes the previous question and code
        at.text_input[0].input("how many rows?")
        [b for b in at.button if b.label == "Analyse"][0].click().run()
        assert not at.exception, at.exception
        assert "Q: mean pm25 by site" in SENT[-1]
        assert len(at.session_state["history"]) == 2
    for prompt in SENT:
        for secret in ["Anand Vihar", "Punjabi Bagh", "123.45", "678.91"]:
            assert secret not in prompt


def _button(at, label):
    return [b for b in at.button if b.label == label][0]


def test_sample_profile_edit_explain_and_downloads():
    SENT.clear()
    REPLIES[:] = [
        "result = df.groupby('station')['pm25_ugm3'].mean()",
        "This code groups the rows by station and averages PM2.5.",   # explanation
    ]
    at = AppTest.from_file(APP, default_timeout=TIMEOUT).run()
    assert any("Your data never leaves your computer" in m.value for m in at.markdown)
    # Help popover holds the data-handling note and the memory tips
    assert any(m.value == "#### How your data is handled" for m in at.markdown)
    assert any(m.value == "#### Memory and performance tips" for m in at.markdown)
    assert at.title[0].value == "Tabular Data Analyser"
    at.button(key="sample_sidebar").click().run()
    assert not at.exception, at.exception
    assert "synthetic sample data" in at.warning[0].value
    # Data quality audit: verdict banner, audit tab and the 'what you can analyse' table
    banner = at.warning[1].value
    assert "Data quality: Needs Attention. Analysis readiness: READY WITH CAUTION." in banner
    labels = [t.label for t in at.tabs]
    for label in ["① Data & quality", "② Ask questions", "③ Export",
                  "Data quality audit", "Preview", "Column details"]:
        assert labels.count(label) == 1, label
    # Example questions built from column names/types only
    examples = [b.label for b in at.button if (b.key or "").startswith("example_")]
    assert "Average pm25_ugm3 for each station" in examples
    assert len(examples) == 5
    assert any("**Overall assessment:** Needs Attention" in m.value for m in at.markdown)
    assert any(m.label == "Warnings" and m.value == "2" for m in at.metric)
    assert any(m.label == "Duplicate rows" for m in at.metric)              # Column details tab
    assert any(s.value == "What you can analyse" for s in at.subheader)
    report = at.session_state["audit"]
    assert report.readiness == "READY WITH CAUTION"
    assert at.session_state["audit_html"].startswith("<!DOCTYPE html>")

    with mock.patch("urllib.request.urlopen", fake_urlopen):
        at.sidebar.text_input[0].input("fake-key").run()
        at.text_input[0].input("average pm25 by station").run()
        _button(at, "Analyse").click().run()
        assert not at.exception, at.exception
        assert len(SENT) == 1
        # result CSV + sample CSV + audit report + .py + .ipynb
        # result CSV, sample CSV, audit report (data tab and export tab), .py, .ipynb
        assert len(at.get("download_button")) == 6
        # The answer is a card with Result | Code | Sent to Gemini tabs
        labels = [t.label for t in at.tabs]
        assert {"Result", "Code", "Sent to Gemini (1)"} <= set(labels)

        # Edit the code and run locally: no new request
        turn_id = at.session_state["history"][0]["id"]
        at.text_area(key=f"code_{turn_id}").input("result = df['pm25_ugm3'].max()").run()
        _button(at, "Run my version").click().run()
        assert not at.exception, at.exception
        assert len(SENT) == 1
        edited = at.session_state["history"][-1]
        assert edited["edited"] and edited["sent_prompts"] == []
        assert any("Nothing was sent to Gemini" in c.value for c in at.caption)

        # Explain the original answer's code
        at.checkbox(key=f"show_explain_{turn_id}").check().run()
        [b for b in at.button if b.key == f"explain_{turn_id}"][0].click().run()
        assert not at.exception, at.exception
        assert len(SENT) == 2 and "Explain what the code below does" in SENT[1]
        assert "groups the rows by station" in at.session_state["history"][0]["explanation"]
        assert len(at.session_state["history"][0]["sent_prompts"]) == 2

    sample = at.session_state["df"]
    audit_phrases = [report.assessment, report.readiness, "exact copies", "written in more than one way",
                     "outliers", "stored as text", "What you can analyse"]
    for prompt in SENT:
        for value in sample["station"].unique()[:5]:
            assert f"'{value}'" not in prompt or value in ("station",)
        assert str(sample["pm25_ugm3"].dropna().iloc[0]) not in prompt
        for phrase in audit_phrases:                 # the audit stays in the browser
            assert phrase not in prompt, phrase
