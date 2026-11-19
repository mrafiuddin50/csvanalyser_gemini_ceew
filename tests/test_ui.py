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
    # One 'Take the tour' button (in the header), plain HTML handled by tour.js
    tour_buttons = [m.value for m in at.markdown if "tda-tour-start" in m.value]
    assert len(tour_buttons) == 1 and "tda-primary" in tour_buttons[0]
    assert any("[Printable guide](guide.html)" in c.value for c in at.caption)      # footer
    # Status line and an empty activity log are always present
    assert any("**Status:** ready. Upload a file" in c.value for c in at.caption)
    assert any(e.label == "Activity log (0)" for e in at.sidebar.expander)


def _status_line(at):
    return next(c.value for c in at.caption if c.value.startswith("**Status:**"))


ASK_TAB, DATA_TAB, EXPORT_TAB = "② Ask questions", "① Data & quality", "③ Export"


def _open(at, key, label):
    """Selects a tab, as a click would. From Streamlit 1.65 only the open tab's contents are built."""
    at.session_state[key] = label
    return at.run()


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
        _open(at, "main_tabs", ASK_TAB)
        at.text_input(key="query").input("average pm25").run()
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
    _open(at, "main_tabs", ASK_TAB)
    at.text_input(key="query").input("average pm25").run()
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
    at.session_state["main_tabs"] = ASK_TAB
    with mock.patch("urllib.request.urlopen", fake_urlopen):
        at.run()
        at.sidebar.text_input[0].input("fake-key")
        at.text_input(key="query").input("mean pm25 by site")
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
        assert len(at.dataframe) >= 1  # the result table
        # Follow-up question includes the previous question and code
        at.text_input(key="query").input("how many rows?")
        [b for b in at.button if b.label == "Analyse"][0].click().run()
        assert not at.exception, at.exception
        assert "Q: mean pm25 by site" in SENT[-1]
        assert len(at.session_state["history"]) == 2
    for prompt in SENT:
        for secret in ["Anand Vihar", "Punjabi Bagh", "123.45", "678.91"]:
            assert secret not in prompt


def _button(at, label):
    return [b for b in at.button if b.label == label][0]


def _images(at):
    # AppTest calls this element "imgs" before Streamlit 1.65 and "image" from then on.
    return list(at.main.get("image")) + list(at.main.get("imgs"))


def test_ceew_chart_with_the_researchers_words_and_downloads():
    SENT.clear()
    REPLIES[:] = ["t = df.groupby('site_name', as_index=False)['pm25'].mean()\n"
                  "ceew.bar(t, 'site_name', 'pm25', why='Magnitude: one value per site.')\nresult = t"]
    at = AppTest.from_file(APP, default_timeout=TIMEOUT)
    at.session_state["df"] = df()
    at.session_state["file_name"] = "test.csv"
    at.session_state["main_tabs"] = ASK_TAB
    with mock.patch("urllib.request.urlopen", fake_urlopen):
        at.run()
        at.sidebar.text_input[0].input("fake-key")
        at.text_input(key="query").input("mean pm25 by site as a chart")
        at.run()
        _button(at, "Analyse").click().run()
        assert not at.exception, at.exception
    turn = at.session_state["history"][0]
    turn_id = turn["id"]
    assert turn["outputs"][0][0] == "chart"
    assert len(_images(at)) == 1
    assert any("Gemini's reason for this chart:** Magnitude" in c.value for c in at.caption)
    assert any("largest to smallest" in c.value for c in at.caption)           # a tip from the chart rules
    first_png = turn["chart_png"][0][1]
    labels = {d.proto.label for d in at.get("download_button")}
    assert {"Download PNG (300 dpi)", "Download result as CSV"} <= labels

    # The researcher writes the words; the chart is redrawn locally and the headline is checked.
    at.text_input(key=f"chartfield_headline_0_{turn_id}").input("Pollution is high because of traffic")
    at.text_input(key=f"chartfield_source_0_{turn_id}").input("Test data")
    _button(at, "Update the chart").click().run()
    assert not at.exception, at.exception
    turn = at.session_state["history"][0]
    assert turn["chart_texts"][0]["headline"] == "Pollution is high because of traffic"
    assert turn["chart_png"][0][1] != first_png
    assert any("“because of” states a cause" in w.value for w in at.warning)
    assert at.text_input(key=f"chartfield_x_title_0_{turn_id}").value == "pm25"     # from the column name

    # An editable SVG on request, named after the headline.
    at.button(key=f"chart_make_svg_0_{turn_id}").click().run()
    svg = at.session_state["history"][0]["chart_svg"][0][1]
    assert b"Pollution is high because of traffic" in svg and b"Source: Test data" in svg
    assert any(d.proto.label == "Download SVG" for d in at.get("download_button"))

    # Changing the chart style redraws the chart.
    at.sidebar.radio(key="chart_use").set_value("outreach").run()
    assert not at.exception, at.exception
    assert at.session_state["history"][0]["chart_png"][0][0][0] == "outreach"

    # Only the question and column names went to Gemini; the words stayed here.
    assert len(SENT) == 1
    for secret in ["Anand Vihar", "Punjabi Bagh", "123.45", "678.91", "because of traffic"]:
        assert secret not in SENT[0]


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
    assert any("**Overall assessment:** Needs Attention" in m.value for m in at.markdown)
    assert any(m.label == "Warnings" and m.value == "2" for m in at.metric)
    assert any(s.value == "What you can analyse" for s in at.subheader)
    report = at.session_state["audit"]
    assert report.readiness == "READY WITH CAUTION"
    assert at.session_state["audit_html"].startswith("<!DOCTYPE html>")
    _open(at, "data_tabs", "Column details")
    assert any(m.label == "Duplicate rows" for m in at.metric)
    # Example questions built from column names/types only
    _open(at, "main_tabs", ASK_TAB)
    examples = [b.label for b in at.button if (b.key or "").startswith("example_")]
    assert "Average pm25_ugm3 for each station" in examples
    assert len(examples) == 5
    # Before any answer the notebook/script buttons are shown but inactive
    _open(at, "main_tabs", EXPORT_TAB)
    downloads = {d.proto.label: d.proto.disabled for d in at.get("download_button")}
    assert downloads["Jupyter notebook (.ipynb)"] is True and downloads["Python script (.py)"] is True

    with mock.patch("urllib.request.urlopen", fake_urlopen):
        at.sidebar.text_input[0].input("fake-key").run()
        _open(at, "main_tabs", ASK_TAB)
        at.text_input(key="query").input("average pm25 by station").run()
        _button(at, "Analyse").click().run()
        assert not at.exception, at.exception
        assert len(SENT) == 1
        labels = {d.proto.label for d in at.get("download_button")}
        assert {"Download result as CSV", "Download the sample as CSV"} <= labels
        _open(at, "main_tabs", EXPORT_TAB)
        downloads = {d.proto.label: d.proto.disabled for d in at.get("download_button")}
        assert downloads["Jupyter notebook (.ipynb)"] is False and downloads["Python script (.py)"] is False
        assert "Download the audit report (HTML)" in downloads
        _open(at, "main_tabs", ASK_TAB)
        # The answer is a card with Result | Code | Sent to Gemini tabs
        labels = [t.label for t in at.tabs]
        assert {"Result", "Code", "Sent to Gemini (1)"} <= set(labels)

        # Edit the code and run locally: no new request
        turn_id = at.session_state["history"][0]["id"]
        _open(at, f"card_tabs_{turn_id}", "Code")
        at.text_area(key=f"code_{turn_id}").input("result = df['pm25_ugm3'].max()").run()
        _button(at, "Run my version").click().run()
        assert not at.exception, at.exception
        assert len(SENT) == 1
        edited = at.session_state["history"][-1]
        assert edited["edited"] and edited["sent_prompts"] == []
        _open(at, f"card_tabs_{edited['id']}", "Sent to Gemini (0)")
        assert any("Nothing was sent to Gemini" in c.value for c in at.caption)
        # The edited text is kept while the answer's Code section is hidden and shown again
        _open(at, f"card_tabs_{turn_id}", "Result")
        _open(at, f"card_tabs_{turn_id}", "Code")
        assert at.text_area(key=f"code_{turn_id}").value == "result = df['pm25_ugm3'].max()"

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
