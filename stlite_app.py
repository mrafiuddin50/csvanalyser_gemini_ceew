"""Streamlit interface. Runs in the browser through stlite (see index.html).

The logic lives in the csvanalyser package; this file only draws the page.
"""
import os
import time

import pandas as pd
import streamlit as st

from csvanalyser import audit
from csvanalyser.analysis import analyse, past_turns_for_prompt, prepare_request, run_edited_code
from csvanalyser.charts import PAIR_LABELS, SHAPES, USES, Style, check_headline, file_stem, render as render_chart
from csvanalyser.export import (CHARTS_FILE_NAME, build_notebook, build_script, charts_module_source,
                                exportable_turns, uses_ceew_charts)
from csvanalyser.gemini import FALLBACK_MODELS, GeminiError, call_gemini_text, generate_python_code, list_models
from csvanalyser.loading import UPLOAD_TYPES, load_table
from csvanalyser.privacy import restore_names
from csvanalyser.profiling import profile_dataset
from csvanalyser.prompts import build_explain_prompt, build_prompt
from csvanalyser.sample import SAMPLE_FILE_NAME, make_sample_dataset
from csvanalyser.suggestions import suggest_questions


DEFAULT_CONTEXT = (
    "Return the final answer strictly as a pandas dataframe "
    "(e.g. result = summary_df). Streamlit will render it automatically."
)


def _set_dataset(df, name=None, description=None, reader_code=None, is_sample=False):
    """Stores the loaded data and resets everything that depended on the old data."""
    st.session_state["history"] = []
    st.session_state["df"] = df
    st.session_state["file_name"] = name
    st.session_state["file_description"] = description
    st.session_state["reader_code"] = reader_code
    st.session_state["is_sample"] = is_sample
    # New data: start on the data-quality view.
    st.session_state["main_tabs"] = "① Data & quality"
    st.session_state["data_tabs"] = "Data quality audit"
    # Data quality audit: runs once per file, locally; never sent to Gemini.
    st.session_state["audit"], st.session_state["audit_html"], st.session_state["audit_error"] = None, None, None
    report = None
    if df is not None:
        try:
            report = audit.run_audit(df, name or "data")
            st.session_state["audit"] = report
            st.session_state["audit_html"] = audit.build_html_report(report)
        except Exception as exc:  # the audit must never stop people using their data
            st.session_state["audit_error"] = f"{type(exc).__name__}: {exc}"
    # Column details reuse the audit's counts rather than going through the data again.
    st.session_state["profile"] = profile_dataset(df, report) if df is not None else None


# ---------------------------------------------------------------------------
# Background tasks with visible progress
#
# Widget callbacks run before the page is redrawn, so nothing they do can be shown
# while it happens. Instead, callbacks only queue a task; the page then runs it
# inside a status box that shows each step, and records the outcome in the
# activity log behind the status line.
# ---------------------------------------------------------------------------

def _queue(task: str, **details):
    st.session_state["pending"] = {"task": task, **details}


def _log(message: str, ok: bool = True, seconds: float | None = None):
    log = st.session_state.setdefault("activity", [])
    log.append({"ok": ok, "message": message, "seconds": seconds})
    del log[:-20]


def _audit_summary():
    report = st.session_state.get("audit")
    if report is None:
        return "data quality audit could not be completed"
    return f"data quality: {report.assessment}, readiness: {report.readiness}"


def _task_load_file(step):
    uploaded = st.session_state.get("uploaded_csv")
    st.session_state["load_error"] = None
    if uploaded is None:
        _set_dataset(None)
        return True, "File removed"
    step(f"Reading {uploaded.name} ({uploaded.size / 1024:,.0f} KB)…")
    try:
        df, description, reader_code = load_table(uploaded.name, uploaded.getvalue())
    except Exception as exc:
        _set_dataset(None)
        st.session_state["load_error"] = (
            f"Could not read {uploaded.name}: {exc}. "
            "Check that it is a CSV, TSV or Excel (.xlsx) file with a header row."
        )
        return False, f"Could not read {uploaded.name}"
    if df.empty and len(df.columns) == 0:
        _set_dataset(None)
        st.session_state["load_error"] = f"{uploaded.name} has no columns."
        return False, f"{uploaded.name} has no columns"
    step(f"Read {len(df):,} rows and {df.shape[1]} columns ({description}).")
    step("Checking data quality…")
    _set_dataset(df, uploaded.name, description, reader_code)
    return True, f"Loaded {uploaded.name}: {len(df):,} rows; {_audit_summary()}"


def _task_load_sample(step):
    st.session_state["load_error"] = None
    step("Creating the synthetic sample dataset…")
    df = make_sample_dataset()
    step(f"Created {len(df):,} rows and {df.shape[1]} columns.")
    step("Checking data quality…")
    _set_dataset(df, SAMPLE_FILE_NAME, "synthetic sample data (not real measurements)",
                 f"pd.read_csv({SAMPLE_FILE_NAME!r})", is_sample=True)
    return True, f"Loaded the synthetic sample: {len(df):,} rows; {_audit_summary()}"


def _on_upload():
    _queue("load_file")


def _load_sample():
    _queue("load_sample")


KEEP_RESULTS = 10   # answers that keep their result tables and charts; older ones keep question and code only


def _add_turn(turn: dict):
    st.session_state.setdefault("next_turn_id", 1)
    turn["id"] = st.session_state["next_turn_id"]
    st.session_state["next_turn_id"] += 1
    history = st.session_state["history"]
    history.append(turn)
    # Results stay in memory for the whole session; keep only the latest ones.
    for old in history[:-KEEP_RESULTS]:
        if old.get("result") is not None or old.get("outputs"):
            old["result"], old["outputs"], old["csv"], old["result_dropped"] = None, [], None, True
            old["chart_png"], old["chart_svg"] = {}, {}


def _find_turn(turn_id):
    return next((t for t in st.session_state.get("history", []) if t.get("id") == turn_id), None)


def _task_run_edited(step, turn_id):
    turn = _find_turn(turn_id)
    df = st.session_state.get("df")
    if turn is None or df is None:
        return False, "That answer is no longer available"
    code = st.session_state.get(f"code_{turn_id}", turn["code"])
    step("Running your version of the code on your computer (nothing is sent to Gemini)…")
    new_turn = run_edited_code(turn, code, df, _chart_style())
    # Keep the headline, source and note the researcher already typed for these charts.
    new_turn["chart_texts"] = {index: dict(texts) for index, texts in (turn.get("chart_texts") or {}).items()}
    _add_turn(new_turn)
    if new_turn["error"]:
        return False, f"Your edited code failed: {new_turn['error'].split(':')[0]}"
    return True, "Ran your edited code"


def _task_explain(step, turn_id):
    turn = _find_turn(turn_id)
    df = st.session_state.get("df")
    if turn is None or df is None:
        return False, "That answer is no longer available"
    turn["explanation"], turn["explanation_error"] = None, None
    api_key = (st.session_state.get("api_key") or "").strip()
    if not api_key:
        turn["explanation_error"] = "Enter a Gemini API key in the sidebar to get an explanation."
        return False, "No API key entered"
    request = prepare_request(df, "", "", turn.get("anonymise", False))
    code = st.session_state.get(f"code_{turn_id}", turn["code"])
    prompt = build_explain_prompt(request["schema"], code)
    turn["sent_prompts"].append(prompt)
    model = st.session_state.get("model") or FALLBACK_MODELS[0]
    step(f"Sending the code (and column names only) to {model}…")
    try:
        turn["explanation"] = restore_names(call_gemini_text(prompt, model, api_key), request["reverse"])
    except GeminiError as exc:
        turn["explanation_error"] = str(exc)
        return False, "Explanation failed"
    return True, "Explained the code"


def _task_check_key(step):
    """Checks a newly entered key by listing the models it can use."""
    st.session_state["models"] = None
    st.session_state["key_status"] = None
    key = (st.session_state.get("api_key") or "").strip()
    if not key:
        return True, "API key cleared"
    step("Asking Google which models this key can use…")
    try:
        models = list_models(key)
    except GeminiError as exc:
        st.session_state["key_status"] = ("error", str(exc))
        return False, "API key check failed"
    if not models:
        st.session_state["key_status"] = ("error", "The key works, but no text models are available to it.")
        return False, "API key has no text models"
    st.session_state["models"] = models
    st.session_state["key_status"] = ("ok", f"Key works. {len(models)} models available.")
    return True, f"API key works: {len(models)} models available"


def _on_run_edited(turn_id):
    _queue("run_edited", turn_id=turn_id)


def _on_explain(turn_id):
    _queue("explain", turn_id=turn_id)


def _on_key_change():
    _queue("check_key")


# task name: (status box title, function, where the status box appears)
TASKS = {
    "load_file": ("Loading your file", _task_load_file, "main"),
    "load_sample": ("Loading the synthetic sample", _task_load_sample, "main"),
    "check_key": ("Checking your API key", _task_check_key, "sidebar"),
    "run_edited": ("Running your edited code", _task_run_edited, "answers"),
    "explain": ("Asking Gemini to explain the code", _task_explain, "answers"),
}


def run_pending_task(area: str):
    """Runs a queued task in a live status box, if one is waiting for this part of the page."""
    pending = st.session_state.get("pending")
    if not pending or TASKS.get(pending["task"], (None, None, None))[2] != area:
        return
    st.session_state["pending"] = None
    title, task, _ = TASKS[pending["task"]]
    details = {k: v for k, v in pending.items() if k != "task"}
    place = st.sidebar if area == "sidebar" else st
    start = time.perf_counter()
    with place.status(f"{title}…", expanded=True) as box:
        try:
            ok, summary = task(box.write, **details)
        except Exception as exc:  # report, never crash the page
            ok, summary = False, f"{title} failed ({type(exc).__name__}: {exc})"
        seconds = time.perf_counter() - start
        box.update(label=f"{summary} ({seconds:.1f} s)", state="complete" if ok else "error", expanded=not ok)
    _log(summary, ok, seconds)


def render_status_line(slot):
    log = st.session_state.get("activity") or []
    if not log:
        slot.caption("**Status:** ready. Upload a file or try the synthetic sample to begin.")
        return
    last = log[-1]
    icon = "✅" if last["ok"] else "⚠️"
    took = f" ({last['seconds']:.1f} s)" if last.get("seconds") is not None else ""
    slot.caption(f"**Status:** ready · Last action: {icon} {last['message']}{took} · "
                 "full list under *Activity log* in the sidebar")


def render_activity_log():
    log = st.session_state.get("activity") or []
    with st.sidebar.container(key="tour-activity"), st.expander(f"Activity log ({len(log)})"):
        if not log:
            st.caption("Nothing has happened yet.")
        for entry in reversed(log):
            icon = "✅" if entry["ok"] else "⚠️"
            took = f" ({entry['seconds']:.1f} s)" if entry.get("seconds") is not None else ""
            st.markdown(f"{icon} {entry['message']}{took}")
        st.caption("While something is running, its progress appears in a box on the page, and the "
                   "running indicator at the top right of the page is active.")


def _clear_history():
    st.session_state["history"] = []


def _result_csv(turn: dict) -> bytes:
    """The answer's result as CSV, written once and kept with the answer.

    Writing a large table as CSV on every page redraw was slow (seconds for 200,000 rows).
    Streamlit can write a download only when it is clicked (data=a function), but in the
    browser version (stlite 1.12) that gives a link to an address that does not exist, so
    the file is written in advance. It is dropped together with the result (KEEP_RESULTS).
    """
    if turn.get("csv") is None:
        turn["csv"] = turn["result"].to_csv().encode("utf-8")
    return turn["csv"]


def _show_value(value):
    if isinstance(value, (pd.DataFrame, pd.Series)):
        st.dataframe(value)
    elif value is not None:
        st.write(value)


def _render_result(turn: dict):
    turn_id = turn.get("id", 0)
    if turn.get("result_dropped"):
        st.info(f"To save memory, only the latest {KEEP_RESULTS} answers keep their results. The question "
                "and code are kept: open **Code** and press **Run my version** to see this result again.")
    for index, (name, args, kwargs) in enumerate(turn["outputs"]):
        try:
            if name == "chart":
                _render_chart(turn, index, args[0])
            elif name == "image":
                st.image(args[0])
            else:
                getattr(st, name)(*args, **kwargs)
        except Exception as exc:
            st.warning(f"Could not display an output ({'chart' if name == 'chart' else 'st.' + name}): {exc}")
    if turn["error"]:
        st.error(f"Error during execution: {turn['error']}")
        return
    _show_value(turn["result"])
    if isinstance(turn["result"], (pd.DataFrame, pd.Series)):
        st.download_button(
            "Download result as CSV", data=_result_csv(turn),
            file_name=f"result_{turn_id}.csv", mime="text/csv", key=f"download_{turn_id}",
        )
    if turn["attempts"] > 1:
        st.caption(f"The first attempt failed; fixed automatically after {turn['attempts']} attempts.")


# ---------------------------------------------------------------------------
# CEEW charts
#
# Charts from the ceew.* functions are stored as specifications and drawn here, so the researcher's
# headline, source and note (and the chart style in the sidebar) can be applied without asking Gemini
# again. Nothing here is sent anywhere.
# ---------------------------------------------------------------------------

CHART_DISPLAY_WIDTH = {"report": 680, "presentation": 460, "outreach": 460}
CHART_FIELDS = [
    ("headline", "Headline",
     "CEEW headlines state the insight, not a description of the data: for example 'Gujarat added more rooftop "
     "solar than the next three states combined', not 'Rooftop solar capacity by state, 2024'. It must claim no "
     "more than the chart shows."),
    ("x_title", "Horizontal axis title", "Taken from your column name; put the unit here (for example 'MW'), "
                                         "not in the tick labels. Write INR, not the rupee symbol."),
    ("y_title", "Vertical axis title", "Taken from your column name; put the unit here, not in the tick labels."),
    ("source", "Source", "Every CEEW chart needs a source line. The app never writes or guesses one."),
    ("note", "Note", "For example 'Figures may not add to 100% due to rounding'. For survey data, give the sample "
                     "size (n = …)."),
]


def _chart_style() -> Style:
    return Style(use=st.session_state.get("chart_use") or "report",
                 shape=st.session_state.get("chart_shape") or "square",
                 pair=st.session_state.get("chart_pair") or "orange-blue")


def _chart_key(style: Style, texts: dict):
    return (style.use, style.shape, style.pair, tuple(sorted(texts.items())))


def _chart_png(turn: dict, index: int, spec) -> bytes:
    """The chart at 300 dpi, drawn again only when the style or the researcher's text changes."""
    texts = (turn.get("chart_texts") or {}).get(index, {})
    style = _chart_style()
    key = _chart_key(style, texts)
    cached = (turn.setdefault("chart_png", {})).get(index)
    if cached is None or cached[0] != key:
        cached = (key, render_chart(spec, style, texts))
        turn["chart_png"][index] = cached
    return cached[1]


def _chart_field_key(field: str, index: int, turn_id) -> str:
    return f"chartfield_{field}_{index}_{turn_id}"


def _on_chart_text(turn_id, index):
    turn = _find_turn(turn_id)
    if turn is None:
        return
    texts = {field: (st.session_state.get(_chart_field_key(field, index, turn_id)) or "").strip()
             for field, _, _ in CHART_FIELDS}
    turn.setdefault("chart_texts", {})[index] = texts
    turn.setdefault("chart_flags", {})[index] = check_headline(texts["headline"], texts["note"])


def _on_make_svg(turn_id, index):
    turn = _find_turn(turn_id)
    if turn is None or index >= len(turn.get("outputs") or []):
        return
    spec = turn["outputs"][index][1][0]
    texts = (turn.get("chart_texts") or {}).get(index, {})
    style = _chart_style()
    turn.setdefault("chart_svg", {})[index] = (_chart_key(style, texts), render_chart(spec, style, texts, fmt="svg"))


def _render_chart(turn: dict, index: int, spec):
    turn_id = turn.get("id", 0)
    if spec.why:
        st.caption(f"**Gemini's reason for this chart:** {spec.why}")
    for level, text in spec.notes:
        if level == "fallback":
            st.info(text)
    if spec.kind == "table":
        st.dataframe(spec.data)
        return
    st.image(_chart_png(turn, index, spec), width=CHART_DISPLAY_WIDTH.get(_chart_style().use, 680))
    tips = [text for level, text in spec.notes if level != "fallback"]
    if tips:
        st.caption("\n".join(f"- {tip}" for tip in tips))
    for flag in (turn.get("chart_flags") or {}).get(index, []):
        st.warning(flag)

    texts = (turn.get("chart_texts") or {}).get(index, {})
    defaults = {"x_title": spec.x_title, "y_title": spec.y_title}      # axis titles start from the column names
    with st.expander("✏️ Add your headline, source and note"):
        st.caption("You own the words on the chart: the app never writes them. They stay on your computer.")
        with st.form(f"chart_form_{index}_{turn_id}", border=False):
            for field, label, help_text in CHART_FIELDS:
                if field in ("x_title", "y_title") and field not in spec.axis_fields:
                    continue
                key = _chart_field_key(field, index, turn_id)
                st.session_state.setdefault(key, texts.get(field, defaults.get(field, "")))
                st.text_input(label, key=key, help=help_text)
            st.form_submit_button("Update the chart", on_click=_on_chart_text, args=(turn_id, index))

    stem = file_stem(spec, texts, f"chart-{turn_id}-{index + 1}")
    png_col, svg_col = st.columns(2)
    png_col.download_button("Download PNG (300 dpi)", data=_chart_png(turn, index, spec), file_name=f"{stem}.png",
                            mime="image/png", key=f"chart_png_{index}_{turn_id}")
    svg = (turn.get("chart_svg") or {}).get(index)
    if svg is not None and svg[0] == _chart_key(_chart_style(), texts):
        svg_col.download_button("Download SVG", data=svg[1], file_name=f"{stem}.svg", mime="image/svg+xml",
                                key=f"chart_svg_{index}_{turn_id}")
    else:
        svg_col.button("Make an SVG copy", key=f"chart_make_svg_{index}_{turn_id}", on_click=_on_make_svg,
                       args=(turn_id, index),
                       help="For the Outreach and design teams to edit in Illustrator, Figma or Inkscape. "
                            "The text stays editable.")


def _render_code(turn: dict, df: pd.DataFrame):
    turn_id = turn.get("id", 0)
    if not turn["code"]:
        st.caption("No code was produced for this question.")
        return
    code_key = f"code_{turn_id}"
    # The starting text goes into Session State rather than value=..., because the box's value is
    # also written back on every run (_keep_inputs_of_hidden_tabs); doing both makes Streamlit warn.
    st.session_state.setdefault(code_key, turn["code"])
    st.text_area("You can edit this code and run your version.", key=code_key,
                 height=min(420, 60 + 22 * turn["code"].count("\n")))
    run_col, explain_col = st.columns(2)
    run_col.button(
        "Run my version", key=f"run_{turn_id}", on_click=_on_run_edited, args=(turn_id,),
        help="Runs the code above on your computer. Nothing is sent to Gemini.",
    )
    explain_col.button(
        "Explain this code", key=f"explain_{turn_id}", on_click=_on_explain, args=(turn_id,),
        help="Sends only the code above and the column names/types to Gemini.",
    )
    if st.checkbox("Show what 'Explain' will send", key=f"show_explain_{turn_id}"):
        request = prepare_request(df, "", "", turn.get("anonymise", False))
        current = st.session_state.get(code_key, turn["code"])
        st.code(build_explain_prompt(request["schema"], current), language="text")
        st.caption("If you typed any data values into the code, they would be sent too.")


def _render_sent(turn: dict):
    if turn["sent_prompts"]:
        for number, prompt in enumerate(turn["sent_prompts"], start=1):
            st.markdown(f"**Request {number}**")
            st.code(prompt, language="text")
    else:
        st.caption("Nothing was sent to Gemini for this answer.")


def render_turn(turn: dict, df: pd.DataFrame):
    """One answer as a card: Result | Code | Sent to Gemini, with any explanation below."""
    turn_id = turn.get("id", 0)
    with st.container(border=True):
        label = " · *your edited version, run locally*" if turn.get("edited") else ""
        st.markdown(f"**Q: {turn['question']}**{label}")
        sent = len(turn["sent_prompts"])
        result_tab, code_tab, sent_tab = _tabs(["Result", "Code", f"Sent to Gemini ({sent})"], f"card_tabs_{turn_id}")

        with result_tab:
            if _visible(result_tab):
                _render_result(turn)
        with code_tab:
            if _visible(code_tab):
                _render_code(turn, df)
        with sent_tab:
            if _visible(sent_tab):
                _render_sent(turn)

        if turn.get("explanation"):
            st.markdown("**Explanation**")
            st.markdown(turn["explanation"])
        if turn.get("explanation_error"):
            st.error(turn["explanation_error"])


DATA_HANDLING_NOTE = """
**Sent to Google's Gemini:** your question, the context instructions in the sidebar, the column names
and types, and the number of rows. For follow-up questions, your last few questions and the code that
answered them. For *Explain this code*, the code itself.

**Never sent:** the rows or values in your file, results, tables, charts, the headline, source and note you
write for a chart, or the file itself. Error
messages are cleaned of data values before the model sees them, and the generated code cannot reach
the internet.

**Your part:** anything you type into a question, the context box or edited code *is* sent, so avoid
typing confidential values there. Turn on *Anonymise column names* if the column names themselves are
sensitive.

**Google's terms:** with a free-tier API key, Google may use what is sent to improve its products and
human reviewers may read it. With a paid (billed) key, Google says it does not use prompts to improve
its products. See the [Gemini API Additional Terms of Service](https://ai.google.dev/gemini-api/terms).
"""


MEMORY_TIPS = """
This page runs a full Python environment inside your browser, which is what keeps your data on your
computer. The cost is memory: expect the tab to use roughly **0.6 to 1 GB or more**, depending on your
browser. Most of that is the Python setup itself, not your file.

- **Use one tab.** Every open copy of this page runs its own Python and needs the same amount of memory again.
- **Reload the page after a heavy session.** Memory used by large files and many analyses is only released
  when the tab is reloaded or closed. Reloading clears your questions and data, so download the session
  (the *③ Export* tab) or any results you need first.
- **Very large files** (hundreds of MB) may be more than the browser allows. Filter or sample the data
  before uploading if the page becomes slow or the tab crashes.
- **On a computer with little memory** (8 GB or less), close other heavy tabs and programs while you work.
"""


STATUS_ICONS = {audit.CAN: "✅ Ready", audit.CAREFUL: "⚠️ Use with caution", audit.CANNOT: "⛔ Not possible"}


def _get_audit(df: pd.DataFrame):
    report = st.session_state.get("audit")
    if report is None and not st.session_state.get("audit_error"):
        try:
            report = audit.run_audit(df, st.session_state.get("file_name") or "data")
            st.session_state["audit"] = report
            st.session_state["audit_html"] = audit.build_html_report(report)
        except Exception as exc:  # the audit must never stop people using their data
            st.session_state["audit_error"] = f"{type(exc).__name__}: {exc}"
    return report


def render_audit_banner(report):
    counts = (f"{len(report.by_severity(audit.CRITICAL))} critical issue(s), "
              f"{len(report.by_severity(audit.WARNING))} warning(s), "
              f"{len(report.by_severity(audit.OBSERVATION))} observation(s)")
    text = (f"**Data quality: {report.assessment}. Analysis readiness: {report.readiness}.** {counts}. "
            "See the **① Data & quality** tab for what you can and cannot analyse.")
    {audit.READY: st.success, audit.CAUTION: st.warning, audit.NOT_READY: st.error}[report.readiness](text)


def _findings_section(title, findings, empty_text):
    st.markdown(f"**{title}**")
    if findings:
        st.table(audit.findings_table(findings).set_index("Issue"))   # st.table wraps long text
    else:
        st.caption(empty_text)


def render_audit(report):
    crit, warn, obs = (report.by_severity(s) for s in (audit.CRITICAL, audit.WARNING, audit.OBSERVATION))
    st.markdown(f"**Overall assessment:** {report.assessment} &nbsp;·&nbsp; "
                f"**Analysis readiness:** {report.readiness}")
    cols = st.columns(3)
    cols[0].metric("Critical issues", len(crit))
    cols[1].metric("Warnings", len(warn))
    cols[2].metric("Observations", len(obs))
    st.markdown(report.assessment_text)

    with st.container(key="tour-capabilities"):
        st.subheader("What you can analyse")
        caps = pd.DataFrame(report.capabilities)
        caps["Status"] = caps["Status"].map(STATUS_ICONS)
        st.table(caps[["Analysis", "Status", "Why"]].set_index("Analysis"))   # st.table wraps long text

    st.subheader("Findings")
    _findings_section("Critical issues", crit, "No critical issues identified by the checks performed.")
    _findings_section("Warnings", warn, "No warnings identified by the checks performed.")
    st.markdown("**Observations**")
    if obs:
        st.markdown("\n".join(
            f"- **{f.issue}**" + (f" ({f.affected:,} records, {f.pct:.1f}%)" if f.pct is not None else "")
            + f". {f.impact}" for f in obs))
    else:
        st.caption("No observations.")

    st.subheader("Recommended actions")
    if report.actions:
        st.markdown("\n".join(f"{i}. {a}" for i, a in enumerate(report.actions, start=1)))
    else:
        st.caption("No actions needed.")

    st.subheader(f"Analysis readiness: {report.readiness}")
    st.markdown(report.readiness_text)

    st.subheader("Column-level summary")
    st.dataframe(audit.column_summary(report), hide_index=True)
    if not report.stats.empty:
        st.subheader("Statistics for numeric columns")
        st.dataframe(report.stats, hide_index=True)
    if not report.correlations.empty:
        st.markdown("**Strongest correlations**")
        st.dataframe(report.correlations, hide_index=True)

    st.subheader("How these checks work")
    st.markdown(
        "Automatic, general-purpose checks run in your browser on the full dataset, following the "
        "*data-quality-auditor-v2* method. Nothing here is sent to Gemini. Unusual values are flagged for "
        "checking, not declared wrong, and values from columns that may hold personal information are never "
        "shown. The checks do not know your subject area, so confirm findings with someone who knows the data.")
    st.markdown("Checks performed:\n" + "\n".join(f"- {c}" for c in report.checks_done)
                + ("\n\nChecks skipped:\n" + "\n".join(f"- {c}" for c in report.checks_skipped)
                   if report.checks_skipped else ""))
    stem = (report.file_name.rsplit(".", 1)[0] or "data").replace(" ", "_")
    st.download_button(
        "Download the audit report (HTML)", data=(st.session_state.get("audit_html")
                                                  or audit.build_html_report(report)).encode("utf-8"),
        file_name=f"data_quality_report_{stem}.html", mime="text/html",
    )
    st.caption("Open the report in any browser. To get a PDF, use the browser's Print → Save as PDF.")


APP_VERSION = "1.0 · November 2026"
GUIDE_URL = "guide.html"
REPO_URL = "https://github.com/mrafiuddin50/csvanalyser_gemini_ceew"
MAIN_TABS = ["① Data & quality", "② Ask questions", "③ Export"]


def _tabs(labels, key):
    """Tabs that remember the selected tab across reruns (Streamlit 1.65+, as on the live site).

    Without a key, Streamlit can jump back to the first tab whenever the page reruns, for example
    after clicking a button inside a tab. Older Streamlit versions lack the option; fall back.
    """
    try:
        return st.tabs(labels, key=key, on_change="rerun")
    except TypeError:
        return st.tabs(labels)


def _select_tab(key, label):
    st.session_state[key] = label


def _visible(tab) -> bool:
    """Whether to build a tab's contents.

    Tabs that remember their selection (see _tabs) say whether they are open, so only the
    visible tab is built and sent to the browser; each click then has far less to redraw.
    Older Streamlit versions do not say (None), so every tab is built as before.
    """
    is_open = getattr(tab, "open", None)
    return is_open is None or bool(is_open)


# Inputs inside tabs, and the tab selections. When a tab is not drawn, Streamlit forgets the values
# of the inputs in it, so a half-typed question or edited code would be lost on switching tabs.
# Writing each value back at the start of every run keeps it. (The tab selections are written back
# too, so a tab chosen by the app, such as Ask questions after clicking an example, also sticks in
# Streamlit's test harness, which does not report tab selections.)
KEPT_INPUTS = ("query", "main_tabs", "data_tabs", "code_", "card_tabs_", "chartfield_")
PER_ANSWER_INPUTS = ("code_", "card_tabs_", "chartfield_")    # keys end in the answer's id


def _keep_inputs_of_hidden_tabs():
    current = {str(t.get("id")) for t in st.session_state.get("history", [])}
    for key in list(st.session_state.keys()):
        if not key.startswith(KEPT_INPUTS):
            continue
        if key.startswith(PER_ANSWER_INPUTS) and key.rsplit("_", 1)[-1] not in current:
            del st.session_state[key]          # belongs to an answer that has been cleared
        else:
            st.session_state[key] = st.session_state[key]


# ---------------------------------------------------------------------------
# Tour
#
# The guided tour runs in the browser (tour.js, loaded by index.html). It moves
# to each part of the page and highlights it. Each part is wrapped in a container
# with a fixed key ("tour-..."), which Streamlit turns into a CSS class
# "st-key-tour-..."; tour.js finds sections by those classes. Keep the keys and
# tour.js in step (tests/test_tour_anchors.py checks this).
# ---------------------------------------------------------------------------

def tour_button(container, primary: bool = False):
    """A plain HTML button: tour.js starts the tour on click, without a Streamlit rerun."""
    style = "tda-tour-start tda-primary" if primary else "tda-tour-start"
    container.markdown(f'<button type="button" class="{style}">🧭 Take the tour</button>', unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Page sections
# ---------------------------------------------------------------------------

def render_header():
    with st.container(key="tour-header"):
        st.title("Tabular Data Analyser")
        st.markdown("Ask questions of your data in plain English. **Your data never leaves your computer.**")
        tour_col, help_area, _ = st.columns([1.3, 2.6, 3])
        tour_button(tour_col, primary=True)
        with help_area.container(key="tour-help"):
            guide_col, help_col = st.columns([1.4, 1])
            guide_col.link_button("📘 Printable guide", GUIDE_URL)
            with help_col.popover("Help"):
                st.markdown("#### How your data is handled")
                st.markdown(DATA_HANDLING_NOTE)
                st.markdown("#### Memory and performance tips")
                st.markdown(MEMORY_TIPS)


def render_welcome():
    with st.container(border=True, key="tour-welcome"):
        st.subheader("Get started in three steps")
        steps = [
            ("1 · Load data", "Upload a CSV or Excel file in the sidebar, or try the synthetic sample."),
            ("2 · Check quality", "An automatic audit shows whether the data is ready and what you can analyse."),
            ("3 · Ask questions", "Connect your Gemini key and ask in plain English. Only column names and "
                                  "types are sent; the analysis runs on your computer."),
        ]
        for column, (title, text) in zip(st.columns(3), steps):
            column.markdown(f"**{title}**")
            column.caption(text)
        st.button("Try the synthetic sample", key="sample_welcome", type="primary", on_click=_load_sample)
        st.caption("To use your own file, upload it in the sidebar under **2 · Data**. New here? Use "
                   "**🧭 Take the tour** at the top of the page.")


def render_sidebar(df):
    """Compact sidebar: 1 Connect, 2 Data, 3 Settings. Returns (api key, model, anonymise, context)."""
    with st.sidebar.container(key="tour-connect"):
        connect_heading = st.empty()
        st.text_input(
            "Gemini API key", type="password", key="api_key", on_change=_on_key_change,
            help="Get a free key from Google AI Studio. Press Enter to check it.",
        )
        run_pending_task("sidebar")
        key_status = st.session_state.get("key_status")
        if key_status:
            (st.success if key_status[0] == "ok" else st.error)(key_status[1])
        connect_heading.subheader(("✅ " if key_status and key_status[0] == "ok" else "") + "1 · Connect")
        model_options = st.session_state.get("models") or FALLBACK_MODELS
        if st.session_state.get("model") not in model_options:
            st.session_state["model"] = model_options[0]
        model = st.selectbox("Model", model_options, key="model")
        if not st.session_state.get("models"):
            st.caption("Default list. Enter a working key to see the models it can use.")

    with st.sidebar.container(key="tour-data"):
        st.subheader(("✅ " if df is not None else "") + "2 · Data")
        st.file_uploader("Upload a CSV or Excel file", type=UPLOAD_TYPES, key="uploaded_csv", on_change=_on_upload)
        st.button(
            "Try the synthetic sample", key="sample_sidebar", on_click=_load_sample,
            help="Made-up daily air-quality readings for five fictional stations in 2024, "
                 "with some missing values, messy names and duplicates to practise on.",
        )
        if df is not None:
            if st.session_state.get("is_sample"):
                st.caption("Using the synthetic sample dataset.")
                st.download_button("Download the sample as CSV", data=df.to_csv(index=False).encode("utf-8"),
                                   file_name=SAMPLE_FILE_NAME, mime="text/csv")
            else:
                description = st.session_state.get("file_description")
                st.caption(f"Using **{st.session_state.get('file_name') or 'your file'}**"
                           + (f", read as {description}." if description else "."))

    with st.sidebar.container(key="tour-settings"):
        with st.expander("3 · Settings"):
            anonymise = st.checkbox(
                "Anonymise column names", key="anonymise", on_change=_clear_history,
                help="Sends col_1, col_2, ... instead of the real column names. Use this when the column names "
                     "themselves are sensitive (for example, site names or respondent IDs as headers). "
                     "Answers may be less accurate because the model cannot tell what each column means.",
            )
            context = st.text_area("Instructions sent with every question", value=DEFAULT_CONTEXT, height=130)
            st.button("Clear question history", key="clear_sidebar", on_click=_clear_history)
    with st.sidebar.container(key="tour-charts"):
        with st.expander("4 · Chart style"):
            st.radio("Charts are for", list(USES), format_func=USES.get, key="chart_use",
                     help="Sets the size, background, font size and labels, following the CEEW data viz skill. "
                          "Research report: A4 width, white, legend below. Presentation: half a 16:9 slide. "
                          "Outreach: 1080 px on cream.")
            if st.session_state.get("chart_use") == "outreach":
                st.radio("Outreach size", list(SHAPES), format_func=SHAPES.get, key="chart_shape")
            st.selectbox("Colours", list(PAIR_LABELS), format_func=PAIR_LABELS.get, key="chart_pair",
                         help="Each chart uses two of CEEW's three primary colours, with grey as the third colour. "
                              "No pair is the default: vary it across charts.")
            st.caption("Charts are redrawn when you change these. Nothing here is sent to Gemini.")
    user_api_key = (st.session_state.get("api_key") or "").strip()
    return user_api_key, model, anonymise, context


def render_data_section(df: pd.DataFrame):
    summary, columns_table = st.session_state.get("profile") or profile_dataset(df)
    report = _get_audit(df)
    audit_tab, preview_tab, profile_tab = _tabs(["Data quality audit", "Preview", "Column details"], "data_tabs")
    with audit_tab:
        if not _visible(audit_tab):
            pass
        elif report is not None:
            render_audit(report)
        else:
            st.warning("The data quality audit could not be completed for this file. You can still analyse "
                       f"the data. Details: {st.session_state.get('audit_error')}")
    with preview_tab:
        if _visible(preview_tab):
            st.caption("The first 20 rows. This view stays in your browser.")
            st.dataframe(df.head(20))
    with profile_tab:
        if not _visible(profile_tab):
            return
        cols = st.columns(4)
        for col, (name, value) in zip(cols, summary.items()):
            col.metric(name.capitalize(), f"{value:,}")
        st.dataframe(columns_table, hide_index=True)
        st.caption("Calculated in your browser. None of this is sent to Gemini.")


def _use_example(question: str):
    st.session_state["query"] = question
    _select_tab("main_tabs", MAIN_TABS[1])


def render_ask(df, api_key, model, anonymise, context):
    report = st.session_state.get("audit")
    examples = suggest_questions(report.column_info) if report else []
    if examples:
        with st.container(key="tour-examples"):
            st.markdown("**Example questions** (click one to use it, then press Analyse):")
            columns = st.columns(2)
            for index, question in enumerate(examples):
                columns[index % 2].button(question, key=f"example_{index}", on_click=_use_example,
                                          args=(question,), help=question)

    with st.container(key="tour-question"):
        query = st.text_input("Your question", key="query", placeholder="For example: average of a column by group")
        if query:
            preview = prepare_request(df, query, context, anonymise)
            past = past_turns_for_prompt(st.session_state["history"], anonymise)
            with st.expander("What will be sent to Gemini"):
                st.caption("Only this text leaves your browser. No rows or values from your file are included.")
                st.code(build_prompt(preview["schema"], preview["question"], preview["context"], past),
                        language="text")
        analyse_clicked = st.button("Analyse", type="primary")

    if analyse_clicked and query:
        if not api_key:
            st.error("Please enter a Gemini API key in the sidebar (1 · Connect) to run analysis.")
            _log("Analysis not started: no API key entered", ok=False)
        else:
            start = time.perf_counter()
            with st.status(f"Analysing your question with {model}…", expanded=True) as box:
                turn = analyse(
                    query, df, context, st.session_state["history"], anonymise,
                    llm=lambda prompt: generate_python_code(prompt, model, api_key),
                    progress=box.write, style=_chart_style(),
                )
                seconds = time.perf_counter() - start
                retries = f" after {turn['attempts']} attempts" if turn["attempts"] > 1 else ""
                if turn["error"]:
                    summary = f"Could not answer the question{retries}"
                    box.update(label=f"{summary} ({seconds:.1f} s)", state="error", expanded=True)
                else:
                    summary = f"Answered the question{retries}"
                    box.update(label=f"{summary} ({seconds:.1f} s)", state="complete", expanded=False)
            _add_turn(turn)
            _log(summary, ok=not turn["error"], seconds=seconds)

    run_pending_task("answers")
    if st.session_state["history"]:
        with st.container(key="tour-answers"):
            st.markdown(f"**Answers** (newest first, {len(st.session_state['history'])})")
            for turn in reversed(st.session_state["history"]):
                render_turn(turn, df)


def render_export(df: pd.DataFrame):
    with st.container(key="tour-export-session"):
        st.subheader("Download your analysis as a notebook or script")
        turns_to_export = exportable_turns(st.session_state["history"])
        if turns_to_export:
            st.caption(f"{len(turns_to_export)} answered question(s) as code. Results are not included, so the file "
                       "can be shared without your data. Run it next to your data file to reproduce the results.")
            reader_code = st.session_state.get("reader_code") or "pd.read_csv('your_file.csv')"
            notebook = build_notebook(st.session_state["history"], reader_code, df.columns)
            script = build_script(st.session_state["history"], reader_code, df.columns)
        else:
            st.caption("These buttons become active after your first answer in **② Ask questions**. The files "
                       "contain your questions and code only, not your data or results.")
            notebook = script = ""
        notebook_col, script_col, _ = st.columns([1.3, 1.1, 2])
        notebook_col.download_button(
            "Jupyter notebook (.ipynb)", data=notebook.encode("utf-8"), disabled=not turns_to_export,
            file_name="analysis_session.ipynb", mime="application/x-ipynb+json", key="export_notebook",
        )
        script_col.download_button(
            "Python script (.py)", data=script.encode("utf-8"), disabled=not turns_to_export,
            file_name="analysis_session.py", mime="text/x-python", key="export_script",
        )
        if turns_to_export and uses_ceew_charts(turns_to_export):
            st.download_button(f"CEEW chart functions ({CHARTS_FILE_NAME})", data=charts_module_source().encode("utf-8"),
                               file_name=CHARTS_FILE_NAME, mime="text/x-python", key="export_charts")
            st.caption(f"Your answers draw CEEW charts. Save **{CHARTS_FILE_NAME}** next to the notebook or script. "
                       "For the CEEW font, put the Familjen Grotesk .ttf files "
                       "([free, from GitHub](https://github.com/Familjen-Sthlm/Familjen-Grotesk/tree/master/fonts/ttf)) "
                       "in a folder called *fonts* next to it.")
    with st.container(key="tour-export-report"):
        st.subheader("Data quality report")
        html = st.session_state.get("audit_html")
        if html:
            stem = (st.session_state.get("file_name") or "data").rsplit(".", 1)[0].replace(" ", "_")
            st.download_button("Download the audit report (HTML)", data=html.encode("utf-8"),
                               file_name=f"data_quality_report_{stem}.html", mime="text/html", key="audit_export")
            st.caption("Open it in any browser; use Print → Save as PDF for a PDF.")
        else:
            st.caption("No audit report is available for this file.")


def render_footer():
    st.divider()
    st.caption(
        f"Tabular Data Analyser {APP_VERSION} · [Printable guide]({GUIDE_URL}) · "
        f"[Source code]({REPO_URL}) · Inspired by "
        "[VayuChat-WebLLM](https://github.com/nipunbatra/vayuchat-webllm) (IIT Gandhinagar Sustainability Lab) · "
        "Python runs in your browser via stlite."
    )


LOGO_PATH = "assets/ceew-logo.png"


def render_logo():
    """CEEW logo in the top-left corner (stays visible on phones when the sidebar is collapsed)."""
    if not os.path.exists(LOGO_PATH):
        return
    try:
        st.logo(LOGO_PATH, size="large", link="https://www.ceew.in")
    except TypeError:          # older Streamlit without the size option
        st.logo(LOGO_PATH, link="https://www.ceew.in")


def main():
    st.set_page_config(layout="wide", page_title="Tabular Data Analyser")
    render_logo()
    st.session_state.setdefault("history", [])
    st.session_state.setdefault("df", None)
    _keep_inputs_of_hidden_tabs()

    render_header()
    # Filled in at the end of each run, so it always shows the latest action.
    with st.container(key="tour-status"):
        status_slot = st.empty()
    run_pending_task("main")

    df = st.session_state.get("df")
    api_key, model, anonymise, context = render_sidebar(df)

    if st.session_state.get("load_error"):
        st.error(st.session_state["load_error"])
    if df is None:
        render_welcome()
        render_footer()
        _finish(status_slot)
        return

    if st.session_state.get("is_sample"):
        st.warning("You are using **synthetic sample data**. It is made up for practice and is not real measurements.")
    report = _get_audit(df)
    if report is not None:
        with st.container(key="tour-banner"):
            render_audit_banner(report)

    with st.container(key="tour-tabs"):
        data_tab, ask_tab, export_tab = _tabs(MAIN_TABS, "main_tabs")
        with data_tab:
            if _visible(data_tab):
                render_data_section(df)
        with ask_tab:
            if _visible(ask_tab):
                render_ask(df, api_key, model, anonymise, context)
        with export_tab:
            if _visible(export_tab):
                render_export(df)

    render_footer()
    _finish(status_slot)


def _finish(status_slot):
    """End of every run: status line, activity log, and drop any task that could not run."""
    st.session_state["pending"] = None
    render_status_line(status_slot)
    render_activity_log()


if __name__ == "__main__":
    main()
