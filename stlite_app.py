"""Streamlit interface. Runs in the browser through stlite (see index.html).

The logic lives in the csvanalyser package; this file only draws the page.
"""
import pandas as pd
import streamlit as st

from csvanalyser.analysis import analyse, past_turns_for_prompt, prepare_request, run_edited_code
from csvanalyser.export import build_notebook, build_script, exportable_turns
from csvanalyser.gemini import FALLBACK_MODELS, GeminiError, call_gemini_text, generate_python_code, list_models
from csvanalyser.loading import UPLOAD_TYPES, load_table
from csvanalyser.privacy import restore_names
from csvanalyser.profiling import profile_dataset
from csvanalyser.prompts import build_explain_prompt, build_prompt
from csvanalyser.sample import SAMPLE_FILE_NAME, make_sample_dataset


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
    st.session_state["profile"] = profile_dataset(df) if df is not None else None


def _on_upload():
    uploaded = st.session_state.get("uploaded_csv")
    st.session_state["load_error"] = None
    if uploaded is None:
        _set_dataset(None)
        return
    try:
        df, description, reader_code = load_table(uploaded.name, uploaded.getvalue())
    except Exception as exc:
        _set_dataset(None)
        st.session_state["load_error"] = (
            f"Could not read {uploaded.name}: {exc}. "
            "Check that it is a CSV, TSV or Excel (.xlsx) file with a header row."
        )
        return
    if df.empty and len(df.columns) == 0:
        _set_dataset(None)
        st.session_state["load_error"] = f"{uploaded.name} has no columns."
        return
    _set_dataset(df, uploaded.name, description, reader_code)


def _load_sample():
    st.session_state["load_error"] = None
    _set_dataset(make_sample_dataset(), SAMPLE_FILE_NAME, "synthetic sample data (not real measurements)",
                 f"pd.read_csv({SAMPLE_FILE_NAME!r})", is_sample=True)


def _add_turn(turn: dict):
    st.session_state.setdefault("next_turn_id", 1)
    turn["id"] = st.session_state["next_turn_id"]
    st.session_state["next_turn_id"] += 1
    st.session_state["history"].append(turn)


def _find_turn(turn_id):
    return next((t for t in st.session_state.get("history", []) if t.get("id") == turn_id), None)


def _on_run_edited(turn_id):
    turn = _find_turn(turn_id)
    df = st.session_state.get("df")
    if turn is None or df is None:
        return
    code = st.session_state.get(f"code_{turn_id}", turn["code"])
    _add_turn(run_edited_code(turn, code, df))


def _on_explain(turn_id):
    turn = _find_turn(turn_id)
    df = st.session_state.get("df")
    if turn is None or df is None:
        return
    turn["explanation"], turn["explanation_error"] = None, None
    api_key = (st.session_state.get("api_key") or "").strip()
    if not api_key:
        turn["explanation_error"] = "Enter a Gemini API key in the sidebar to get an explanation."
        return
    request = prepare_request(df, "", "", turn.get("anonymise", False))
    code = st.session_state.get(f"code_{turn_id}", turn["code"])
    prompt = build_explain_prompt(request["schema"], code)
    turn["sent_prompts"].append(prompt)
    model = st.session_state.get("model") or FALLBACK_MODELS[0]
    try:
        turn["explanation"] = restore_names(call_gemini_text(prompt, model, api_key), request["reverse"])
    except GeminiError as exc:
        turn["explanation_error"] = str(exc)


def _on_key_change():
    """Checks a newly entered key by listing the models it can use."""
    st.session_state["models"] = None
    st.session_state["key_status"] = None
    key = (st.session_state.get("api_key") or "").strip()
    if not key:
        return
    try:
        models = list_models(key)
    except GeminiError as exc:
        st.session_state["key_status"] = ("error", str(exc))
        return
    if not models:
        st.session_state["key_status"] = ("error", "The key works, but no text models are available to it.")
        return
    st.session_state["models"] = models
    st.session_state["key_status"] = ("ok", f"Key works. {len(models)} models available.")


def _clear_history():
    st.session_state["history"] = []


def _show_value(value):
    if isinstance(value, (pd.DataFrame, pd.Series)):
        st.dataframe(value)
    elif value is not None:
        st.write(value)


def render_turn(turn: dict, df: pd.DataFrame):
    turn_id = turn.get("id", 0)
    label = " *(your edited version, run locally)*" if turn.get("edited") else ""
    st.markdown(f"**Q:** {turn['question']}{label}")

    for name, args, kwargs in turn["outputs"]:
        try:
            if name == "image":
                st.image(args[0])
            else:
                getattr(st, name)(*args, **kwargs)
        except Exception as exc:
            st.warning(f"Could not display an output (st.{name}): {exc}")

    if turn["error"]:
        st.error(f"Error during execution: {turn['error']}")
    else:
        _show_value(turn["result"])
        if isinstance(turn["result"], (pd.DataFrame, pd.Series)):
            st.download_button(
                "Download result as CSV", data=turn["result"].to_csv().encode("utf-8"),
                file_name=f"result_{turn_id}.csv", mime="text/csv", key=f"download_{turn_id}",
            )
        if turn["attempts"] > 1:
            st.caption(f"The first attempt failed; fixed automatically after {turn['attempts']} attempts.")

    if turn["code"]:
        with st.expander("Python code: view, edit, run or explain"):
            code_key = f"code_{turn_id}"
            st.text_area(
                "You can edit this code and run your version.", value=turn["code"], key=code_key,
                height=min(420, 60 + 22 * turn["code"].count("\n")),
            )
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

    if turn.get("explanation"):
        st.markdown("**Explanation**")
        st.markdown(turn["explanation"])
    if turn.get("explanation_error"):
        st.error(turn["explanation_error"])

    if turn["sent_prompts"]:
        with st.expander(f"What was sent to Gemini ({len(turn['sent_prompts'])} request(s))"):
            for number, prompt in enumerate(turn["sent_prompts"], start=1):
                st.markdown(f"**Request {number}**")
                st.code(prompt, language="text")
    else:
        st.caption("Nothing was sent to Gemini for this answer.")


DATA_HANDLING_NOTE = """
**Sent to Google's Gemini:** your question, the context instructions in the sidebar, the column names
and types, and the number of rows. For follow-up questions, your last few questions and the code that
answered them. For *Explain this code*, the code itself.

**Never sent:** the rows or values in your file, results, tables, charts, or the file itself. Error
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
  (sidebar, *Export session*) or any results you need first.
- **Very large files** (hundreds of MB) may be more than the browser allows. Filter or sample the data
  before uploading if the page becomes slow or the tab crashes.
- **On a computer with little memory** (8 GB or less), close other heavy tabs and programs while you work.
"""


def render_data_tabs(df: pd.DataFrame):
    summary, columns_table = st.session_state.get("profile") or profile_dataset(df)
    with st.expander("Your data (stays in your browser)", expanded=False):
        preview_tab, profile_tab = st.tabs(["Preview", "Data profile"])
        with preview_tab:
            st.dataframe(df.head(20))
        with profile_tab:
            cols = st.columns(4)
            for col, (name, value) in zip(cols, summary.items()):
                col.metric(name.capitalize(), f"{value:,}")
            if summary["duplicate rows"]:
                st.warning(f"{summary['duplicate rows']:,} rows are exact duplicates of another row.")
            notes = columns_table[columns_table["note"] != ""]
            for _, row in notes.iterrows():
                st.info(f"Column {row['column']!r}: {row['note'].lower()}.")
            st.dataframe(columns_table, hide_index=True)
            st.caption("Calculated in your browser. None of this is sent to Gemini.")


def main():
    st.set_page_config(layout="wide", page_title="Tabular Data Analyser")
    st.markdown(
        """
        <div style='background-color: #f4f6f9; padding: 8px; border-radius: 6px; text-align: center; margin-bottom: 10px; border: 1px solid #e0e4e8;'>
            <span style='font-size: 0.9em;'>✨ Inspired by <a href='https://github.com/nipunbatra/vayuchat-webllm' target='_blank' style='text-decoration: none; font-weight: bold; color: #0366d6;'>VayuChat-WebLLM</a></span>
        </div>
        """,
        unsafe_allow_html=True
    )
    st.title("Tabular Data Analyser")
    st.markdown(
        "Upload a CSV or Excel file and ask questions in plain English. **Your data stays in your browser.** "
        "Only your question, your context instructions and the column names and types are sent to "
        "Google's Gemini, which writes Python code that then runs here on your computer."
    )
    with st.expander("How your data is handled"):
        st.markdown(DATA_HANDLING_NOTE)
    with st.expander("Memory and performance tips"):
        st.markdown(MEMORY_TIPS)

    st.session_state.setdefault("history", [])
    st.session_state.setdefault("df", None)

    st.sidebar.title("API Configuration")
    st.sidebar.text_input(
        "Enter Gemini API Key:", type="password", key="api_key", on_change=_on_key_change,
        help="Get a free key from Google AI Studio. Press Enter to check it.",
    )
    user_api_key = (st.session_state.get("api_key") or "").strip()
    key_status = st.session_state.get("key_status")
    if key_status:
        (st.sidebar.success if key_status[0] == "ok" else st.sidebar.error)(key_status[1])

    st.sidebar.title("Model Configuration")
    model_options = st.session_state.get("models") or FALLBACK_MODELS
    if st.session_state.get("model") not in model_options:
        st.session_state["model"] = model_options[0]
    selected_model = st.sidebar.selectbox("Select Gemini Model:", model_options, key="model")
    if not st.session_state.get("models"):
        st.sidebar.caption("Default list. Enter a working key to see the models it can use.")

    st.sidebar.title("Data Upload")
    st.sidebar.file_uploader(
        "Upload a CSV or Excel file", type=UPLOAD_TYPES, key="uploaded_csv", on_change=_on_upload,
    )
    st.sidebar.button(
        "No file? Try a synthetic sample", on_click=_load_sample,
        help="Made-up daily air-quality readings for five fictional stations in 2024, "
             "with some missing values, messy names and duplicates to practise on.",
    )

    if st.session_state.get("load_error"):
        st.error(st.session_state["load_error"])
    df = st.session_state.get("df")
    if df is None:
        st.info("Please upload a CSV or Excel file in the sidebar to begin, or try the synthetic sample.")
        return

    if st.session_state.get("is_sample"):
        st.sidebar.success("Loaded the synthetic sample dataset.")
        st.warning("You are using **synthetic sample data**. It is made up for practice and is not real measurements.")
        st.sidebar.download_button(
            "Download the sample as CSV", data=df.to_csv(index=False).encode("utf-8"),
            file_name=SAMPLE_FILE_NAME, mime="text/csv",
        )
    else:
        st.sidebar.success(f"Loaded {st.session_state.get('file_name') or 'file'} successfully!")
        if st.session_state.get("file_description"):
            st.sidebar.caption(f"Read as {st.session_state['file_description']}.")

    render_data_tabs(df)

    st.sidebar.markdown("---")
    st.sidebar.subheader("Privacy")
    anonymise = st.sidebar.checkbox(
        "Anonymise column names",
        key="anonymise",
        on_change=_clear_history,
        help="Sends col_1, col_2, ... instead of the real column names. Use this when the column names "
             "themselves are sensitive (for example, site names or respondent IDs as headers). "
             "Answers may be less accurate because the model cannot tell what each column means.",
    )

    st.sidebar.markdown("---")
    st.sidebar.subheader("System Context")
    saved_context = st.sidebar.text_area(
        "Edit persistent context instructions for the AI:",
        value=DEFAULT_CONTEXT,
        height=150
    )
    st.sidebar.button("Clear question history", on_click=_clear_history)
    # Filled at the end of the run, so a question answered in this run is included.
    export_box = st.sidebar.container()

    query = st.text_input("Ask a question about your data:")

    if query:
        preview = prepare_request(df, query, saved_context, anonymise)
        past = past_turns_for_prompt(st.session_state["history"], anonymise)
        with st.expander("What will be sent to Gemini"):
            st.caption("Only this text leaves your browser. No rows or values from your file are included.")
            st.code(build_prompt(preview["schema"], preview["question"], preview["context"], past), language="text")

    if st.button("Analyse") and query:
        if not user_api_key:
            st.error("Please enter a Gemini API Key in the sidebar to run analysis.")
        else:
            with st.spinner("Processing..."):
                turn = analyse(
                    query, df, saved_context, st.session_state["history"], anonymise,
                    llm=lambda prompt: generate_python_code(prompt, selected_model, user_api_key),
                )
            _add_turn(turn)

    for index, turn in enumerate(reversed(st.session_state["history"])):
        if index:
            st.divider()
        render_turn(turn, df)

    render_export(export_box, df)


def render_export(box, df: pd.DataFrame):
    turns_to_export = exportable_turns(st.session_state["history"])
    if not turns_to_export:
        return
    box.markdown("---")
    box.subheader("Export session")
    box.caption(
        f"{len(turns_to_export)} answered question(s) as code. Results are not included, "
        "so the file can be shared without your data."
    )
    reader_code = st.session_state.get("reader_code") or "pd.read_csv('your_file.csv')"
    box.download_button(
        "Python script (.py)", data=build_script(st.session_state["history"], reader_code, df.columns),
        file_name="analysis_session.py", mime="text/x-python",
    )
    box.download_button(
        "Jupyter notebook (.ipynb)", data=build_notebook(st.session_state["history"], reader_code, df.columns),
        file_name="analysis_session.ipynb", mime="application/x-ipynb+json",
    )


if __name__ == "__main__":
    main()
