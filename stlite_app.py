import builtins
import contextlib
import io
import json
import re
import traceback
import urllib.request

import numpy as np
import pandas as pd
import streamlit as st

# Core rule of this app: the model never sees the data. Only the question, the
# user's context text, column names/types and the row count are sent to Gemini.
# Row values, results and raw error messages stay in the browser.

MAX_ATTEMPTS = 3            # first try + up to two self-corrections
PAST_TURNS_IN_PROMPT = 3    # earlier questions + code sent for follow-ups
GENERATED_FILENAME = "<generated code>"
REDACTED = "<redacted>"

DEFAULT_CONTEXT = (
    "Return the final answer strictly as a pandas dataframe "
    "(e.g. result = summary_df). Streamlit will render it automatically."
)

# Streamlit functions the generated code may call. Calls are recorded and
# replayed so results survive reruns; layout functions are not supported.
ALLOWED_ST_CALLS = {
    "write", "markdown", "text", "caption", "title", "header", "subheader",
    "dataframe", "table", "metric", "json", "code",
    "info", "success", "warning", "error",
    "bar_chart", "line_chart", "area_chart", "scatter_chart", "map",
    "pyplot", "altair_chart", "vega_lite_chart",
}


def generate_python_code(prompt: str, model_name: str, api_key: str) -> str:
    """Calls Gemini REST API directly using standard urllib."""
    clean_key = api_key.strip()
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model_name}:generateContent?key={clean_key}"

    payload = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {"temperature": 0.0}
    }

    data = json.dumps(payload).encode('utf-8')
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})

    try:
        # In Pyodide, urlopen might not raise HTTPError for 4xx/5xx
        with urllib.request.urlopen(req) as response:
            body = response.read().decode('utf-8')
            if response.status == 200:
                result_data = json.loads(body)
                content = result_data["candidates"][0]["content"]["parts"][0]["text"]
                match = re.search(r'```python(.*?)```', content, re.DOTALL)
                if match:
                    return match.group(1).strip()
                return content.strip()
            else:
                raise Exception(f"API Error {response.status}: {body}")
    except urllib.error.HTTPError as e:
        error_body = e.read().decode('utf-8')
        raise Exception(f"HTTP Error {e.code}: {error_body}")
    except Exception as e:
        raise Exception(f"Request failed: {str(e)}")


# ---------------------------------------------------------------------------
# What the model is allowed to see
# ---------------------------------------------------------------------------

def make_aliases(columns) -> dict:
    """Maps each original column name to a neutral alias (col_1, col_2, ...)."""
    return {col: f"col_{i}" for i, col in enumerate(columns, start=1)}


def anonymise_text(text: str, aliases: dict) -> str:
    """Replaces original column names typed in free text with their aliases."""
    if not aliases or not text:
        return text
    # Longest names first so 'PM2.5 avg' is replaced before 'PM2.5'.
    for original in sorted(aliases, key=lambda c: len(str(c)), reverse=True):
        name = str(original)
        if len(name) < 2:
            continue
        pattern = r"(?<!\w)" + re.escape(name) + r"(?!\w)"
        text = re.sub(pattern, aliases[original], text, flags=re.IGNORECASE)
    return text


def describe_schema(df: pd.DataFrame) -> str:
    """Column names and types only. Never includes any cell values."""
    lines = [
        f"The dataframe `df` has {len(df)} rows and {df.shape[1]} columns.",
        "Columns (name: type):",
    ]
    for col, dtype in df.dtypes.items():
        lines.append(f"- {str(col)!r}: {dtype}")
    return "\n".join(lines)


def build_prompt(schema: str, question: str, context: str, past_turns=(), failure=None) -> str:
    """Builds the full text sent to Gemini.

    past_turns: list of (question, code) from earlier in the session.
    failure: (failed_code, sanitised_error) from the previous attempt, if any.
    """
    parts = [
        "You are a Python data analyst writing pandas code.",
        "",
        "PRIVACY: You cannot see any data values, only the column names and types below. "
        "Never guess or hard-code data values; write code that finds them in the data instead.",
        "",
        schema,
    ]

    if past_turns:
        parts += ["", "Earlier questions in this session and the code that answered them. "
                      "Each run starts fresh with only `df`, so recompute anything you need:"]
        for past_question, past_code in past_turns:
            parts += [f"Q: {past_question}", "```python", past_code, "```"]

    parts += [
        "",
        "User context:",
        context or "(none)",
        "",
        f"Question: {question}",
        "",
        "RULES:",
        "- `df` is already loaded. Do not read any files. `pd`, `np`, `plt` and `st` are already available; do not import streamlit.",
        "- YOU MUST ASSIGN YOUR FINAL OUTPUT TO A VARIABLE NAMED `result` at the end of the script (e.g. `result = filtered_df`).",
        "- When the question names a specific item (such as a place, site or category), find it in the relevant text column with "
        "`df[col].astype(str).str.contains(name, case=False, na=False, regex=False)`. If nothing matches, fall back to "
        "`import difflib; difflib.get_close_matches(name.lower(), [str(x).lower() for x in df[col].dropna().unique()], cutoff=0.3)`, "
        "map the match back to the original value, and tell the user which value was used with `st.write`.",
        "- To show extra tables or text alongside `result`, call `st.write`, `st.markdown`, `st.subheader` or `st.dataframe` directly. "
        "Only these Streamlit functions are available: " + ", ".join(f"st.{name}" for name in sorted(ALLOWED_ST_CALLS)) + ". "
        "Do not use layout functions such as st.columns, st.expander, st.tabs or st.sidebar.",
        "- For plots, use Streamlit charts (e.g. `st.bar_chart(data)`, `st.line_chart(data)`) or create a matplotlib figure with "
        "`fig, ax = plt.subplots()` and show it with `st.pyplot(fig)`. NEVER use `plt.show()`. If plotting, set `result` to a short message.",
        "- If you use `.agg()`, NEVER use the string '50%'. Use 'median' instead.",
        "- To reset the index of a dataframe, use `.reset_index()`. NEVER use `.reset()`.",
        "- PUT ALL YOUR CODE IN ONE SINGLE ```python ... ``` BLOCK. Do not write any explanations outside the code block.",
    ]

    if failure:
        failed_code, error_text = failure
        parts += [
            "",
            "Your previous attempt failed. The code was:",
            "```python",
            failed_code,
            "```",
            f"Error: {error_text}",
            "(Data values in the error message have been removed for privacy.)",
            "Fix the problem and return the complete corrected script.",
        ]

    return "\n".join(parts)


def _text_values(df: pd.DataFrame) -> set:
    """Distinct text values in the data, used to scrub error messages."""
    values = set()
    for col in df.columns:
        series = df[col]
        if (pd.api.types.is_object_dtype(series) or pd.api.types.is_string_dtype(series)
                or isinstance(series.dtype, pd.CategoricalDtype)):
            for value in series.dropna().unique():
                text = str(value).strip()
                if len(text) >= 3:
                    values.add(text)
    return values


def sanitise_error(exc: BaseException, code: str, df: pd.DataFrame) -> str:
    """Turns an exception into text that is safe to send to Gemini.

    Keeps the error type, the first line of the message and the failing line of
    the model's own code. Removes anything that could be a data value: known text
    values from the data, quoted strings and tokens containing digits, unless
    they already appear in the generated code or are column names.
    """
    allowed = {str(c) for c in df.columns}

    def is_known(token: str) -> bool:
        return token in allowed or (token != "" and token in code)

    message = str(exc).splitlines()[0] if str(exc) else ""
    message = message[:300]

    # 1. Exact data values (catches unquoted values such as dates or names).
    for value in sorted(_text_values(df), key=len, reverse=True):
        if value in message and not is_known(value):
            message = message.replace(value, REDACTED)

    # 2. Quoted strings.
    def _quoted(match):
        inner = match.group(2)
        return match.group(0) if is_known(inner) or inner == REDACTED else f"{match.group(1)}{REDACTED}{match.group(1)}"
    message = re.sub(r"(['\"])(.*?)\1", _quoted, message)

    # 3. Anything containing a digit (numbers, dates, IDs).
    def _digits(match):
        token = match.group(0)
        return token if is_known(token.strip(".,;:()[]{}")) else REDACTED
    message = re.sub(r"[^\s'\"]*\d[^\s'\"]*", _digits, message)

    text = type(exc).__name__ + (f": {message}" if message else "")

    # Point at the failing line of the generated code (the model's own text).
    frames = [f for f in traceback.extract_tb(exc.__traceback__) if f.filename == GENERATED_FILENAME]
    if frames and frames[-1].lineno:
        lineno = frames[-1].lineno
        code_lines = code.splitlines()
        if 0 < lineno <= len(code_lines):
            text += f" (at line {lineno}: `{code_lines[lineno - 1].strip()}`)"
    return text


# ---------------------------------------------------------------------------
# Running generated code locally
# ---------------------------------------------------------------------------

class NoResultError(Exception):
    pass


def restore_names(obj, reverse: dict):
    """Puts original column names back on results when headers were anonymised."""
    if not reverse:
        return obj.copy() if isinstance(obj, (pd.DataFrame, pd.Series)) else obj
    if isinstance(obj, pd.DataFrame):
        out = obj.rename(columns=reverse, index=reverse)
        out.index.names = [reverse.get(n, n) for n in out.index.names]
        return out
    if isinstance(obj, pd.Series):
        out = obj.rename(index=reverse)
        out.index.names = [reverse.get(n, n) for n in out.index.names]
        try:
            out.name = reverse.get(out.name, out.name)
        except TypeError:
            pass
        return out
    if isinstance(obj, str):
        return re.sub(r"\bcol_\d+\b", lambda m: str(reverse.get(m.group(0), m.group(0))), obj)
    return obj


def _figure_to_png(fig) -> bytes:
    import matplotlib.pyplot as plt
    buffer = io.BytesIO()
    fig.savefig(buffer, format="png", bbox_inches="tight", dpi=110)
    plt.close(fig)
    return buffer.getvalue()


def _is_figure(obj) -> bool:
    try:
        from matplotlib.figure import Figure
    except ImportError:
        return False
    return isinstance(obj, Figure)


class OutputRecorder:
    """Stands in for `st` inside generated code and records each display call."""

    def __init__(self, reverse_aliases: dict):
        self.outputs = []
        self._reverse = reverse_aliases

    def __getattr__(self, name):
        if name.startswith("_"):
            raise AttributeError(name)
        if name not in ALLOWED_ST_CALLS:
            raise AttributeError(
                f"st.{name} is not available here. Use only: "
                + ", ".join(f"st.{n}" for n in sorted(ALLOWED_ST_CALLS))
            )
        return lambda *args, **kwargs: self._record(name, args, kwargs)

    def _record(self, name, args, kwargs):
        import matplotlib.pyplot as plt
        if name == "pyplot":
            fig = args[0] if args else kwargs.get("fig") or plt.gcf()
            self.outputs.append(("image", (_figure_to_png(fig),), {}))
            return None
        figures = [a for a in args if _is_figure(a)]
        if figures:
            for fig in figures:
                self.outputs.append(("image", (_figure_to_png(fig),), {}))
            return None
        args = tuple(restore_names(a, self._reverse) for a in args)
        kwargs = {k: restore_names(v, self._reverse) for k, v in kwargs.items()}
        self.outputs.append((name, args, kwargs))
        return None


def run_generated_code(code: str, df: pd.DataFrame, reverse_aliases: dict):
    """Executes generated code in its own namespace on a copy of the data.

    Returns (result, outputs) where outputs is a list of recorded display calls.
    Raises NoResultError if the code does not define `result`.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    recorder = OutputRecorder(reverse_aliases)

    def _import(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "streamlit" or name.startswith("streamlit."):
            return recorder
        return builtins.__import__(name, globals, locals, fromlist, level)

    safe_builtins = dict(vars(builtins))
    safe_builtins["__import__"] = _import
    namespace = {
        "__builtins__": safe_builtins,
        "__name__": "__generated__",
        "df": df.copy(),
        "pd": pd,
        "np": np,
        "plt": plt,
        "st": recorder,
    }

    plt.close("all")
    printed = io.StringIO()
    try:
        compiled = compile(code, GENERATED_FILENAME, "exec")
        with contextlib.redirect_stdout(printed):
            exec(compiled, namespace)
        # Figures the code drew but never passed to st.pyplot.
        for number in plt.get_fignums():
            recorder.outputs.append(("image", (_figure_to_png(plt.figure(number)),), {}))
    finally:
        plt.close("all")

    if printed.getvalue().strip():
        recorder.outputs.append(("text", (restore_names(printed.getvalue(), reverse_aliases),), {}))

    if "result" not in namespace:
        raise NoResultError("The code ran but did not define a variable named `result`.")

    result = namespace["result"]
    if _is_figure(result):
        recorder.outputs.append(("image", (_figure_to_png(result),), {}))
        result = None
    return restore_names(result, reverse_aliases), recorder.outputs


# ---------------------------------------------------------------------------
# One question, end to end
# ---------------------------------------------------------------------------

def prepare_request(df: pd.DataFrame, question: str, context: str, anonymise: bool):
    """Works out exactly what would be sent for a question."""
    aliases = make_aliases(df.columns) if anonymise else {}
    work_df = df.rename(columns=aliases) if aliases else df
    return {
        "aliases": aliases,
        "reverse": {alias: original for original, alias in aliases.items()},
        "work_df": work_df,
        "schema": describe_schema(work_df),
        "question": anonymise_text(question, aliases),
        "context": anonymise_text(context, aliases),
    }


def past_turns_for_prompt(history: list, anonymise: bool) -> list:
    """Earlier successful questions + code. Never includes results."""
    usable = [
        (turn["sent_question"], turn["code"])
        for turn in history
        if turn.get("code") and not turn.get("error") and turn.get("anonymise") == anonymise
    ]
    return usable[-PAST_TURNS_IN_PROMPT:]


def analyse(question: str, df: pd.DataFrame, context: str, history: list, anonymise: bool, llm) -> dict:
    """Asks the model for code, runs it locally and retries with sanitised errors.

    llm: callable taking the prompt text and returning Python code.
    """
    request = prepare_request(df, question, context, anonymise)
    past = past_turns_for_prompt(history, anonymise)
    turn = {
        "question": question,
        "sent_question": request["question"],
        "anonymise": anonymise,
        "sent_prompts": [],
        "code": "",
        "attempts": 0,
        "outputs": [],
        "result": None,
        "error": None,
    }

    failure = None
    for attempt in range(1, MAX_ATTEMPTS + 1):
        prompt = build_prompt(request["schema"], request["question"], request["context"], past, failure)
        turn["sent_prompts"].append(prompt)
        turn["attempts"] = attempt
        try:
            code = llm(prompt)
        except Exception as exc:
            turn["error"] = f"Gemini request failed: {exc}"
            return turn
        turn["code"] = code
        try:
            result, outputs = run_generated_code(code, request["work_df"], request["reverse"])
        except Exception as exc:
            failure = (code, sanitise_error(exc, code, request["work_df"]))
            # Full error is shown only to the user, never sent to the model.
            turn["error"] = f"{type(exc).__name__}: {restore_names(str(exc), request['reverse'])}"
            continue
        turn["result"], turn["outputs"], turn["error"] = result, outputs, None
        return turn
    return turn


# ---------------------------------------------------------------------------
# User interface
# ---------------------------------------------------------------------------

def _on_upload():
    uploaded = st.session_state.get("uploaded_csv")
    st.session_state["history"] = []
    st.session_state["load_error"] = None
    if uploaded is None:
        st.session_state["df"] = None
        st.session_state["file_name"] = None
        return
    try:
        st.session_state["df"] = pd.read_csv(uploaded)
        st.session_state["file_name"] = uploaded.name
    except Exception as exc:
        st.session_state["df"] = None
        st.session_state["load_error"] = f"Could not read {uploaded.name}: {exc}"


def _clear_history():
    st.session_state["history"] = []


def _show_value(value):
    if isinstance(value, (pd.DataFrame, pd.Series)):
        st.dataframe(value)
    elif value is not None:
        st.write(value)


def render_turn(turn: dict):
    st.markdown(f"**Q:** {turn['question']}")

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
        if turn["attempts"] > 1:
            st.caption(f"The first attempt failed; fixed automatically after {turn['attempts']} attempts.")

    if turn["code"]:
        with st.expander("View generated Python code"):
            st.code(turn["code"], language="python")
    with st.expander(f"What was sent to Gemini ({len(turn['sent_prompts'])} request(s))"):
        for number, prompt in enumerate(turn["sent_prompts"], start=1):
            st.markdown(f"**Request {number}**")
            st.code(prompt, language="text")


def main():
    st.set_page_config(layout="wide", page_title="Universal Data AI")
    st.markdown(
        """
        <div style='background-color: #f4f6f9; padding: 8px; border-radius: 6px; text-align: center; margin-bottom: 10px; border: 1px solid #e0e4e8;'>
            <span style='font-size: 0.9em;'>✨ Inspired by <a href='https://github.com/nipunbatra/vayuchat-webllm' target='_blank' style='text-decoration: none; font-weight: bold; color: #0366d6;'>VayuChat-WebLLM</a></span>
        </div>
        """,
        unsafe_allow_html=True
    )
    st.title("Universal CSV Analyser (WebAssembly)")
    st.markdown("This app runs **entirely in your browser**! Upload a CSV file and ask questions.")

    st.session_state.setdefault("history", [])
    st.session_state.setdefault("df", None)

    st.sidebar.title("API Configuration")
    user_api_key = st.sidebar.text_input("Enter Gemini API Key:", type="password", help="Get a free key from Google AI Studio")
    if st.sidebar.button("Enter", key="save_api_btn"):
        st.sidebar.success("API Key Saved")

    st.sidebar.title("Model Configuration")
    selected_model = st.sidebar.selectbox(
        "Select Gemini Model:",
        [
            "gemini-3.5-flash-lite",
            "gemma-4-26b-a4b-it",
            "gemini-3.5-flash",
            "gemini-2.5-flash",
            "gemini-2.5-pro",
            "gemini-flash-latest",
            "gemini-pro-latest"
        ],
        index=0
    )

    st.sidebar.title("Data Upload")
    st.sidebar.file_uploader("Upload a CSV file", type=["csv"], key="uploaded_csv", on_change=_on_upload)

    if st.session_state.get("load_error"):
        st.error(st.session_state["load_error"])
    df = st.session_state.get("df")
    if df is None:
        st.info("Please upload a CSV file in the sidebar to begin.")
        return

    st.sidebar.success(f"Loaded {st.session_state.get('file_name') or 'file'} successfully!")

    with st.expander("Preview Dataset"):
        st.dataframe(df.head())

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
    if st.sidebar.button("Clear question history"):
        _clear_history()

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
            st.session_state["history"].append(turn)

    for index, turn in enumerate(reversed(st.session_state["history"])):
        if index:
            st.divider()
        render_turn(turn)


if __name__ == "__main__":
    main()
