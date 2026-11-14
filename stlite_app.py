import builtins
import codecs
import contextlib
import csv
import io
import json
import re
import traceback
import urllib.error
import urllib.parse
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

# Modules the generated code may import. Anything else (networking, sys, os,
# js, pyodide ...) is refused so the code cannot send data out of the browser.
ALLOWED_IMPORTS = {
    "pandas", "numpy", "matplotlib", "mpl_toolkits",
    "difflib", "math", "statistics", "datetime", "calendar", "zoneinfo", "dateutil",
    "re", "string", "textwrap", "json", "collections", "itertools", "functools", "operator",
    "decimal", "fractions", "random", "warnings",
}

# Builtins removed from the generated code's namespace.
BLOCKED_BUILTINS = ("open", "input", "breakpoint", "help", "exit", "quit")


GEMINI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta"

# Shown before a key is entered, or if the model list cannot be fetched.
# Also sets the preferred order when the real list is available.
FALLBACK_MODELS = [
    "gemini-3.5-flash-lite",
    "gemma-4-26b-a4b-it",
    "gemini-3.5-flash",
    "gemini-2.5-flash",
    "gemini-2.5-pro",
    "gemini-flash-latest",
    "gemini-pro-latest",
]

# Models that support generateContent but are not text/code models.
EXCLUDED_MODEL_WORDS = (
    "embedding", "aqa", "tts", "image", "imagen", "veo",
    "native-audio", "live", "robotics", "computer-use",
)


class GeminiError(Exception):
    """A Gemini API problem, with a message written for the user."""


def describe_api_error(status: int, body: str) -> str:
    """Turns an HTTP error from Gemini into a plain-language message."""
    message, reasons = "", set()
    try:
        error = json.loads(body).get("error", {})
        message = error.get("message", "")
        reasons = {d.get("reason") for d in error.get("details", []) if isinstance(d, dict)}
    except (ValueError, AttributeError):
        pass

    if "API_KEY_INVALID" in reasons or "API key not valid" in message:
        return "Gemini rejected the API key. Check it was copied correctly from Google AI Studio."
    if status == 403:
        return f"This API key is not allowed to use the Gemini API (permission denied). {message}".strip()
    if status == 404:
        return "The selected model is not available for this API key. Choose another model in the sidebar."
    if status == 429:
        return ("The quota or rate limit for this API key has been reached. "
                "Wait a minute and try again, or choose another model.")
    if status >= 500:
        return f"Gemini is having problems right now (error {status}). Try again shortly."
    return f"Gemini returned error {status}: {message or body[:300]}"


def _gemini_request(path: str, api_key: str, payload=None) -> dict:
    """Calls the Gemini REST API. The key goes in a header, never in the URL."""
    headers = {"x-goog-api-key": api_key.strip()}
    data = None
    if payload is not None:
        data = json.dumps(payload).encode("utf-8")
        headers["Content-Type"] = "application/json"
    req = urllib.request.Request(
        f"{GEMINI_BASE_URL}/{path}", data=data, headers=headers,
        method="POST" if data is not None else "GET",
    )
    try:
        # In Pyodide, urlopen may return 4xx/5xx responses instead of raising.
        with urllib.request.urlopen(req) as response:
            status = getattr(response, "status", 200)
            body = response.read().decode("utf-8", errors="replace")
    except urllib.error.HTTPError as exc:
        status = exc.code
        body = exc.read().decode("utf-8", errors="replace")
    except Exception as exc:
        reason = getattr(exc, "reason", exc)
        raise GeminiError(f"Could not reach Gemini ({reason}). Check your internet connection.") from None

    if status != 200:
        raise GeminiError(describe_api_error(status, body))
    try:
        return json.loads(body)
    except ValueError:
        raise GeminiError("Gemini sent a response that could not be read. Try again.") from None


def order_models(names) -> list:
    """Preferred models first (in FALLBACK_MODELS order), then the rest A-Z."""
    names = list(dict.fromkeys(names))
    preferred = [m for m in FALLBACK_MODELS if m in names]
    return preferred + sorted(m for m in names if m not in preferred)


def list_models(api_key: str) -> list:
    """Text models this key can use for generateContent."""
    models, page_token = [], None
    for _ in range(5):
        path = "models?pageSize=1000"
        if page_token:
            path += "&pageToken=" + urllib.parse.quote(page_token)
        data = _gemini_request(path, api_key)
        for model in data.get("models", []):
            if "generateContent" not in model.get("supportedGenerationMethods", []):
                continue
            name = model.get("name", "").removeprefix("models/")
            if name and not any(word in name for word in EXCLUDED_MODEL_WORDS):
                models.append(name)
        page_token = data.get("nextPageToken")
        if not page_token:
            break
    return order_models(models)


def extract_code(text: str) -> str:
    """Pulls Python code out of a reply, whether or not the block is tagged."""
    blocks = re.findall(r"```[ \t]*([A-Za-z0-9_+-]*)[^\n]*\n(.*?)```", text, re.DOTALL)
    for language, body in blocks:
        if language.lower() in ("python", "py", "python3"):
            return body.strip()
    if blocks:
        return blocks[0][1].strip()
    unterminated = re.search(r"```[ \t]*(?:python3?|py)?[^\n]*\n(.*)", text, re.DOTALL)
    if unterminated:
        return unterminated.group(1).strip()
    return text.strip()


def generate_python_code(prompt: str, model_name: str, api_key: str) -> str:
    """Asks Gemini for code and returns just the code."""
    code = extract_code(call_gemini_text(prompt, model_name, api_key))
    if not code:
        raise GeminiError("Gemini's reply did not contain any code. Try again.")
    return code


def call_gemini_text(prompt: str, model_name: str, api_key: str) -> str:
    """Sends one prompt to Gemini and returns the reply text."""
    payload = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {"temperature": 0.0},
    }
    data = _gemini_request(f"models/{model_name}:generateContent", api_key, payload)

    block_reason = (data.get("promptFeedback") or {}).get("blockReason")
    if block_reason:
        raise GeminiError(f"Gemini blocked the request ({block_reason}). Try rephrasing the question.")
    candidates = data.get("candidates") or []
    if not candidates:
        raise GeminiError("Gemini returned an empty response. Try again or choose another model.")
    candidate = candidates[0]
    parts = (candidate.get("content") or {}).get("parts") or []
    text = "".join(p.get("text", "") for p in parts if not p.get("thought"))
    if not text.strip():
        reason = candidate.get("finishReason", "unknown")
        raise GeminiError(f"Gemini returned no text (finish reason: {reason}). Try rephrasing the question.")
    return text


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


class NetworkBlockedError(PermissionError):
    pass


@contextlib.contextmanager
def network_blocked():
    """Disables Python's network functions while generated code runs.

    Catches indirect routes such as pd.read_csv("https://...") that the import
    allow-list alone would miss.
    """
    import http.client
    import socket

    def _blocked(*args, **kwargs):
        raise NetworkBlockedError("Network access is disabled for generated code. Work only with `df`.")

    targets = [
        (urllib.request, "urlopen"),
        (urllib.request.OpenerDirector, "open"),
        (http.client.HTTPConnection, "connect"),
        (socket.socket, "connect"),
        (socket, "create_connection"),
    ]
    saved = [(obj, attr, getattr(obj, attr)) for obj, attr in targets if hasattr(obj, attr)]
    try:
        for obj, attr, _ in saved:
            setattr(obj, attr, _blocked)
        yield
    finally:
        for obj, attr, original in saved:
            setattr(obj, attr, original)


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
        root = name.split(".")[0]
        if level == 0 and root == "streamlit":
            return recorder
        if level != 0 or root not in ALLOWED_IMPORTS:
            raise ImportError(
                f"Importing {name} is not allowed here. Allowed modules: "
                + ", ".join(sorted(ALLOWED_IMPORTS))
            )
        return builtins.__import__(name, globals, locals, fromlist, level)

    safe_builtins = {k: v for k, v in vars(builtins).items() if k not in BLOCKED_BUILTINS}
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
        with contextlib.redirect_stdout(printed), network_blocked():
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
        # Code the user edited may contain values they typed, so it is never sent automatically.
        if turn.get("code") and not turn.get("error") and not turn.get("edited")
        and turn.get("anonymise") == anonymise
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


def run_edited_code(base_turn: dict, code: str, df: pd.DataFrame) -> dict:
    """Runs code the user edited, locally only. Nothing is sent to Gemini."""
    request = prepare_request(df, "", "", base_turn.get("anonymise", False))
    turn = {
        "question": base_turn["question"],
        "sent_question": base_turn.get("sent_question", base_turn["question"]),
        "anonymise": base_turn.get("anonymise", False),
        "edited": True,
        "sent_prompts": [],
        "code": code,
        "attempts": 1,
        "outputs": [],
        "result": None,
        "error": None,
    }
    try:
        turn["result"], turn["outputs"] = run_generated_code(code, request["work_df"], request["reverse"])
    except Exception as exc:
        turn["error"] = f"{type(exc).__name__}: {restore_names(str(exc), request['reverse'])}"
    return turn


def build_explain_prompt(schema: str, code: str) -> str:
    """Prompt for 'Explain this code'. Contains only the column list and the code."""
    return "\n".join([
        "You are teaching a researcher who is learning Python and pandas.",
        "Explain what the code below does, step by step, in plain English with UK spelling.",
        "Refer to line numbers. Point out any assumptions the code makes about the data and anything "
        "that could give a misleading result (for example, how missing values or duplicates are treated).",
        "Keep it under 250 words. Do not rewrite the code.",
        "",
        "The code works on a pandas dataframe `df`.",
        schema,
        "",
        "```python",
        code,
        "```",
    ])


# ---------------------------------------------------------------------------
# Data profile (computed locally, never sent)
# ---------------------------------------------------------------------------

def _looks_like_dates(series: pd.Series) -> bool:
    import warnings
    sample = series.dropna().astype(str).str.strip()
    sample = sample[sample.str.contains(r"\d", regex=True)].head(200)
    if len(sample) < 3 or len(sample) < 0.5 * min(len(series.dropna()), 200):
        return False
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            parsed = pd.to_datetime(sample, errors="coerce", format="mixed")
        except (TypeError, ValueError):
            parsed = pd.to_datetime(sample, errors="coerce")
    return parsed.notna().mean() >= 0.8


def profile_dataset(df: pd.DataFrame):
    """Returns (summary dict, per-column table) describing data quality."""
    rows = []
    for col in df.columns:
        series = df[col]
        missing = int(series.isna().sum())
        row = {
            "column": str(col),
            "type": str(series.dtype),
            "missing": missing,
            "missing %": round(100 * missing / len(series), 1) if len(series) else 0.0,
            "distinct": int(series.nunique(dropna=True)),
            "min": None, "median": None, "mean": None, "max": None,
            "most common": "",
            "note": "",
        }
        if pd.api.types.is_numeric_dtype(series) and not pd.api.types.is_bool_dtype(series):
            values = series.dropna()
            if len(values):
                row.update({
                    "min": float(values.min()), "median": float(values.median()),
                    "mean": round(float(values.mean()), 3), "max": float(values.max()),
                })
        else:
            counts = series.dropna().astype(str).value_counts().head(3)
            row["most common"] = ", ".join(f"{value} ({count})" for value, count in counts.items())
            if not pd.api.types.is_datetime64_any_dtype(series) and _looks_like_dates(series):
                row["note"] = "Looks like dates stored as text"
        if missing == len(series) and len(series):
            row["note"] = "Entirely empty"
        rows.append(row)

    summary = {
        "rows": len(df),
        "columns": df.shape[1],
        "duplicate rows": int(df.duplicated().sum()) if len(df.columns) else 0,
        "missing cells": int(df.isna().sum().sum()),
    }
    return summary, pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Synthetic sample dataset
# ---------------------------------------------------------------------------

SAMPLE_FILE_NAME = "synthetic_air_quality_2024.csv"
SAMPLE_STATIONS = {   # fictional station: typical PM2.5 level
    "Riverside": 55, "Old Market": 95, "Industrial Estate": 110,
    "University Campus": 45, "Green Park": 35,
}


def make_sample_dataset(seed: int = 42) -> pd.DataFrame:
    """Synthetic daily air-quality readings for five fictional stations in 2024.

    Not real measurements. Deliberately includes problems to practise on:
    missing values, inconsistent station names, dates stored as text and a few
    duplicate rows.
    """
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2024-01-01", "2024-12-31", freq="D")
    day = dates.dayofyear.to_numpy()
    winter = 1 + 0.8 * np.cos(2 * np.pi * (day - 5) / 366)        # peaks in early January
    frames = []
    for station, level in SAMPLE_STATIONS.items():
        n = len(dates)
        wind = np.clip(rng.gamma(2.0, 1.2, n), 0.2, None)
        pm25 = level * winter * rng.lognormal(0, 0.25, n) / np.sqrt(wind / 2.4)
        frames.append(pd.DataFrame({
            "date": dates.strftime("%Y-%m-%d"),
            "station": station,
            "pm25_ugm3": np.round(pm25, 1),
            "pm10_ugm3": np.round(pm25 * rng.uniform(1.5, 2.2, n), 1),
            "temperature_c": np.round(25 - 9 * np.cos(2 * np.pi * (day - 15) / 366) + rng.normal(0, 1.5, n), 1),
            "relative_humidity_pct": np.round(np.clip(60 + 20 * np.sin(2 * np.pi * (day - 120) / 366)
                                                      + rng.normal(0, 8, n), 10, 100), 0),
            "wind_speed_ms": np.round(wind, 1),
        }))
    df = pd.concat(frames, ignore_index=True)

    # Problems to practise on.
    for col, share in (("pm25_ugm3", 0.04), ("pm10_ugm3", 0.06), ("relative_humidity_pct", 0.02)):
        df.loc[rng.random(len(df)) < share, col] = np.nan
    messy = rng.random(len(df)) < 0.02
    df.loc[messy, "station"] = df.loc[messy, "station"].str.lower() + " "
    duplicates = df.sample(5, random_state=seed)
    return pd.concat([df, duplicates], ignore_index=True)


# ---------------------------------------------------------------------------
# Exporting a session
# ---------------------------------------------------------------------------

_EXPORT_SHIM = '''class _StreamlitStandIn:
    """Lets the st.* calls written for the app run outside it."""
    _charts = {"bar_chart": "bar", "line_chart": "line", "area_chart": "area"}

    def __getattr__(self, name):
        def show(*args, **kwargs):
            if name == "pyplot":
                plt.show()
            elif name in self._charts and args:
                pd.DataFrame(args[0]).plot(kind=self._charts[name], x=kwargs.get("x"), y=kwargs.get("y"))
                plt.show()
            else:
                for arg in args:
                    try:
                        display(arg)   # Jupyter
                    except NameError:
                        print(arg)
        return show


st = _StreamlitStandIn()'''


def exportable_turns(history: list) -> list:
    """Successful turns, oldest first."""
    return [t for t in history if t.get("code") and not t.get("error")]


def _export_setup(reader_code: str, columns, turns: list) -> str:
    lines = [
        "import pandas as pd",
        "import numpy as np",
        "import matplotlib.pyplot as plt",
        "",
        _EXPORT_SHIM,
        "",
        "# Change the path if your data file is somewhere else.",
        f"original_df = {reader_code}",
    ]
    if any(t.get("anonymise") for t in turns):
        aliases = {str(k): v for k, v in make_aliases(columns).items()}
        lines += [
            "",
            "# Some questions were asked with anonymised column names.",
            f"ALIASES = {aliases!r}",
        ]
    return "\n".join(lines)


def _export_cell(turn: dict) -> str:
    start = "df = original_df.rename(columns=ALIASES)" if turn.get("anonymise") else "df = original_df.copy()"
    return f"{start}\n\n{turn['code'].rstrip()}"


def _question_label(number: int, turn: dict) -> str:
    question = " ".join(str(turn["question"]).split())
    return f"Question {number}: {question}" + (" (edited by you)" if turn.get("edited") else "")


EXPORT_INTRO = (
    "Analysis session exported from the Universal CSV Analyser.\n"
    "Each section is one question; every section starts again from the original data, as the app does.\n"
    "Results are not included, so this file can be shared without your data. "
    "Run it next to your data file to reproduce them."
)


def build_script(history: list, reader_code: str, columns) -> str:
    turns = exportable_turns(history)
    parts = ['"""' + EXPORT_INTRO + '\n"""', _export_setup(reader_code, columns, turns)]
    for number, turn in enumerate(turns, start=1):
        parts.append(
            f"# %% {_question_label(number, turn)}\n"
            f"{_export_cell(turn)}\n\n"
            "if result is not None:\n    print(result)"
        )
    return "\n\n\n".join(parts) + "\n"


def build_notebook(history: list, reader_code: str, columns) -> str:
    turns = exportable_turns(history)

    def cell(kind, text):
        lines = text.splitlines(keepends=True)
        if kind == "markdown":
            return {"cell_type": "markdown", "metadata": {}, "source": lines}
        return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": lines}

    cells = [
        cell("markdown", "# Analysis session\n\n" + EXPORT_INTRO),
        cell("code", _export_setup(reader_code, columns, turns)),
    ]
    for number, turn in enumerate(turns, start=1):
        cells.append(cell("markdown", f"## {_question_label(number, turn)}"))
        cells.append(cell("code", _export_cell(turn) + "\n\nresult"))
    notebook = {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python"},
        },
        "nbformat": 4,
        "nbformat_minor": 4,
    }
    return json.dumps(notebook, indent=1, ensure_ascii=False)


# ---------------------------------------------------------------------------
# Loading files
# ---------------------------------------------------------------------------

UPLOAD_TYPES = ["csv", "tsv", "txt", "xlsx"]
DELIMITER_NAMES = {",": "comma", ";": "semicolon", "\t": "tab", "|": "pipe"}


def decode_bytes(raw: bytes):
    """Decodes text, trying UTF-8 first, then the common Windows/Latin encodings."""
    if raw.startswith(codecs.BOM_UTF8):
        return raw[len(codecs.BOM_UTF8):].decode("utf-8"), "UTF-8 with BOM"
    for encoding, label in (("utf-8", "UTF-8"), ("cp1252", "Windows-1252")):
        try:
            return raw.decode(encoding), label
        except UnicodeDecodeError:
            continue
    return raw.decode("latin-1"), "Latin-1"


def sniff_delimiter(text: str, default: str = ",") -> str:
    """Picks the delimiter that splits the first lines into a consistent number of columns."""
    lines = [line for line in text.splitlines()[:50] if line.strip()][:20]
    if not lines:
        return default
    best, best_score = default, (0.0, 0)
    for delimiter in [default] + [d for d in DELIMITER_NAMES if d != default]:
        try:
            widths = [len(row) for row in csv.reader(lines, delimiter=delimiter)]
        except csv.Error:
            continue
        if not widths or widths[0] < 2:
            continue
        score = (sum(w == widths[0] for w in widths) / len(widths), widths[0])
        if score > best_score:
            best, best_score = delimiter, score
    return best


ENCODING_CODECS = {"UTF-8": None, "UTF-8 with BOM": "utf-8-sig", "Windows-1252": "cp1252", "Latin-1": "latin-1"}


def load_table(name: str, raw: bytes):
    """Reads an uploaded CSV/TSV/TXT or Excel file.

    Returns (df, description, reader_code), where reader_code is the pandas call
    that reads the same file, used in exported scripts and notebooks.
    """
    lower = name.lower()
    if lower.endswith((".xlsx", ".xlsm")):
        with pd.ExcelFile(io.BytesIO(raw)) as book:
            sheets = book.sheet_names
            df = book.parse(sheets[0])
        description = f"Excel sheet '{sheets[0]}'"
        if len(sheets) > 1:
            description += f" (only the first of {len(sheets)} sheets is used)"
        return df, description, f"pd.read_excel({name!r}, sheet_name={sheets[0]!r})"

    text, encoding = decode_bytes(raw)
    delimiter = sniff_delimiter(text, default="\t" if lower.endswith(".tsv") else ",")
    df = pd.read_csv(io.StringIO(text), sep=delimiter)
    args = [repr(name)]
    if delimiter != ",":
        args.append(f"sep={delimiter!r}")
    if ENCODING_CODECS[encoding]:
        args.append(f"encoding={ENCODING_CODECS[encoding]!r}")
    return df, f"{encoding}, {DELIMITER_NAMES[delimiter]}-separated", f"pd.read_csv({', '.join(args)})"


# ---------------------------------------------------------------------------
# User interface
# ---------------------------------------------------------------------------

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
    st.markdown(
        "Upload a CSV or Excel file and ask questions in plain English. **Your data stays in your browser.** "
        "Only your question, your context instructions and the column names and types are sent to "
        "Google's Gemini, which writes Python code that then runs here on your computer."
    )
    with st.expander("How your data is handled"):
        st.markdown(DATA_HANDLING_NOTE)

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
