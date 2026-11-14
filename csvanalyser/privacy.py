"""What the model is allowed to see.

Core rule of this app: the model never sees the data. Only the question, the
user's context text, column names/types and the row count are sent to Gemini.
Row values, results and raw error messages stay in the browser.
"""

import re
import traceback

import pandas as pd


GENERATED_FILENAME = "<generated code>"


REDACTED = "<redacted>"


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
