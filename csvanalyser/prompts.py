"""Text sent to Gemini: the analysis prompt and the 'Explain this code' prompt."""

import pandas as pd

from .sandbox import ALLOWED_ST_CALLS


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
