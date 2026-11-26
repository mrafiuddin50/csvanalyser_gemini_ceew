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


# Charts follow the CEEW data viz skill. Gemini picks the chart from the question and the column names;
# the ceew functions check the rules that need the data on the user's computer.
CHART_RULES = "\n".join([
    "- CHARTS: draw every chart with the `ceew` chart functions (already available; do not import them). They apply "
    "the CEEW house style and check the chart rules against the data. Pass the table you plotted: one row per category "
    "or time point, already aggregated (except ceew.histogram, ceew.boxplot and ceew.scatter, which take one row per "
    "observation). Arguments name columns of that table. Set `result` to the table you plotted; for "
    "ceew.histogram, ceew.boxplot and ceew.scatter, set it to a short summary instead (for example "
    "`.describe()` for each group), not the raw rows.",
    "  ceew.bar(data, category, value, orientation='auto', sort=None, highlight=None, why='')  # ranking, magnitude, a few time periods",
    "  ceew.lollipop(data, category, value, sort=None, highlight=None, why='')",
    "  ceew.grouped_bar(data, category, values=[...2 to 4 columns], orientation='auto', sort=None, why='')",
    "  ceew.diverging_bar(data, category, value, reference='zero', sort=None, why='')  # deviation from zero, a target or an average",
    "  ceew.stacked_bar(data, category, parts=[...], orientation='auto', why='')  # parts of one whole that add up to 100% in every row",
    "  ceew.pie(data, category, value, why='') / ceew.donut(...)  # at most 5 parts of one whole adding up to 100%",
    "  ceew.line(data, x, y, series=None, highlight=None, why='')  # change over time; y is one column or a list (wide), or use series= for long data",
    "  ceew.area(data, x, y, why='')  # one series, or stacked series that are parts of one whole",
    "  ceew.slope(data, category, start, end, why='') / ceew.range_plot(data, category, start, end, sort=None, why='')  # change between two values",
    "  ceew.histogram(data, value, bin_width=None, why='')  # distribution; pass bin_width only if the question gives one",
    "  ceew.boxplot(data, group, value, orientation='auto', why='')  # distribution, for specialist readers; "
    "orientation='vertical' or 'horizontal' when the question asks for one",
    "  ceew.scatter(data, x, y, colour=None, why='')  # correlation of two numeric columns",
    "  ceew.heatmap(data, row, column, value, why='')  # two categorical columns and one value",
    "  sort: None (largest first), 'descending', 'ascending', 'alphabetical' or 'data' (keep the table's order); "
    "only pass it if the question asks for an order. Time periods always stay in time order.",
    "  Pass arguments by these names only; do not pass title, xlabel, color, figsize or other plotting options.",
    "- Choose the chart from what the question compares (ranking, magnitude, change over time, part-to-whole, "
    "distribution, correlation, deviation). Prefer the simplest chart that shows it. A chart named in the question is a "
    "proposal: use it only if it fits; otherwise use the chart that fits.",
    "- Pass `why=`: one or two sentences naming the relationship (for example 'magnitude') and why this chart shows it. "
    "Base it only on the question and the column names.",
    "- Do NOT write titles, headlines, annotations, source lines or notes, and do not choose colours: the researcher "
    "writes the words and the app sets the colours. Never use two y-axes, a size encoding, 3D, or a trend line.",
    "- highlight: only items the question names. Never choose items to highlight yourself.",
    "- Not available: line + column charts (two axes), violin, chord, arc and circle timeline charts (the CEEW Outreach "
    "data viz team makes these), maps and Venn diagrams. If asked for one, say so with st.info and draw the nearest "
    "chart from the list.",
    "- Use Streamlit charts (st.line_chart, st.bar_chart) only if the question asks for an interactive chart. "
    "Use plain matplotlib (`fig, ax = plt.subplots()`, then `st.pyplot(fig)`) only for a chart the list cannot draw; "
    "NEVER use `plt.show()`.",
])


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
        CHART_RULES,
        "- Timestamps held as text (type str or object) must go through `pd.to_datetime` before use. "
        "'By hour', 'by month' or 'by weekday' means the hour of the day (0 to 23), the month of the year (1 to 12) "
        "or the day of the week, pooled over the whole period (for example `df.groupby(times.dt.hour)`), unless "
        "the question asks for each day or each month of each year. Name such a column 'hour', 'month' or 'weekday'.",
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
