"""Example questions, built only from column names and types. Nothing here looks at data values."""

QUANTITATIVE = {"quantitative", "quantitative (stored as text)"}
CATEGORIES = {"categorical", "geographic (place names)", "boolean (text)"}
DATES = {"datetime", "datetime (stored as text)"}


def suggest_questions(column_info, limit: int = 5) -> list:
    """Returns up to `limit` plain-English example questions for the loaded columns.

    column_info: the audit's ColumnInfo list (name + inferred type). Columns that may hold
    personal information are skipped.
    """
    usable = [c for c in column_info if not getattr(c, "sensitive", False)]
    numbers = [c.name for c in usable if c.semantic in QUANTITATIVE]
    groups = [c.name for c in usable if c.semantic in CATEGORIES and 2 <= c.unique <= 50]
    dates = [c.name for c in usable if c.semantic in DATES]
    missing = any(c.missing for c in column_info)

    # Kept short so they fit on a button; the full text goes into the question box when clicked.
    questions = []
    if numbers and groups:
        questions.append(f"Average {numbers[0]} for each {groups[0]}")
    if numbers and dates:
        questions.append(f"Monthly average {numbers[0]} over time, as a line chart")
    if groups:
        questions.append(f"Number of rows for each {groups[0]}, as a bar chart")
    if numbers:
        questions.append(f"The 10 rows with the highest {numbers[0]}")
    if len(numbers) >= 2:
        questions.append(f"Is {numbers[0]} related to {numbers[1]}? Show a scatter plot")
    if missing:
        questions.append("Which columns have the most missing values?")
    if not questions:
        questions.append("A short summary of each column")
    return questions[:limit]
