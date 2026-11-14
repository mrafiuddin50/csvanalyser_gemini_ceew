"""Exporting a session as a Python script or Jupyter notebook (code only, no results)."""

import json

from .privacy import make_aliases


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
