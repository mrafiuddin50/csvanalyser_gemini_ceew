"""Runs generated code locally, sealed off from the app, the network and most modules."""

import builtins
import contextlib
import io
import urllib.request

import numpy as np
import pandas as pd

from .privacy import GENERATED_FILENAME, restore_names


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


class NoResultError(Exception):
    pass


# pandas 3 and later use copy-on-write: a light copy shares memory with the original but any change
# made through it copies the affected part first, so the original is never altered. Older pandas
# needs a full copy (twice the memory for a moment, about 0.5 s for 200,000 rows).
COPY_ON_WRITE = int(pd.__version__.split(".")[0]) >= 3


def private_copy(df: pd.DataFrame) -> pd.DataFrame:
    """A copy of the data that generated code can change without touching the original."""
    return df.copy(deep=not COPY_ON_WRITE)


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
        "df": private_copy(df),
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
