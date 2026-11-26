"""CEEW house-style charts, following the CEEW data viz skill.

The skill's rules (chart menu, rules R1 to R8, colour rules and house style) are applied here, on the
user's computer. Gemini only chooses a chart function from the question and the column names; the
checks that need the data (do the parts add up to 100%? are there gaps in a time series? how many
categories?) run here, and their messages are shown to the researcher only. Nothing in this file is
sent to Gemini.

The researcher owns the words: charts carry placeholders for the headline, source and note until
the researcher types them in.

This file imports only pandas, numpy and matplotlib, so it can also be saved next to an exported
notebook and used there (see notebook_charts).
"""

from __future__ import annotations

import io
import math
import os
import re
import textwrap
from dataclasses import dataclass, field

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Palette and fixed colours (colour rules, sections 1 and 3)
# ---------------------------------------------------------------------------

SHADES = {
    "orange": {"lightest": "#F39673", "light": "#F37E51", "base": "#E85724", "dark": "#B93A26", "darkest": "#811C19"},
    "green": {"lightest": "#CCDF84", "light": "#BCD265", "base": "#8CB73F", "dark": "#5E833A", "darkest": "#005C3A"},
    "blue": {"lightest": "#86BFDE", "light": "#58AEDB", "base": "#029BD6", "dark": "#036493", "darkest": "#023F5C"},
}
SHADE_ORDER = ("lightest", "light", "base", "dark", "darkest")
GREY_LIGHTEST = "#CFCFCE"
GREY_LIGHT = "#7F7F7F"
GREY_DARK = "#333232"
GREY_DARKEST = "#0C0B0B"
CREAM = "#F5F0E7"
WHITE = "#FFFFFF"

TEXT_COLOUR = GREY_DARK        # all text
AXIS_COLOUR = GREY_LIGHT       # axis lines
GRID_COLOUR = GREY_LIGHTEST    # gridlines

FONT_FAMILY = "Familjen Grotesk"

# For Streamlit's own interactive charts (theme settings in index.html use the same values).
STREAMLIT_CATEGORICAL = [SHADES["orange"]["base"], SHADES["blue"]["base"], GREY_LIGHT, SHADES["orange"]["lightest"],
                         SHADES["blue"]["lightest"], SHADES["orange"]["dark"], SHADES["blue"]["dark"], GREY_LIGHTEST]


# ---------------------------------------------------------------------------
# Style: where the chart will be used and which two primaries it uses
# ---------------------------------------------------------------------------

USES = {"report": "Research report", "presentation": "Presentation", "outreach": "Outreach"}
SHAPES = {"square": "Square, 1080 × 1080 px", "portrait": "Portrait, 1080 × 1350 px"}
PAIRS = {
    "orange-blue": ("orange", "blue"),
    "orange-green": ("orange", "green"),
    "green-blue": ("green", "blue"),
}
PAIR_LABELS = {"orange-blue": "Orange and blue", "orange-green": "Orange and green", "green-blue": "Green and blue"}


@dataclass(frozen=True)
class Style:
    use: str = "report"          # report | presentation | outreach
    shape: str = "square"        # outreach only: square | portrait
    pair: str = "orange-blue"    # the two primaries for this chart (no pair is the default in the skill)

    @property
    def primaries(self):
        return PAIRS.get(self.pair, PAIRS["orange-blue"])

    @property
    def background(self):
        return CREAM if self.use == "outreach" else WHITE


# ---------------------------------------------------------------------------
# Fonts
# ---------------------------------------------------------------------------

_FONTS_FOUND = None


def fonts_available() -> bool:
    """Registers Familjen Grotesk with matplotlib the first time a chart is drawn."""
    global _FONTS_FOUND
    if _FONTS_FOUND is None:
        _FONTS_FOUND = _register_fonts()
    return _FONTS_FOUND


def _register_fonts() -> bool:
    try:
        from matplotlib import font_manager
    except ImportError:
        return False
    here = os.path.dirname(os.path.abspath(__file__))
    folders = [os.path.join(here, "..", "assets", "fonts"), os.path.join(here, "fonts"), here,
               os.path.join(os.getcwd(), "assets", "fonts"), os.path.join(os.getcwd(), "fonts")]
    found = False
    for folder in folders:
        if not os.path.isdir(folder):
            continue
        for name in sorted(os.listdir(folder)):
            if name.startswith("FamiljenGrotesk") and name.endswith(".ttf"):
                font_manager.fontManager.addfont(os.path.join(folder, name))
                found = True
    return found


def house_rc(style: Style | None = None, font_size: float = 11) -> dict:
    """Matplotlib settings for the house style. Also used for charts drawn with plain matplotlib."""
    from cycler import cycler
    style = style or Style()
    colours, _ = categorical_colours([f"series {i}" for i in range(8)], style)
    return {
        "font.family": [FONT_FAMILY, "DejaVu Sans"] if fonts_available() else ["DejaVu Sans"],
        "font.size": font_size,
        "axes.titlesize": font_size, "axes.labelsize": font_size, "legend.fontsize": font_size,
        "xtick.labelsize": font_size, "ytick.labelsize": font_size, "figure.titlesize": font_size,
        "axes.titlelocation": "left", "axes.titleweight": "semibold",
        "text.color": TEXT_COLOUR, "axes.labelcolor": TEXT_COLOUR, "axes.titlecolor": TEXT_COLOUR,
        "xtick.labelcolor": TEXT_COLOUR, "ytick.labelcolor": TEXT_COLOUR,
        "xtick.color": AXIS_COLOUR, "ytick.color": AXIS_COLOUR,
        "axes.edgecolor": AXIS_COLOUR, "axes.linewidth": 0.8,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": False, "grid.color": GRID_COLOUR, "grid.linewidth": 0.8, "axes.axisbelow": True,
        "axes.facecolor": style.background, "figure.facecolor": style.background,
        "savefig.facecolor": style.background,
        "axes.prop_cycle": cycler(color=colours),
        "legend.frameon": False,
        "axes.formatter.useoffset": False,
        "svg.fonttype": "none",
    }


# ---------------------------------------------------------------------------
# Colour rules
# ---------------------------------------------------------------------------

_OTHER = re.compile(r"^\s*(others?|other\s.*|not available|n/?a|unknown|none|missing|not known|"
                    r"don'?t know|can'?t say|no response|not reported|rest)\s*$", re.IGNORECASE)


def is_other(label) -> bool:
    """'Other', 'Not available' and similar categories are always grey (colour rule 7)."""
    return bool(_OTHER.match(str(label)))


def categorical_colours(labels, style: Style):
    """Colours for unrelated categories: the two primaries and their shades, plus grey; at most eight.

    Returns (colours, greyed) where greyed lists categories beyond eight, which turn light grey.
    """
    p1, p2 = style.primaries
    cycle = [SHADES[p1]["base"], SHADES[p2]["base"], GREY_LIGHT, SHADES[p1]["lightest"],
             SHADES[p2]["lightest"], SHADES[p1]["dark"], SHADES[p2]["dark"], SHADES[p1]["light"]]
    others = [is_other(label) for label in labels]
    if any(others):
        cycle = [c for c in cycle if c != GREY_LIGHT]
    colours, greyed, index = [], [], 0
    for label, other in zip(labels, others):
        if other:
            colours.append(GREY_LIGHT)
        elif index < len(cycle):
            colours.append(cycle[index])
            index += 1
        else:
            colours.append(GREY_LIGHTEST)
            greyed.append(str(label))
    return colours, greyed


def sequential_colours(style: Style):
    """One primary from lightest to darkest: light is low, dark is high."""
    return [SHADES[style.primaries[0]][s] for s in SHADE_ORDER]


DIVERGING_COLOURS = [SHADES["orange"]["dark"], SHADES["orange"]["lightest"], GREY_LIGHTEST,
                     SHADES["blue"]["lightest"], SHADES["blue"]["dark"]]


def highlight_colours(style: Style):
    p1, p2 = style.primaries
    return [SHADES[p1]["base"], SHADES[p2]["base"], SHADES[p1]["dark"]]


# ---------------------------------------------------------------------------
# What a chart function records
# ---------------------------------------------------------------------------

MAX_BARS = 15            # horizontal and vertical bar charts hold at most 15 categories
MAX_COLOURED = 8         # up to eight categories get their own colour; more turn grey
ASK_HIGHLIGHT_ABOVE = 10  # more categories than this: offer the grey-out option
MAX_PIE_SLICES = 5
MAX_SLOPE_CATEGORIES = 4

PLACEHOLDERS = {
    "headline": "[Your headline here]",
    "source": "[Source: ...]",
    "note": "[Note: ...]",
    "axis": "[Axis title, with unit]",
}


@dataclass
class ChartSpec:
    kind: str                       # bar, lollipop, grouped_bar, stacked_bar, diverging_bar, line, area,
                                    # scatter, histogram, pie, slope, range_plot, heatmap, boxplot, table
    data: pd.DataFrame
    params: dict = field(default_factory=dict)
    x_title: str = ""
    y_title: str = ""
    why: str = ""
    notes: list = field(default_factory=list)    # (level, text); level "fallback" or "tip"

    def note(self, text: str, level: str = "tip"):
        self.notes.append((level, text))

    @property
    def axis_fields(self):
        """Which axis titles the researcher can edit."""
        return () if self.kind in ("pie", "table") else ("x_title", "y_title")

    @property
    def value_axes(self):
        """Axis titles that must be shown (they carry the unit); empty ones get a placeholder."""
        return self.params.get("value_axes", ())


class ChartInputError(ValueError):
    """A problem with how a chart function was called; the message never contains data values."""


# ---------------------------------------------------------------------------
# Input helpers
# ---------------------------------------------------------------------------

def _frame(data, func: str) -> pd.DataFrame:
    if isinstance(data, pd.Series):
        name = data.name if data.name is not None else "value"
        index_name = data.index.name if data.index.name is not None else "category"
        return data.rename(name).rename_axis(index_name).reset_index()
    if isinstance(data, pd.DataFrame):
        frame = data
        if not isinstance(frame.index, pd.RangeIndex) and any(n is not None for n in frame.index.names):
            frame = frame.reset_index()
        return frame
    raise ChartInputError(f"ceew.{func} needs a pandas DataFrame or Series as `data`.")


def _column(frame: pd.DataFrame, name, func: str, role: str):
    if name is None:
        raise ChartInputError(f"ceew.{func} needs `{role}` (a column name).")
    if name not in frame.columns:
        raise ChartInputError(f"ceew.{func}: the column given as `{role}` is not in `data`. "
                              f"Pass column names of the dataframe you are plotting.")
    return name


def _columns(frame, names, func, role):
    if isinstance(names, (str, int)) or names is None:
        names = [names]
    return [_column(frame, n, func, role) for n in names]


def _numeric(frame, cols, func):
    for col in cols:
        if not pd.api.types.is_numeric_dtype(frame[col]) or pd.api.types.is_bool_dtype(frame[col]):
            raise ChartInputError(f"ceew.{func}: the column `{col}` must be numeric.")


def _unique(frame, col, func):
    if frame[col].duplicated().any():
        raise ChartInputError(f"ceew.{func} needs one row per `{col}`. Aggregate the data first "
                              f"(for example with groupby), then plot the summary table.")


# Whole-number columns that count time: the name gives the unit, the values must fit its range.
# Hour of day, month and weekday come out of pandas (df[col].dt.hour) as plain integers, often still
# carrying the original column's name, such as 'observation_hour_start'.
TIME_UNITS = [
    (re.compile(r"year|^yr$|^fy$"), 1800, 2200),
    (re.compile(r"hour|^hr$"), 0, 23),
    (re.compile(r"weekday|day_?of_?week|dow"), 0, 7),
    (re.compile(r"month"), 1, 12),
    (re.compile(r"quarter|^qtr$"), 1, 4),
    (re.compile(r"week"), 0, 53),
    (re.compile(r"day"), 1, 31),
]


def is_time(series: pd.Series) -> bool:
    """Dates, periods, and whole numbers that count years, months, weekdays, days or hours."""
    if pd.api.types.is_datetime64_any_dtype(series) or isinstance(series.dtype, pd.PeriodDtype):
        return True
    if not pd.api.types.is_integer_dtype(series) or pd.api.types.is_bool_dtype(series):
        return False
    name = str(series.name or "").lower()
    values = series.dropna()
    if len(values) == 0:
        return False
    for pattern, low, high in TIME_UNITS:
        if pattern.search(name):
            return bool(values.between(low, high).all())
    return False


def _as_time(series: pd.Series) -> pd.Series:
    if isinstance(series.dtype, pd.PeriodDtype):
        return series.dt.to_timestamp()
    return series


def as_dates(series: pd.Series) -> pd.Series:
    """Dates held as Python date objects or text ('2020-11-05', '2020-11') become real dates.

    Generated code often groups by df[col].dt.date, which gives plain date objects; a chart would treat each
    day as a separate label. Text that is mostly not dates (station names, 'Monday') is left alone.
    """
    series = _as_time(series)
    if not (pd.api.types.is_object_dtype(series) or pd.api.types.is_string_dtype(series)):
        return series
    values = series.dropna()
    if len(values) == 0 or not values.map(lambda v: hasattr(v, "year") or isinstance(v, str)).all():
        return series
    if values.map(lambda v: isinstance(v, str) and not any(ch.isdigit() for ch in v)).any():
        return series                                            # words, not dates
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        parsed = pd.to_datetime(series, errors="coerce")
    return parsed if parsed[series.notna()].notna().mean() >= 0.95 else series


def _is_ordered(series: pd.Series) -> bool:
    """Ordered categories (for example from pd.cut): Good, Satisfactory, Moderate ... keep their order."""
    return isinstance(series.dtype, pd.CategoricalDtype) and bool(series.dtype.ordered)


# Scales whose order is part of their meaning. A chart keeps them in this order rather than by size.
ORDERED_SCALES = [
    ["good", "satisfactory", "moderate", "poor", "very poor", "severe"],                 # India's AQI categories
    ["very low", "low", "lower middle", "middle", "medium", "moderate", "upper middle", "high", "very high"],
    ["strongly disagree", "disagree", "neutral", "neither agree nor disagree", "agree", "strongly agree"],
    ["never", "rarely", "sometimes", "often", "always"],
    ["very poor", "poor", "fair", "good", "very good", "excellent"],
]
_LEADING_NUMBER = re.compile(r"^\W*(-?\d+(?:\.\d+)?)")


def natural_order(series: pd.Series):
    """Sort keys when the labels have an order of their own: number bands ('0-30', '(30, 60]') or a known scale."""
    labels = [str(v).strip().lower() for v in series]
    numbers = [_LEADING_NUMBER.match(label) for label in labels]
    if labels and all(numbers):
        return [float(m.group(1)) for m in numbers]
    for scale in ORDERED_SCALES:
        if len(labels) >= 2 and all(label in scale for label in labels):
            return [scale.index(label) for label in labels]
    return None


def labels_of(series: pd.Series) -> list:
    """Category labels as text. Dates are written as plainly as the data allows."""
    series = _as_time(series)
    if pd.api.types.is_datetime64_any_dtype(series):
        values = pd.DatetimeIndex(series)
        at_midnight = (values.hour == 0).all() and (values.minute == 0).all()
        if at_midnight and (values.day == 1).all() and (values.month == 1).all():
            fmt = "%Y"
        elif at_midnight and (values.day == 1).all():
            fmt = "%b %Y"
        elif at_midnight:
            fmt = "%d %b %Y"
        else:
            fmt = "%d %b %Y %H:%M"
        return [v.strftime(fmt) if not pd.isna(v) else "" for v in values]
    return [str(v) for v in series]


SORTS = ("descending", "ascending", "alphabetical", "data")


def _sort(frame, category, value, sort, spec, func):
    """Bar order: time order always wins; otherwise the requested order, largest first by default."""
    if is_time(frame[category]):
        if sort not in (None, "data", "ascending"):
            spec.note("Time periods are kept in time order (the chart rules say time order always wins).")
        return frame.sort_values(category, kind="stable")
    if _is_ordered(frame[category]) and sort in (None, "data"):
        return frame.sort_values(category, kind="stable")       # ordered categories keep their own order
    order = natural_order(frame[category]) if sort is None else None
    if order is not None:
        spec.note("Kept in the order of the categories themselves. To sort by size instead, say so in your question.")
        return frame.assign(_order=order).sort_values("_order", kind="stable").drop(columns="_order")
    if sort is None:
        spec.note("Sorted from largest to smallest. For another order, say so in your question "
                  "(ascending, alphabetical, or the order in your data).")
        sort = "descending"
    if sort not in SORTS:
        raise ChartInputError(f"ceew.{func}: `sort` must be one of {', '.join(SORTS)}.")
    if sort == "descending":
        return frame.sort_values(value, ascending=False, kind="stable")
    if sort == "ascending":
        return frame.sort_values(value, ascending=True, kind="stable")
    if sort == "alphabetical":
        return frame.sort_values(category, key=lambda s: s.astype(str).str.lower(), kind="stable")
    return frame


def _match_highlight(labels, highlight, spec):
    if highlight is None:
        return []
    if isinstance(highlight, str):
        highlight = [highlight]
    wanted = {str(h).strip().lower(): str(h) for h in highlight}
    lower = {label.strip().lower(): label for label in labels}
    matched = [lower[w] for w in wanted if w in lower]
    missing = [wanted[w] for w in wanted if w not in lower]
    if missing:
        spec.note("Not found among the categories, so not highlighted: " + ", ".join(missing) + ".")
    if len(matched) > 3:
        spec.note("Highlight one to three items so the highlight stands out; all requested items are highlighted.")
    return matched


def _offer_grey_out(n, highlight, spec):
    if n > ASK_HIGHLIGHT_ABOVE and not highlight:
        spec.note(f"This chart has {n} categories. If you want, name one to three of them to highlight in your "
                  "question; the rest then turn grey as context. Choosing them is choosing the story, so the "
                  "app never picks them for you.")


def _table(frame, spec, message) -> ChartSpec:
    spec.kind = "table"
    spec.data = frame
    spec.note(message, "fallback")
    return spec


_SHARE_NAME = re.compile(r"share|percent|pct|%|proportion|ratio|rate|fraction", re.IGNORECASE)


def _are_counts(series: pd.Series) -> bool:
    """Whole, non-negative numbers in a column not named as a share: counts, not percentages."""
    values = series.dropna()
    return (len(values) > 0 and not _SHARE_NAME.search(str(series.name or ""))
            and bool((values >= 0).all()) and bool(np.allclose(values, np.round(values))))


def _long_labels(labels, limit=10):
    return max((len(label) for label in labels), default=0) > limit


# ---------------------------------------------------------------------------
# Chart functions (what generated code calls as ceew.<name>)
# ---------------------------------------------------------------------------

# Other names Gemini uses for the arguments (often matplotlib's or seaborn's). They are mapped to ours,
# so a chart is not lost to a retry over a keyword. Anything else is set aside with a note (_flexible).
_ONE_VALUE = {"x": "category", "y": "value", "categories": "category", "labels": "category", "label": "category",
              "names": "category", "by": "category", "values": "value", "column": "value"}
ARGUMENT_ALIASES = {
    "bar": _ONE_VALUE, "lollipop": _ONE_VALUE, "diverging_bar": _ONE_VALUE, "pie": _ONE_VALUE,
    "donut": _ONE_VALUE,
    "grouped_bar": {"x": "category", "y": "values", "by": "category", "series": "values", "columns": "values"},
    "stacked_bar": {"x": "category", "y": "parts", "by": "category", "values": "parts", "columns": "parts",
                    "series": "parts"},
    "line": {"category": "x", "value": "y", "values": "y", "columns": "y", "hue": "series", "group": "series",
             "by": "series", "color": "series", "colour": "series"},
    "area": {"category": "x", "value": "y", "values": "y", "columns": "y"},
    "slope": {"x": "category", "before": "start", "after": "end"},
    "range_plot": {"x": "category", "before": "start", "after": "end"},
    "histogram": {"x": "value", "column": "value", "y": "value"},
    "boxplot": {"x": "group", "y": "value", "by": "group", "category": "group", "column": "value",
                "vert": "orientation"},
    "scatter": {"hue": "colour", "color": "colour", "group": "colour", "by": "colour"},
    "heatmap": {"x": "column", "y": "row", "columns": "column", "index": "row", "values": "value"},
}

# Other names for the chart functions themselves, with any arguments they imply.
FUNCTION_ALIASES = {
    "bar_chart": ("bar", {}), "barh": ("bar", {"orientation": "horizontal"}),
    "horizontal_bar": ("bar", {"orientation": "horizontal"}), "column": ("bar", {"orientation": "vertical"}),
    "column_chart": ("bar", {"orientation": "vertical"}), "vertical_bar": ("bar", {"orientation": "vertical"}),
    "line_chart": ("line", {}), "area_chart": ("area", {}), "pie_chart": ("pie", {}),
    "donut_chart": ("donut", {}), "doughnut": ("donut", {}), "box": ("boxplot", {}), "box_plot": ("boxplot", {}),
    "hist": ("histogram", {}), "scatter_plot": ("scatter", {}), "scatterplot": ("scatter", {}),
    "heat_map": ("heatmap", {}), "grouped_bar_chart": ("grouped_bar", {}), "stacked_bar_chart": ("stacked_bar", {}),
    "slope_chart": ("slope", {}), "dumbbell": ("range_plot", {}), "range": ("range_plot", {}),
}


def _flexible(method):
    """Accepts Gemini's other names for arguments, and sets aside arguments the rules do not use."""
    import functools
    import inspect

    params = list(inspect.signature(method).parameters)[1:]      # without self
    aliases = ARGUMENT_ALIASES.get(method.__name__, {})

    @functools.wraps(method)
    def wrapper(self, *args, **kwargs):
        given = set(params[:len(args)]) | set(kwargs)
        ignored = []
        for alias in [k for k in kwargs if k not in params]:
            value = kwargs.pop(alias)
            canonical = aliases.get(alias)
            if canonical and canonical not in given:
                kwargs[canonical] = value
                given.add(canonical)
            else:
                ignored.append(alias)
        outermost = not self.__dict__.get("_depth")
        if outermost:
            self._ignored = []
        self._ignored = (self.__dict__.get("_ignored") or []) + ignored
        self._depth = self.__dict__.get("_depth", 0) + 1
        try:
            return method(self, *args, **kwargs)
        finally:
            self._depth -= 1
            if outermost:
                self._ignored = []
    return wrapper


def _orientation(value, func):
    """'vertical' / 'horizontal' / 'auto' from the forms Gemini uses (v, h, vert=True ...)."""
    if value is True:
        return "vertical"
    if value is False:
        return "horizontal"
    text = str(value or "auto").strip().lower()
    if text in ("v", "vertical", "vert", "column", "columns", "up"):
        return "vertical"
    if text in ("h", "horizontal", "horiz", "bar", "barh", "sideways"):
        return "horizontal"
    if text == "auto":
        return "auto"
    raise ChartInputError(f"ceew.{func}: `orientation` must be 'auto', 'horizontal' or 'vertical'.")


def _wrap_label(label: str, width: int = 14) -> str:
    """Long category names on a horizontal axis go on several lines instead of being rotated."""
    return "\n".join(textwrap.wrap(str(label), width, break_long_words=False)) or str(label)


class ChartHandle:
    """What a chart function returns. Generated code sometimes treats it like a matplotlib figure or axes
    (fig = ceew.bar(...); ax.set_title(...); st.pyplot(fig)); every such call is accepted and does nothing,
    because the app draws the chart itself."""

    def __init__(self, kind: str, data=None):
        self.kind = kind
        self.data = data         # the table the chart shows; used as the answer's table if result = ceew.bar(...)

    def __getattr__(self, name):
        if name.startswith("__"):
            raise AttributeError(name)
        return lambda *args, **kwargs: self

    def __call__(self, *args, **kwargs):
        return self

    def __iter__(self):                  # fig, ax = ceew.bar(...)
        return iter((self, self))

    def __repr__(self):
        return f"<CEEW {self.kind} chart: drawn by the app>"


class Charts:
    """The chart functions. `sink` receives each ChartSpec (the app stores it; a notebook draws it)."""

    def __init__(self, sink):
        self._sink = sink

    def __getattr__(self, name):
        if name.startswith("_"):
            raise AttributeError(name)
        if name in FUNCTION_ALIASES:                     # ceew.box_plot, ceew.bar_chart, ceew.column_chart ...
            target, preset = FUNCTION_ALIASES[name]
            method = getattr(self, target)
            return lambda *args, **kwargs: method(*args, **{**preset, **kwargs})
        raise AttributeError(f"ceew.{name} is not available. Chart functions: " + ", ".join(
            f"ceew.{n}" for n in CHART_FUNCTIONS))

    def _emit(self, spec: ChartSpec):
        ignored = self.__dict__.get("_ignored") or []
        if ignored:
            spec.note("Not used: " + ", ".join(ignored) + ". The chart rules set the titles, colours and sizes; "
                      "you write the words under the chart.")
            self._ignored = []
        self._sink(spec)
        return ChartHandle(spec.kind, spec.data)

    # --- Ranking and magnitude ------------------------------------------------------------------

    def bar(self, data, category=None, value=None, orientation="auto", sort=None, highlight=None, why=""):
        """Bar chart of one value per category. Horizontal by default; vertical for time periods."""
        func = "bar"
        orientation = _orientation(orientation, func)
        if isinstance(value, (list, tuple)):
            if len(value) > 1:      # several value columns: that is a grouped bar chart
                return self.grouped_bar(data, category, list(value), orientation=orientation, sort=sort, why=why)
            value = value[0] if value else None
        frame = _frame(data, func)
        category = _column(frame, category if category is not None else frame.columns[0], func, "category")
        value = _column(frame, value if value is not None else frame.columns[-1], func, "value")
        _numeric(frame, [value], func)
        _unique(frame, category, func)
        frame = frame[[category, value]].copy()
        frame[category] = as_dates(frame[category])
        spec = ChartSpec("bar", frame, why=str(why or ""))
        time = is_time(frame[category])
        n = len(frame)
        if n == 0:
            return self._emit(_table(frame, spec, "There is nothing to plot: the table is empty."))
        if n > MAX_BARS:
            if time:
                spec.note(f"{n} time periods are too many for bars (at most {MAX_BARS}), so this is a line chart.",
                          "fallback")
                return self.line(frame, x=category, y=value, why=why, _spec=spec)
            return self._emit(_table(frame, spec, (
                f"A bar chart holds at most {MAX_BARS} categories; this table has {n}. Narrow the question to the "
                f"categories you need (for example the top {MAX_BARS}, if that suits your story), or use the table.")))
        frame = _sort(frame, category, value, sort, spec, func)
        labels = labels_of(frame[category])
        if orientation == "auto":
            orientation = "vertical" if time else "horizontal"
        if orientation == "vertical" and not time and _long_labels(labels):
            spec.note("Category names are long, so the bars are horizontal (the chart rules avoid rotated labels).")
            orientation = "horizontal"
        if (frame[value] < 0).any() and (frame[value] > 0).any():
            spec.note("Some values are negative. If they show movement away from a reference point, ask for a "
                      "diverging bar chart, which colours the two directions differently.")
        spec.data = frame
        spec.params = {"category": category, "value": value, "orientation": orientation, "labels": labels,
                       "highlight": _match_highlight(labels, highlight, spec),
                       "value_axes": ("x_title",) if orientation == "horizontal" else ("y_title",)}
        _offer_grey_out(n, highlight, spec)
        if orientation == "horizontal":
            spec.x_title = str(value)
        else:
            spec.y_title = str(value)
        return self._emit(spec)

    def lollipop(self, data, category=None, value=None, sort=None, highlight=None, why=""):
        """Lollipop chart: a thin line and a dot per category, always from zero."""
        func = "lollipop"
        frame = _frame(data, func)
        category = _column(frame, category if category is not None else frame.columns[0], func, "category")
        value = _column(frame, value if value is not None else frame.columns[-1], func, "value")
        _numeric(frame, [value], func)
        _unique(frame, category, func)
        frame = frame[[category, value]].copy()
        spec = ChartSpec("lollipop", frame, why=str(why or ""))
        if len(frame) > MAX_BARS:
            return self._emit(_table(frame, spec, f"A lollipop chart holds at most {MAX_BARS} categories; this "
                                     f"table has {len(frame)}. Narrow the question, or use the table."))
        frame = _sort(frame, category, value, sort, spec, func)
        labels = labels_of(frame[category])
        spec.data = frame
        spec.params = {"category": category, "value": value, "labels": labels,
                       "highlight": _match_highlight(labels, highlight, spec), "value_axes": ("x_title",)}
        spec.x_title = str(value)
        return self._emit(spec)

    def grouped_bar(self, data, category=None, values=None, orientation="auto", sort=None, why=""):
        """Grouped bars: two to four series (vertical) or two or three (horizontal) side by side."""
        func = "grouped_bar"
        orientation = _orientation(orientation, func)
        frame = _frame(data, func)
        category = _column(frame, category if category is not None else frame.columns[0], func, "category")
        if values is None:
            values = [c for c in frame.columns if c != category]
        values = _columns(frame, values, func, "values")
        _numeric(frame, values, func)
        _unique(frame, category, func)
        frame = frame[[category] + values].copy()
        frame[category] = as_dates(frame[category])
        spec = ChartSpec("grouped_bar", frame, why=str(why or ""))
        if len(values) == 1:
            return self.bar(frame, category, values[0], orientation=orientation, sort=sort, why=why)
        if is_time(frame[category]) and (len(frame) > MAX_BARS or len(values) > 4):
            spec.note(f"{len(frame)} time periods and {len(values)} series are too many for grouped bars, so this "
                      "is a line chart with one line per series.", "fallback")
            return self.line(frame, x=category, y=values, why=why, _spec=spec)
        if len(values) > 4:
            return self._emit(_table(frame, spec, f"Grouped bars hold two to four series; this table has "
                                     f"{len(values)}. Ask for fewer series, or for one chart per series."))
        if len(frame) > MAX_BARS:
            return self._emit(_table(frame, spec, f"A bar chart holds at most {MAX_BARS} categories; this table "
                                     f"has {len(frame)}. Narrow the question, or use the table."))
        frame = _sort(frame, category, values[0], sort, spec, func)
        labels = labels_of(frame[category])
        if orientation == "auto":
            orientation = "horizontal" if (_long_labels(labels) and len(values) <= 3) else "vertical"
        if orientation == "horizontal" and len(values) > 3:
            spec.note("Horizontal grouped bars hold at most three series, so these bars are vertical.")
            orientation = "vertical"
        if orientation == "vertical" and _long_labels(labels, 14):
            if len(values) <= 3:
                orientation = "horizontal"
                spec.note("Category names are long, so the bars are horizontal.")
            else:
                labels = [_wrap_label(label) for label in labels]
        spec.data = frame
        spec.params = {"category": category, "values": values, "orientation": orientation, "labels": labels,
                       "value_axes": ("x_title",) if orientation == "horizontal" else ("y_title",)}
        return self._emit(spec)

    def diverging_bar(self, data, category=None, value=None, reference="zero", above="blue", below="orange",
                      sort=None, why=""):
        """Bars to the left and right of a reference point (zero, a target or an average)."""
        func = "diverging_bar"
        frame = _frame(data, func)
        category = _column(frame, category if category is not None else frame.columns[0], func, "category")
        value = _column(frame, value if value is not None else frame.columns[-1], func, "value")
        _numeric(frame, [value], func)
        _unique(frame, category, func)
        frame = frame[[category, value]].copy()
        spec = ChartSpec("diverging_bar", frame, why=str(why or ""))
        if len(frame) > MAX_BARS:
            return self._emit(_table(frame, spec, f"A bar chart holds at most {MAX_BARS} categories; this table "
                                     f"has {len(frame)}. Narrow the question, or use the table."))
        if above not in SHADES or below not in SHADES or above == below:
            above, below = "blue", "orange"
        if below == "green" or above == "orange":
            spec.note("Green is never used for values below and orange never for values above (readers take green "
                      "as up and red as down), so the colours are blue above and orange below.")
            above, below = "blue", "orange"
        frame = _sort(frame, category, value, sort, spec, func)
        labels = labels_of(frame[category])
        spec.data = frame
        spec.params = {"category": category, "value": value, "labels": labels, "reference": str(reference),
                       "above": above, "below": below, "value_axes": ("x_title",)}
        spec.x_title = str(value)
        spec.note("The colours for values above and below the reference are your choice; to change them, say so in "
                  "your question (green may not mark values below, nor orange values above).")
        return self._emit(spec)

    # --- Part-to-whole ------------------------------------------------------------------------

    def stacked_bar(self, data, category=None, parts=None, orientation="auto", why=""):
        """Proportional stacked bars: each bar is one whole, split into parts that add up to 100%."""
        func = "stacked_bar"
        orientation = _orientation(orientation, func)
        frame = _frame(data, func)
        category = _column(frame, category if category is not None else frame.columns[0], func, "category")
        if parts is None:
            parts = [c for c in frame.columns if c != category]
        parts = _columns(frame, parts, func, "parts")
        _numeric(frame, parts, func)
        _unique(frame, category, func)
        frame = frame[[category] + parts].copy()
        frame[category] = as_dates(frame[category])
        spec = ChartSpec("stacked_bar", frame, why=str(why or ""))
        if len(frame) > MAX_BARS:
            return self._emit(_table(frame, spec, f"A bar chart holds at most {MAX_BARS} bars; this table has "
                                     f"{len(frame)}. Narrow the question, or use the table."))
        totals = frame[parts].sum(axis=1, min_count=1)
        if totals.between(0.99, 1.01).all() and (frame[parts] >= 0).all().all():
            frame[parts] = frame[parts] * 100
            totals = totals * 100
            spec.note("The parts were shares of 1; they are shown as percentages.")
        bad = ~totals.between(99, 101) | (frame[parts] < 0).any(axis=1)
        if bad.any():
            row = labels_of(frame.loc[bad, category])[0]
            message = (f"Stacked 100% bars need every bar to add up to 100% (rule R1, no double counting); "
                       f"'{row}' adds up to {totals[bad].iloc[0]:.0f}%. That happens, for example, when one "
                       "respondent can pick several answers.")
            if len(parts) <= 4:
                spec.note(message + " Drawn as grouped bars instead.", "fallback")
                return self._grouped_from(frame, category, parts, spec, why)
            return self._emit(_table(frame, spec, message + " Use the table, or supply parts of one whole."))
        time = is_time(frame[category])
        if time:
            frame = frame.sort_values(category, kind="stable")
        labels = labels_of(frame[category])
        if orientation == "auto":
            orientation = "vertical" if time and not _long_labels(labels) else "horizontal"
        part_labels = [str(p) for p in parts]
        colours, greyed = categorical_colours(part_labels, Style())
        if greyed:
            spec.note(f"More than {MAX_COLOURED} parts: {', '.join(greyed)} are shown in light grey.")
        spec.data = frame
        spec.params = {"category": category, "parts": parts, "orientation": orientation, "labels": labels,
                       "value_axes": ("x_title",) if orientation == "horizontal" else ("y_title",)}
        if orientation == "horizontal":
            spec.x_title = "Share (%)"
        else:
            spec.y_title = "Share (%)"
        spec.note("The axis title 'Share (%)' is a placeholder: say what the parts are shares of in the axis "
                  "title box below the chart.")
        return self._emit(spec)

    def _grouped_from(self, frame, category, parts, spec, why):
        labels = labels_of(frame[category])
        spec.kind = "grouped_bar"
        spec.data = frame
        orientation = "horizontal" if (_long_labels(labels) and len(parts) <= 3) else "vertical"
        spec.params = {"category": category, "values": parts, "orientation": orientation, "labels": labels,
                       "value_axes": ("x_title",) if orientation == "horizontal" else ("y_title",)}
        spec.why = str(why or "")
        return self._emit(spec)

    def pie(self, data, category=None, value=None, donut=False, why=""):
        """Pie or donut chart: at most five parts of one whole that add up to 100%."""
        func = "pie"
        frame = _frame(data, func)
        category = _column(frame, category if category is not None else frame.columns[0], func, "category")
        value = _column(frame, value if value is not None else frame.columns[-1], func, "value")
        _numeric(frame, [value], func)
        _unique(frame, category, func)
        frame = frame[[category, value]].copy()
        spec = ChartSpec("pie", frame, why=str(why or ""))
        total = frame[value].sum()
        if 0.99 <= total <= 1.01 and (frame[value] >= 0).all():
            frame[value] = frame[value] * 100
            total *= 100
        elif _are_counts(frame[value]) and not 99 <= total <= 101 and total > 0:
            # Counts of one whole (households per income band): a pie shows them as shares of the total.
            # Shares that add up to more than 100% are refused below; counts cannot show double counting,
            # so the note asks the researcher to confirm each unit is counted once.
            frame[value] = frame[value] / total * 100
            total = 100.0
            spec.note("The slices are shares of the total count. A pie is right only if each unit (a household, "
                      "a person) is counted once (rule R1); with multiple answers per unit, ask for a bar chart.")
        problem = None
        if (frame[value] < 0).any():
            problem = "a pie chart cannot show negative values"
        elif not 99 <= total <= 101:
            problem = (f"the parts add up to {total:,.0f}%, not 100% (rule R1: parts of one whole only); shares "
                       "that add up to more than 100% usually mean one unit was counted more than once")
        elif len(frame) > MAX_PIE_SLICES:
            problem = f"a pie chart holds at most {MAX_PIE_SLICES} slices; this has {len(frame)}"
        if problem:
            spec.note(f"Not drawn as a pie: {problem}. Drawn as a horizontal bar chart instead.", "fallback")
            return self._bar_from(frame, category, value, spec)
        others = frame[category].map(is_other)
        frame = pd.concat([frame[~others].sort_values(value, ascending=False, kind="stable"), frame[others]])
        spec.data = frame
        spec.params = {"category": category, "value": value, "labels": labels_of(frame[category]),
                       "donut": bool(donut)}
        return self._emit(spec)

    def donut(self, data, category=None, value=None, why=""):
        """Donut chart: the same rules as the pie chart."""
        return self.pie(data, category, value, donut=True, why=why)

    def _bar_from(self, frame, category, value, spec):
        if len(frame) > MAX_BARS:
            return self._emit(_table(frame, spec, f"A bar chart holds at most {MAX_BARS} categories; this table "
                                     f"has {len(frame)}. Narrow the question, or use the table."))
        frame = frame.sort_values(value, ascending=False, kind="stable")
        labels = labels_of(frame[category])
        spec.kind = "bar"
        spec.data = frame
        spec.params = {"category": category, "value": value, "orientation": "horizontal", "labels": labels,
                       "highlight": [], "value_axes": ("x_title",)}
        spec.x_title = str(value)
        return self._emit(spec)

    # --- Change over time ---------------------------------------------------------------------

    def line(self, data, x=None, y=None, series=None, highlight=None, why="", _spec=None):
        """Line chart over time. Gaps in the data break the line; it never joins across them."""
        func = "line"
        frame = _frame(data, func)
        x = _column(frame, x if x is not None else frame.columns[0], func, "x")
        spec = _spec or ChartSpec("line", frame, why=str(why or ""))
        spec.kind = "line"
        spec.why = spec.why or str(why or "")
        if series is not None:
            series = _column(frame, series, func, "series")
            y = _column(frame, y if not isinstance(y, (list, tuple)) else y[0], func, "y")
            _numeric(frame, [y], func)
            if frame.duplicated([x, series]).any():
                raise ChartInputError("ceew.line needs one row per x value and series. Aggregate the data first.")
            frame = frame.pivot(index=x, columns=series, values=y).reset_index()
            frame.columns = [str(c) if c != x else c for c in frame.columns]
            ys = [c for c in frame.columns if c != x]
            y_title = str(y)
        else:
            if y is None:
                y = [c for c in frame.columns if c != x]
            ys = _columns(frame, y, func, "y")
            _numeric(frame, ys, func)
            _unique(frame, x, func)
            frame = frame[[x] + ys].copy()
            y_title = str(ys[0]) if len(ys) == 1 else ""
        xs = as_dates(frame[x])
        if not (pd.api.types.is_datetime64_any_dtype(xs) or pd.api.types.is_numeric_dtype(xs)):
            spec.note("The x-axis is not a date or a number, so the points are evenly spaced in the order given. "
                      "For real time spacing, convert the column to dates in the code.")
            frame = frame.reset_index(drop=True)
        else:
            frame[x] = xs
            frame = frame.sort_values(x, kind="stable")
            frame, gaps = _break_gaps(frame, x)
            if gaps:
                spec.note(f"The data has {gaps} gap(s) in time; the line breaks there instead of joining across.")
        labels = [str(c) for c in ys]
        if len(ys) > MAX_COLOURED:
            spec.note(f"{len(ys)} lines: more than {MAX_COLOURED}, so the extra ones are light grey.")
        _offer_grey_out(len(ys), highlight, spec)
        spec.data = frame
        spec.params = {"x": x, "ys": ys, "labels": labels, "highlight": _match_highlight(labels, highlight, spec),
                       "value_axes": ("y_title",)}
        spec.y_title = y_title
        return self._emit(spec)

    def area(self, data, x=None, y=None, why=""):
        """Area chart: one series, or stacked series that are parts of one whole (largest at the bottom)."""
        func = "area"
        frame = _frame(data, func)
        x = _column(frame, x if x is not None else frame.columns[0], func, "x")
        if y is None:
            y = [c for c in frame.columns if c != x]
        ys = _columns(frame, y, func, "y")
        _numeric(frame, ys, func)
        _unique(frame, x, func)
        frame = frame[[x] + ys].copy()
        spec = ChartSpec("area", frame, why=str(why or ""))
        if frame[ys].isna().any().any() or (frame[ys] < 0).any().any():
            spec.note("Area charts cannot show missing or negative values, so this is a line chart.", "fallback")
            return self.line(frame, x=x, y=ys, why=why, _spec=spec)
        frame[x] = as_dates(frame[x])
        frame = frame.sort_values(x, kind="stable")
        order = frame[ys].sum().sort_values(ascending=False).index.tolist()
        if len(ys) > 1:
            spec.note("Stacked areas must be parts of one whole (rule R1: one unit counted once). The largest "
                      "series is at the bottom.")
        spec.data = frame
        spec.params = {"x": x, "ys": order, "labels": [str(c) for c in order], "value_axes": ("y_title",)}
        spec.y_title = str(ys[0]) if len(ys) == 1 else ""
        return self._emit(spec)

    def slope(self, data, category=None, start=None, end=None, why=""):
        """Slope chart: each category's value at two points, joined by a line. At most four categories."""
        func = "slope"
        frame = _frame(data, func)
        category = _column(frame, category if category is not None else frame.columns[0], func, "category")
        start = _column(frame, start if start is not None else frame.columns[1], func, "start")
        end = _column(frame, end if end is not None else frame.columns[2], func, "end")
        _numeric(frame, [start, end], func)
        _unique(frame, category, func)
        frame = frame[[category, start, end]].copy()
        spec = ChartSpec("slope", frame, why=str(why or ""))
        if len(frame) > MAX_SLOPE_CATEGORIES:
            spec.note(f"Slope charts hold at most {MAX_SLOPE_CATEGORIES} categories when comparing values; this has "
                      f"{len(frame)}, so it is drawn as a range plot, which shows the change for many categories.",
                      "fallback")
            return self.range_plot(frame, category, start, end, why=why, _spec=spec)
        spec.params = {"category": category, "start": start, "end": end, "labels": labels_of(frame[category]),
                       "value_axes": ("y_title",)}
        spec.note("Both columns must use the same unit. Add a y-axis title with the unit below the chart.")
        return self._emit(spec)

    # --- Distribution and deviation -----------------------------------------------------------

    def range_plot(self, data, category=None, start=None, end=None, sort=None, why="", _spec=None):
        """Range plot: two values per category (for example two years), joined by a line."""
        func = "range_plot"
        frame = _frame(data, func)
        category = _column(frame, category if category is not None else frame.columns[0], func, "category")
        start = _column(frame, start if start is not None else frame.columns[1], func, "start")
        end = _column(frame, end if end is not None else frame.columns[2], func, "end")
        _numeric(frame, [start, end], func)
        _unique(frame, category, func)
        frame = frame[[category, start, end]].copy()
        spec = _spec or ChartSpec("range_plot", frame, why=str(why or ""))
        spec.kind = "range_plot"
        if len(frame) > MAX_BARS:
            return self._emit(_table(frame, spec, f"A range plot holds at most {MAX_BARS} categories here; this "
                                     f"table has {len(frame)}. Narrow the question, or use the table."))
        frame = _sort(frame, category, end, sort, spec, func)
        spec.data = frame
        spec.params = {"category": category, "start": start, "end": end, "labels": labels_of(frame[category]),
                       "value_axes": ("x_title",)}
        return self._emit(spec)

    def histogram(self, data, value=None, bin_width=None, why=""):
        """Histogram of one numeric column, one row per observation. Equal bin widths."""
        func = "histogram"
        frame = _frame(data, func)
        value = _column(frame, value if value is not None else frame.columns[-1], func, "value")
        _numeric(frame, [value], func)
        values = frame[value].dropna().to_numpy(dtype=float)
        spec = ChartSpec("histogram", frame, why=str(why or ""))
        if len(values) == 0:
            return self._emit(_table(frame[[value]], spec, "There are no values to plot."))
        low, high = float(values.min()), float(values.max())
        if bin_width is None:
            if high > low:
                edges = np.histogram_bin_edges(values, bins="auto")
                width = _nice(float(edges[1] - edges[0]))
            else:
                width = 1.0
            spec.note(f"Bin width {_fmt(width)} was chosen automatically. Bins are your decision: to set your own, "
                      "give a bin width in your question (for example 'bins of 5'), and state it in the axis title "
                      "or the note.")
        else:
            width = float(bin_width)
            if not width > 0:
                raise ChartInputError("ceew.histogram: `bin_width` must be a positive number.")
        start = math.floor(low / width) * width
        count = max(1, int(math.ceil((high - start) / width + 1e-9)))
        if high >= start + count * width:
            count += 1
        if count > 200:
            raise ChartInputError("ceew.histogram: the bin width gives more than 200 bins; use a larger `bin_width`.")
        edges = start + width * np.arange(count + 1)
        counts, _ = np.histogram(values, bins=edges)
        bins = pd.DataFrame({"bin_start": edges[:-1], "bin_end": edges[1:], "count": counts})
        spec.data = bins
        spec.params = {"value": value, "width": width, "n": int(len(values)), "value_axes": ("x_title", "y_title")}
        spec.x_title = str(value)
        spec.y_title = "Count"
        spec.note(f"n = {len(values):,} values. For survey data, give the sample size in the note.")
        return self._emit(spec)

    def boxplot(self, data, group=None, value=None, orientation="auto", why=""):
        """Boxplot of a numeric column, one box per group. For specialist readers only.

        orientation: 'vertical' (boxes standing up, groups along the bottom), 'horizontal', or 'auto'
        (vertical unless the group names are long).
        """
        func = "boxplot"
        orientation = _orientation(orientation, func)
        frame = _frame(data, func)
        value = _column(frame, value if value is not None else frame.columns[-1], func, "value")
        _numeric(frame, [value], func)
        if group is not None:
            group = _column(frame, group, func, "group")
            groups = [g for g in pd.unique(frame[group].dropna())]
            if is_time(frame[group]):
                groups = sorted(groups)
        else:
            groups = [None]
        spec = ChartSpec("boxplot", frame, why=str(why or ""))
        if len(groups) > MAX_BARS:
            return self._emit(_table(frame.groupby(group)[value].describe(), spec,
                                     f"A boxplot holds at most {MAX_BARS} groups here; this has {len(groups)}."))
        boxes, labels = [], []
        for g in groups:
            values = (frame[value] if g is None else frame.loc[frame[group] == g, value]).dropna().to_numpy(float)
            boxes.append(values)
            labels.append(str(value) if g is None else labels_of(pd.Series([g]))[0])
        summary = pd.DataFrame({"group": labels, "n": [len(b) for b in boxes],
                                "median": [np.median(b) if len(b) else np.nan for b in boxes]})
        spec.data = summary
        if orientation == "auto":
            orientation = "horizontal" if _long_labels(labels, 14) else "vertical"
        # Standing boxes with long group names: the names go on several lines, never at an angle.
        tick_labels = [_wrap_label(label) for label in labels] if orientation == "vertical" else labels
        spec.params = {"boxes": boxes, "labels": labels, "tick_labels": tick_labels, "value": value,
                       "orientation": orientation}
        spec.params["value_axes"] = ("x_title",) if spec.params["orientation"] == "horizontal" else ("y_title",)
        if spec.params["orientation"] == "horizontal":
            spec.x_title = str(value)
        else:
            spec.y_title = str(value)
        spec.note("Boxplots are for specialist readers. Add a 'How to read this chart' line in the note: the line "
                  "inside each box is the median, the box holds the middle half of the values, the whiskers reach "
                  "the furthest values within 1.5 times the box height, and dots are values beyond them.")
        return self._emit(spec)

    # --- Correlation --------------------------------------------------------------------------

    def scatter(self, data, x=None, y=None, colour=None, why=""):
        """Scatterplot of two numeric columns. Colour may show a category or a value. No trend line."""
        func = "scatter"
        frame = _frame(data, func)
        x = _column(frame, x if x is not None else frame.columns[0], func, "x")
        y = _column(frame, y if y is not None else frame.columns[1], func, "y")
        _numeric(frame, [x, y], func)
        spec = ChartSpec("scatter", frame, why=str(why or ""))
        if colour is not None and colour not in frame.columns:
            # color='red' and the like: the chart rules choose colours.
            spec.note("Not used: colour. The chart rules choose the colours; colour= can only name a column.")
            colour = None
        cols = [x, y] + ([colour] if colour is not None else [])
        frame = frame[list(dict.fromkeys(cols))].dropna(subset=[x, y]).copy()
        spec.data = frame
        mode = None
        if colour is not None:
            if pd.api.types.is_numeric_dtype(frame[colour]) and frame[colour].nunique() > MAX_COLOURED:
                mode = "value"
            else:
                mode = "category"
                n = frame[colour].nunique()
                if n > MAX_COLOURED:
                    spec.note(f"{n} colour groups: more than {MAX_COLOURED}, so the extra ones are light grey.")
        spec.params = {"x": x, "y": y, "colour": colour, "mode": mode, "value_axes": ("x_title", "y_title")}
        spec.x_title, spec.y_title = str(x), str(y)
        return self._emit(spec)

    def heatmap(self, data, row=None, column=None, value=None, why=""):
        """XY heatmap: two categorical columns and one value, in square cells.

        Also takes a wide table of numbers, one row per row label (for example df.corr()), when `column` and
        `value` are not given.
        """
        func = "heatmap"
        if column is None and value is None and isinstance(data, pd.DataFrame):
            wide = data.set_index(row) if (row is not None and row in data.columns) else data
            labels = [c for c in wide.columns if not pd.api.types.is_numeric_dtype(wide[c])]
            if len(labels) == 1 and row is None:
                wide = wide.set_index(labels[0])
                labels = []
            numbers = wide.select_dtypes("number")
            if not labels and numbers.shape[1] >= 2 and wide.index.is_unique:
                row_name = str(numbers.index.name or "row")
                column_name = "column" if row_name != "column" else "column_"
                long = (numbers.rename_axis(row_name).rename_axis(column_name, axis=1).stack()
                        .rename("value").reset_index())
                return self.heatmap(long, row_name, column_name, "value", why=why)
        frame = _frame(data, func)
        row = _column(frame, row if row is not None else frame.columns[0], func, "row")
        column = _column(frame, column if column is not None else frame.columns[1], func, "column")
        value = _column(frame, value if value is not None else frame.columns[-1], func, "value")
        _numeric(frame, [value], func)
        if frame.duplicated([row, column]).any():
            raise ChartInputError("ceew.heatmap needs one row per (row, column) pair. Aggregate the data first.")
        spec = ChartSpec("heatmap", frame[[row, column, value]].copy(), why=str(why or ""))
        grid = frame.pivot(index=row, columns=column, values=value)
        if len(grid.index) > 40 or len(grid.columns) > 40:
            return self._emit(_table(grid, spec, "A heatmap holds at most 40 rows and 40 columns here. Narrow the "
                                     "question, or use the table."))
        spec.params = {"grid": grid, "row": row, "column": column, "value": value,
                       "row_labels": labels_of(pd.Series(grid.index, name=row)),
                       "column_labels": labels_of(pd.Series(grid.columns, name=column))}
        spec.note(f"Colours show '{value}' in five equal steps; the unit belongs in the note.")
        return self._emit(spec)


CHART_FUNCTIONS = ("bar", "lollipop", "grouped_bar", "diverging_bar", "stacked_bar", "pie", "donut", "line",
                   "area", "slope", "range_plot", "histogram", "boxplot", "scatter", "heatmap")

for _name in CHART_FUNCTIONS:
    setattr(Charts, _name, _flexible(getattr(Charts, _name)))


def _break_gaps(frame, x):
    """Inserts empty rows where the time series skips a step, so lines break instead of joining."""
    values = frame[x]
    if len(values) < 3:
        return frame, 0
    if pd.api.types.is_datetime64_any_dtype(values):
        steps = values.diff().dropna().dt.total_seconds().to_numpy()
    else:
        steps = values.diff().dropna().to_numpy(dtype=float)
    steps = steps[steps > 0]
    if len(steps) == 0:
        return frame, 0
    usual = float(np.median(steps))
    # Calendar months and years vary in length, so allow some slack before calling it a gap.
    big = np.where(steps > usual * 1.6)[0]
    if len(big) == 0:
        return frame, 0
    frame = frame.reset_index(drop=True)
    pieces, last = [], 0
    for position in big:
        cut = position + 1
        pieces.append(frame.iloc[last:cut])
        gap_x = values.iloc[position] + (values.iloc[cut] - values.iloc[position]) / 2
        filler = pd.DataFrame({c: [gap_x] if c == x else [np.nan] for c in frame.columns})
        pieces.append(filler.astype({x: frame[x].dtype}))
        last = cut
    pieces.append(frame.iloc[last:])
    return pd.concat(pieces, ignore_index=True), len(big)


# ---------------------------------------------------------------------------
# Numbers
# ---------------------------------------------------------------------------

def _nice(value: float) -> float:
    """1, 2, 2.5 or 5 times a power of ten, at least `value`."""
    if value <= 0 or not math.isfinite(value):
        return 1.0
    power = 10 ** math.floor(math.log10(value))
    for step in (1, 2, 2.5, 5, 10):
        if step * power >= value * 0.999:
            return step * power
    return 10 * power


def _fmt(value: float, decimals: int | None = None) -> str:
    """Numbers as CEEW charts show them: thousands separators from 10,000, a true minus sign."""
    value = float(value)
    if decimals is None:
        text = f"{value:,.0f}" if value.is_integer() else f"{value:,.2f}".rstrip("0").rstrip(".")
    else:
        text = f"{value:,.{decimals}f}"
    if abs(value) < 10000:
        text = text.replace(",", "")
    return text.replace("-", "−")


def _value_formatter(axis=None):
    """Tick labels with as many decimals as the tick step needs, worked out once per axis."""
    from matplotlib.ticker import Formatter

    class ValueFormatter(Formatter):
        def format_ticks(self, values):
            steps = np.diff(np.sort(np.asarray(values, dtype=float)))
            steps = steps[steps > 0]
            step = float(steps.min()) if len(steps) else 1.0
            decimals = max(0, -int(math.floor(math.log10(step) + 1e-9)))
            return [_fmt(0.0 if abs(v) < step / 1e6 else v, decimals) for v in values]

        def __call__(self, value, pos=None):
            return _fmt(value)

    return ValueFormatter()


def _classes(values, diverging: bool):
    """Five equal colour classes with round bounds."""
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return np.array([0, 1]), [GREY_LIGHTEST]
    low, high = float(values.min()), float(values.max())
    if diverging:
        reach = max(abs(low), abs(high))
        step = _nice(reach / 2.5)
        while 2.5 * step < reach:
            step = _nice(step * 1.01)
        return step * np.array([-2.5, -1.5, -0.5, 0.5, 1.5, 2.5]), None
    if high == low:
        high = low + 1
    step = _nice((high - low) / 5)
    while True:
        start = math.floor(low / step) * step
        count = int(math.ceil((high - start) / step - 1e-9))
        if count <= 5:
            break
        step = _nice(step * 1.01)
    count = max(count, 1)
    return start + step * np.arange(count + 1), None


# ---------------------------------------------------------------------------
# Drawing
# ---------------------------------------------------------------------------

def _canvas(spec: ChartSpec, style: Style):
    """(width in inches, chart-area height or None for a fixed canvas, total height, font size in points)."""
    items = len(spec.params.get("labels", []) or spec.params.get("row_labels", []) or [])
    labels = spec.params.get("labels") or []
    busy = items > 10 or _long_labels(labels, 18)
    horizontal = spec.params.get("orientation") == "horizontal" or spec.kind in ("lollipop", "diverging_bar",
                                                                                 "range_plot")
    if style.use == "presentation":
        return 13.333 / 2, None, 7.5, 11 if busy else 14
    if style.use == "outreach":
        size = (24 if busy else 28) * 72 / 300           # 24 to 28 px on a 1080 px canvas at 300 dpi
        return 3.6, None, 4.5 if style.shape == "portrait" else 3.6, size
    # Research report: fits an A4 page; height follows the number of items.
    if horizontal:
        bars = items * (len(spec.params.get("values", [None])) if spec.kind == "grouped_bar" else 1)
        plot = min(7.5, max(2.0, 0.8 + 0.28 * bars))
    elif spec.kind == "heatmap":
        # Square cells: as wide as the columns allow (about 4.6 inches for the cells), at most 0.38 inches.
        cell = min(0.38, 4.6 / max(len(spec.params["column_labels"]), 1))
        row_need = _heatmap_rows(spec.params["row_labels"], 11)[1]
        plot = min(7.0, max(1.6, 0.9 + max(cell, row_need) * len(spec.params["row_labels"])))
    elif spec.kind == "pie":
        plot = 3.0
    else:
        plot = 3.4
    return 6.3, plot, None, 11


def _wrap(text, width_in, font_size):
    chars = max(20, int(width_in * 72 / (font_size * 0.5)))
    lines = []
    for paragraph in str(text).splitlines() or [""]:
        lines += textwrap.wrap(paragraph, chars) or [""]
    return lines


def _prefixed(text, word):
    text = (text or "").strip()
    if not text:
        return ""
    return text if text.lower().startswith(word.lower()) else f"{word}: {text}"


def footer_lines(spec, texts):
    source = _prefixed(texts.get("source"), "Source") or PLACEHOLDERS["source"]
    note_default = "[How to read this chart: ...] [Note: ...]" if spec.kind == "boxplot" else PLACEHOLDERS["note"]
    note = _prefixed(texts.get("note"), "Note") or note_default
    return source, note


def axis_titles(spec, texts):
    titles = {}
    for name in ("x_title", "y_title"):
        given = texts.get(name)
        title = (given if given is not None else getattr(spec, name)) or ""
        if not title.strip() and name in spec.value_axes:
            title = PLACEHOLDERS["axis"]
        titles[name] = title.strip()
    return titles


def render(spec: ChartSpec, style: Style | None = None, texts: dict | None = None, fmt: str = "png",
           dpi: int = 300) -> bytes:
    """Draws the chart in the house style and returns PNG (300 dpi) or SVG (editable text) bytes."""
    import matplotlib
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

    style = style or Style()
    texts = texts or {}
    width, plot_h, total_h, font_size = _canvas(spec, style)
    margin = max(0.12, font_size / 72 * 1.1)
    gap = font_size / 72 * 0.9
    line_h = font_size / 72 * 1.35

    with matplotlib.rc_context(house_rc(style, font_size)):
        fig = Figure(figsize=(width, total_h or 6), dpi=dpi)
        canvas = FigureCanvasAgg(fig)
        ax = fig.add_axes([0.1, 0.1, 0.8, 0.8])
        legend_items, ends = _DRAW[spec.kind](ax, spec, style, font_size)
        # Presentation and outreach: label up to eight lines directly instead of a legend.
        direct = bool(ends) and style.use != "report" and len(legend_items) <= MAX_COLOURED
        titles = axis_titles(spec, texts)
        if spec.kind not in ("pie",):
            ax.set_xlabel(titles["x_title"])
            ax.set_ylabel(titles["y_title"])

        headline = (texts.get("headline") or "").strip() or PLACEHOLDERS["headline"]
        head_lines = _wrap(headline, width - 2 * margin, font_size)
        source, note = footer_lines(spec, texts)
        foot_lines = _wrap(source, width - 2 * margin, font_size) + _wrap(note, width - 2 * margin, font_size)
        head_h = len(head_lines) * line_h
        foot_h = len(foot_lines) * line_h

        renderer = canvas.get_renderer()
        legend = None
        legend_h = 0.0
        if legend_items and not direct:
            legend, legend_h = _legend(fig, legend_items, width - 2 * margin, font_size, renderer)

        if total_h is None:
            total_h = margin + head_h + gap * 1.5 + plot_h + (legend_h + gap if legend else 0) + gap + foot_h + margin
            fig.set_size_inches(width, total_h)
            renderer = canvas.get_renderer()

        fig.text(margin / width, 1 - margin / total_h, "\n".join(head_lines), ha="left", va="top",
                 fontweight="semibold", linespacing=1.25)
        fig.text(margin / width, margin / total_h, "\n".join(foot_lines), ha="left", va="bottom", linespacing=1.25)
        bottom = margin + foot_h + gap
        if legend is not None:
            legend.set_bbox_to_anchor((margin / width, bottom / total_h), transform=fig.transFigure)
            bottom += legend_h + gap
        top = total_h - margin - head_h - gap * 1.5
        zone = (margin / width, bottom / total_h, 1 - margin / width, top / total_h)
        _fit(fig, ax, zone, renderer)
        if direct:
            _direct_labels(ax, legend_items, ends, font_size, fig)
            _fit(fig, ax, zone, renderer)
            _direct_labels(ax, legend_items, ends, font_size, fig)

        buffer = io.BytesIO()
        if fmt == "svg":
            fig.savefig(buffer, format="svg")
        else:
            fig.savefig(buffer, format="png", dpi=dpi)
    return buffer.getvalue()


def _legend(fig, items, max_width, font_size, renderer):
    from matplotlib.patches import Patch
    handles = [Patch(facecolor=colour, edgecolor="none", label=label) for label, colour in items]
    ncol = len(handles)
    while True:
        legend = fig.legend(handles=handles, loc="lower left", ncol=ncol, handlelength=0.9, handleheight=0.9,
                            handletextpad=0.5, columnspacing=1.4, borderaxespad=0, borderpad=0,
                            labelspacing=0.5, alignment="left")
        box = legend.get_window_extent(renderer)
        if box.width / fig.dpi <= max_width or ncol == 1:
            return legend, box.height / fig.dpi
        legend.remove()
        ncol -= 1


def _fit(fig, ax, zone, renderer):
    """Moves the plot so its tick labels and axis titles fit inside the zone."""
    left, bottom, right, top = zone
    for _ in range(3):
        fig_w, fig_h = fig.get_size_inches() * fig.dpi
        tight = ax.get_tightbbox(renderer)
        box = ax.get_position()
        pos = ax.get_window_extent(renderer)
        extra_left = (pos.x0 - tight.x0) / fig_w
        extra_right = (tight.x1 - pos.x1) / fig_w
        extra_bottom = (pos.y0 - tight.y0) / fig_h
        extra_top = (tight.y1 - pos.y1) / fig_h
        x0 = left + max(0.0, extra_left)
        x1 = right - max(0.0, extra_right)
        y0 = bottom + max(0.0, extra_bottom)
        y1 = top - max(0.0, extra_top)
        new = [x0, y0, max(0.05, x1 - x0), max(0.05, y1 - y0)]
        if np.allclose(new, [box.x0, box.y0, box.width, box.height], atol=1e-4):
            break
        ax.set_position(new)


def _direct_labels(ax, items, mode, font_size, fig):
    """Labels at the right end of each line instead of a legend (presentation and outreach)."""
    for text in list(getattr(ax, "_ceew_labels", [])):
        text.remove()
    ax._ceew_labels = []
    ends = [(label, y) for (label, _colour), y in zip(items, mode) if y is not None and np.isfinite(y)]
    if not ends:
        return
    low, high = ax.get_ylim()
    height_in = ax.get_position().height * fig.get_size_inches()[1]
    min_gap = (font_size * 1.2 / 72) / max(height_in, 0.1) * (high - low)
    ends.sort(key=lambda item: item[1])
    placed = []
    for label, y in ends:
        if placed and y - placed[-1][1] < min_gap:
            y = placed[-1][1] + min_gap
        placed.append((label, y))
    x_end = ax.get_xlim()[1]
    for label, y in placed:
        text = ax.annotate(label, xy=(x_end, y), xytext=(6, 0), textcoords="offset points", va="center",
                           ha="left", annotation_clip=False, color=TEXT_COLOUR)
        ax._ceew_labels.append(text)


def _category_axis(ax, horizontal: bool):
    """Category axis line shown, value axis line hidden, gridlines along the value axis."""
    if horizontal:
        ax.spines["bottom"].set_visible(False)
        ax.xaxis.grid(True)
        ax.tick_params(axis="y", length=0, pad=6)
        ax.tick_params(axis="x", length=0, pad=4)
        ax.xaxis.set_major_formatter(_value_formatter(ax.xaxis))
    else:
        ax.spines["left"].set_visible(False)
        ax.yaxis.grid(True)
        ax.tick_params(axis="x", length=0, pad=6)
        ax.tick_params(axis="y", length=0, pad=4)
        ax.yaxis.set_major_formatter(_value_formatter(ax.yaxis))


def _zero_baseline(ax, values, horizontal):
    """Bars start at zero (rule R2)."""
    low = min(0.0, float(np.nanmin(values))) if len(values) else 0.0
    high = max(0.0, float(np.nanmax(values))) if len(values) else 1.0
    pad = (high - low) * 0.04 or 1
    lims = (low - (pad if low < 0 else 0), high + (pad if high > 0 else 0))
    (ax.set_xlim if horizontal else ax.set_ylim)(*lims)


def _draw_bar(ax, spec, style, font_size):
    p = spec.params
    values = spec.data[p["value"]].to_numpy(dtype=float)
    p1 = SHADES[style.primaries[0]]["base"]
    if p["highlight"]:
        colours = [p1 if label in p["highlight"] else GREY_LIGHTEST for label in p["labels"]]
    else:
        colours = [p1] * len(values)
    horizontal = p["orientation"] == "horizontal"
    positions = np.arange(len(values))
    if horizontal:
        positions = positions[::-1]
        ax.barh(positions, values, height=0.68, color=colours)
        ax.set_yticks(positions, p["labels"])
        ax.set_ylim(-0.6, len(values) - 0.4)
    else:
        ax.bar(positions, values, width=0.68, color=colours)
        ax.set_xticks(positions, p["labels"])
        ax.set_xlim(-0.6, len(values) - 0.4)
    _category_axis(ax, horizontal)
    _zero_baseline(ax, values, horizontal)
    _thin_time_labels(ax, spec, horizontal)
    return [], None


def _thin_time_labels(ax, spec, horizontal):
    """Year labels may be thinned, but only in regular steps of 2 or 5."""
    if horizontal or not is_time(spec.data[spec.params["category"]]):
        return
    labels = spec.params["labels"]
    if len(labels) <= 10:
        return
    step = 2 if len(labels) <= 20 else 5
    for index, tick in enumerate(ax.get_xticklabels()):
        tick.set_visible(index % step == 0)


def _draw_lollipop(ax, spec, style, font_size):
    p = spec.params
    values = spec.data[p["value"]].to_numpy(dtype=float)
    p1 = SHADES[style.primaries[0]]["base"]
    colours = [p1 if not p["highlight"] or label in p["highlight"] else GREY_LIGHTEST for label in p["labels"]]
    positions = np.arange(len(values))[::-1]
    ax.hlines(positions, 0, values, colors=colours, linewidth=font_size / 11 * 1.6)
    ax.scatter(values, positions, s=(font_size * 0.75) ** 2, color=colours, zorder=3, linewidths=0)
    ax.set_yticks(positions, p["labels"])
    ax.set_ylim(-0.6, len(values) - 0.4)
    _category_axis(ax, True)
    _zero_baseline(ax, values, True)
    return [], None


def _draw_grouped_bar(ax, spec, style, font_size):
    p = spec.params
    series = p["values"]
    colours, _ = categorical_colours([str(s) for s in series], style)
    horizontal = p["orientation"] == "horizontal"
    count = len(series)
    width = 0.8 / count
    base = np.arange(len(spec.data))
    if horizontal:
        base = base[::-1]
    for index, (name, colour) in enumerate(zip(series, colours)):
        offset = (index - (count - 1) / 2) * width
        values = spec.data[name].to_numpy(dtype=float)
        if horizontal:
            ax.barh(base - offset, values, height=width * 0.95, color=colour)
        else:
            ax.bar(base + offset, values, width=width * 0.95, color=colour)
    if horizontal:
        ax.set_yticks(base, p["labels"])
    else:
        ax.set_xticks(base, p["labels"])
    _category_axis(ax, horizontal)
    _zero_baseline(ax, spec.data[series].to_numpy(dtype=float).ravel(), horizontal)
    return [(str(s), c) for s, c in zip(series, colours)], None


def _draw_stacked_bar(ax, spec, style, font_size):
    p = spec.params
    parts = p["parts"]
    colours, _ = categorical_colours([str(s) for s in parts], style)
    horizontal = p["orientation"] == "horizontal"
    base = np.arange(len(spec.data))
    if horizontal:
        base = base[::-1]
    running = np.zeros(len(spec.data))
    for name, colour in zip(parts, colours):
        values = np.nan_to_num(spec.data[name].to_numpy(dtype=float))
        if horizontal:
            ax.barh(base, values, left=running, height=0.68, color=colour, edgecolor=style.background, linewidth=0.6)
        else:
            ax.bar(base, values, bottom=running, width=0.68, color=colour, edgecolor=style.background, linewidth=0.6)
        running += values
    if horizontal:
        ax.set_yticks(base, p["labels"])
        ax.set_xlim(0, 100)
    else:
        ax.set_xticks(base, p["labels"])
        ax.set_ylim(0, 100)
    _category_axis(ax, horizontal)
    return [(str(s), c) for s, c in zip(parts, colours)], None


def _draw_diverging_bar(ax, spec, style, font_size):
    p = spec.params
    values = spec.data[p["value"]].to_numpy(dtype=float)
    above, below = SHADES[p["above"]]["base"], SHADES[p["below"]]["base"]
    colours = [above if v >= 0 else below for v in values]
    positions = np.arange(len(values))[::-1]
    ax.barh(positions, values, height=0.68, color=colours)
    ax.set_yticks(positions, p["labels"])
    ax.set_ylim(-0.6, len(values) - 0.4)
    ax.axvline(0, color=AXIS_COLOUR, linewidth=0.9, zorder=3)
    ax.spines["left"].set_visible(False)
    _category_axis(ax, True)
    _zero_baseline(ax, values, True)
    reference = p["reference"]
    words = "zero" if reference.lower() in ("0", "zero") else reference
    return [(f"Above {words}", above), (f"Below {words}", below)], None


def _time_axis(ax, xs):
    """Real time spacing; tick labels without rotation, years thinned in steps of 1, 2, 5 or 10."""
    import matplotlib.dates as mdates
    from matplotlib.ticker import MaxNLocator
    if pd.api.types.is_datetime64_any_dtype(xs):
        span_days = (xs.max() - xs.min()).days if len(xs) else 0
        if span_days >= 3 * 365:
            years = span_days / 365.25
            step = next((s for s in (1, 2, 5, 10, 20, 50) if years / s <= 8), 100)
            ax.xaxis.set_major_locator(mdates.YearLocator(step))
            ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
        elif span_days >= 60:
            months = span_days / 30.4
            step = next((s for s in (1, 2, 3, 6) if months / s <= 6), 12)
            ax.xaxis.set_major_locator(mdates.MonthLocator(interval=step))
            ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
        else:
            ax.xaxis.set_major_locator(mdates.AutoDateLocator(minticks=3, maxticks=6))
            ax.xaxis.set_major_formatter(mdates.DateFormatter("%d %b"))
    elif is_time(xs):
        # Years: plain numbers (no thousands separator), in steps of 1, 2, 5 or 10.
        from matplotlib.ticker import FormatStrFormatter
        ax.xaxis.set_major_locator(MaxNLocator(nbins=8, integer=True, steps=[1, 2, 5, 10]))
        ax.xaxis.set_major_formatter(FormatStrFormatter("%d"))
    else:
        ax.xaxis.set_major_formatter(_value_formatter(ax.xaxis))


def _line_axes(ax):
    ax.spines["left"].set_visible(False)
    ax.yaxis.grid(True)
    ax.tick_params(axis="y", length=0, pad=4)
    ax.tick_params(axis="x", length=3, pad=4)
    ax.yaxis.set_major_formatter(_value_formatter(ax.yaxis))


def _draw_line(ax, spec, style, font_size):
    p = spec.params
    frame = spec.data
    xs = frame[p["x"]]
    numeric_x = pd.api.types.is_datetime64_any_dtype(xs) or pd.api.types.is_numeric_dtype(xs)
    x_values = xs if numeric_x else np.arange(len(frame))
    labels = p["labels"]
    if p["highlight"]:
        palette = highlight_colours(style)
        colours = []
        for label in labels:
            colours.append(palette[p["highlight"].index(label) % len(palette)] if label in p["highlight"]
                           else GREY_LIGHTEST)
    else:
        colours, _ = categorical_colours(labels, style)
    width = font_size / 11 * 1.8
    ends = []
    for name, label, colour in zip(p["ys"], labels, colours):
        highlighted = not p["highlight"] or label in p["highlight"]
        ax.plot(x_values, frame[name].to_numpy(dtype=float), color=colour, linewidth=width,
                zorder=3 if highlighted else 2, solid_capstyle="round")
        valid = frame[name].dropna()
        ends.append(float(valid.iloc[-1]) if len(valid) else None)
    if numeric_x:
        _time_axis(ax, xs)
    else:
        # Text along the x-axis: at most about eight labels, at regular steps, never at an angle.
        step = next((s for s in (1, 2, 3, 4, 5, 6, 10, 12, 20, 25, 50) if len(x_values) / s <= 8), 100)
        ticks = np.arange(0, len(x_values), step)
        ax.set_xticks(ticks, [labels_of(xs)[i] for i in ticks])
    ax.margins(x=0.01)
    _line_axes(ax)
    items = [(label, colour) for label, colour in zip(labels, colours)]
    if len(items) == 1:
        return [], None
    if p["highlight"]:
        items = [(label, colour) for label, colour in items if label in p["highlight"]]
        ends = [e for e, label in zip(ends, labels) if label in p["highlight"]]
    return items, ends


def _draw_area(ax, spec, style, font_size):
    p = spec.params
    frame = spec.data
    xs = frame[p["x"]]
    colours, _ = categorical_colours(p["labels"], style)
    ax.stackplot(xs, [frame[c].to_numpy(dtype=float) for c in p["ys"]], colors=colours,
                 edgecolor=style.background, linewidth=1.0)
    if pd.api.types.is_datetime64_any_dtype(xs) or pd.api.types.is_numeric_dtype(xs):
        _time_axis(ax, xs)
    ax.set_ylim(bottom=0)
    ax.margins(x=0)
    _line_axes(ax)
    if len(p["ys"]) == 1:
        return [], None
    return [(label, colour) for label, colour in zip(p["labels"], colours)], None


def _draw_slope(ax, spec, style, font_size):
    p = spec.params
    colours, _ = categorical_colours(p["labels"], style)
    width = font_size / 11 * 1.8
    ends = []
    for (_, row), colour in zip(spec.data.iterrows(), colours):
        ax.plot([0, 1], [row[p["start"]], row[p["end"]]], color=colour, linewidth=width, marker="o",
                markersize=font_size * 0.45, zorder=3)
        ends.append(float(row[p["end"]]))
    ax.set_xticks([0, 1], [str(p["start"]), str(p["end"])])
    ax.set_xlim(-0.12, 1.12)
    _line_axes(ax)
    return [(label, colour) for label, colour in zip(p["labels"], colours)], ends


def _draw_range_plot(ax, spec, style, font_size):
    p = spec.params
    start = spec.data[p["start"]].to_numpy(dtype=float)
    end = spec.data[p["end"]].to_numpy(dtype=float)
    positions = np.arange(len(start))[::-1]
    end_colour = SHADES[style.primaries[0]]["base"]
    ax.hlines(positions, start, end, colors=GREY_LIGHTEST, linewidth=font_size / 11 * 3, zorder=2)
    size = (font_size * 0.8) ** 2
    ax.scatter(start, positions, s=size, color=GREY_LIGHT, zorder=3, linewidths=0)
    ax.scatter(end, positions, s=size, color=end_colour, zorder=4, linewidths=0)
    ax.set_yticks(positions, p["labels"])
    ax.set_ylim(-0.6, len(start) - 0.4)
    _category_axis(ax, True)
    return [(str(p["start"]), GREY_LIGHT), (str(p["end"]), end_colour)], None


def _draw_histogram(ax, spec, style, font_size):
    bins = spec.data
    colour = SHADES[style.primaries[0]]["base"]
    ax.bar(bins["bin_start"], bins["count"], width=spec.params["width"], align="edge", color=colour,
           edgecolor=style.background, linewidth=0.8)
    ax.set_xlim(bins["bin_start"].min(), bins["bin_end"].max())
    _line_axes(ax)
    ax.xaxis.set_major_formatter(_value_formatter(ax.xaxis))
    ax.set_ylim(bottom=0)
    return [], None


def _draw_boxplot(ax, spec, style, font_size):
    p = spec.params
    hue = style.primaries[0]
    horizontal = p["orientation"] == "horizontal"
    boxes = [b if len(b) else np.array([np.nan]) for b in p["boxes"]]
    positions = np.arange(len(boxes))[::-1] if horizontal else np.arange(len(boxes))
    options = dict(positions=positions, patch_artist=True, widths=0.5,
                   boxprops={"facecolor": SHADES[hue]["lightest"], "edgecolor": SHADES[hue]["dark"]},
                   medianprops={"color": SHADES[hue]["darkest"], "linewidth": 1.6},
                   whiskerprops={"color": GREY_LIGHT}, capprops={"color": GREY_LIGHT},
                   flierprops={"marker": "o", "markerfacecolor": GREY_LIGHT, "markeredgecolor": "none",
                               "markersize": font_size * 0.35})
    try:
        ax.boxplot(boxes, orientation="horizontal" if horizontal else "vertical", **options)
    except TypeError:        # matplotlib before 3.10
        ax.boxplot(boxes, vert=not horizontal, **options)
    if horizontal:
        ax.set_yticks(positions, p["labels"])
    else:
        ax.set_xticks(positions, p.get("tick_labels") or p["labels"])
    _category_axis(ax, horizontal)
    return [], None


def _draw_scatter(ax, spec, style, font_size):
    p = spec.params
    frame = spec.data
    size = (font_size * 0.5) ** 2
    items = []
    if p["mode"] == "category":
        groups = [g for g in pd.unique(frame[p["colour"]])]
        labels = labels_of(pd.Series(groups, name=p["colour"]))
        colours, _ = categorical_colours(labels, style)
        for group, label, colour in zip(groups, labels, colours):
            part = frame[frame[p["colour"]] == group] if not pd.isna(group) else frame[frame[p["colour"]].isna()]
            ax.scatter(part[p["x"]], part[p["y"]], s=size, color=colour, linewidths=0)
            items.append((label, colour))
    elif p["mode"] == "value":
        from matplotlib.colors import BoundaryNorm, ListedColormap
        values = frame[p["colour"]]
        bounds, _ = _classes(values, False)
        colours = sequential_colours(style)[-(len(bounds) - 1):] if len(bounds) - 1 <= 5 else sequential_colours(style)
        cmap = ListedColormap(colours)
        ax.scatter(frame[p["x"]], frame[p["y"]], c=values, cmap=cmap, norm=BoundaryNorm(bounds, cmap.N), s=size,
                   linewidths=0)
        items = [(f"{_fmt(lo)} to {_fmt(hi)}", c) for lo, hi, c in zip(bounds[:-1], bounds[1:], colours)]
    else:
        ax.scatter(frame[p["x"]], frame[p["y"]], s=size, color=SHADES[style.primaries[0]]["base"], linewidths=0)
    ax.grid(True)
    ax.tick_params(length=0, pad=4)
    ax.xaxis.set_major_formatter(_value_formatter(ax.xaxis))
    ax.yaxis.set_major_formatter(_value_formatter(ax.yaxis))
    return items, None


def _draw_pie(ax, spec, style, font_size):
    p = spec.params
    values = spec.data[p["value"]].to_numpy(dtype=float)
    colours, _ = categorical_colours(p["labels"], style)
    width = 0.38 if p["donut"] else None
    ax.pie(values, colors=colours, startangle=90, counterclock=False, labels=[f"{v:.0f}%" for v in values],
           labeldistance=1.14, wedgeprops={"edgecolor": style.background, "linewidth": 1.2,
                                           **({"width": width} if width else {})},
           textprops={"color": TEXT_COLOUR})
    ax.set_aspect("equal")
    return [(label, colour) for label, colour in zip(p["labels"], colours)], None


def _heatmap_rows(labels, font_size):
    """Row names wrapped onto at most a few lines, and the height (inches) one row needs to show them."""
    wrapped = [_wrap_label(label, 26) for label in labels]
    lines = max((label.count("\n") + 1 for label in wrapped), default=1)
    return wrapped, lines * font_size * 1.3 / 72


def _draw_heatmap(ax, spec, style, font_size):
    from matplotlib.colors import BoundaryNorm, ListedColormap
    p = spec.params
    grid = p["grid"].to_numpy(dtype=float)
    finite = grid[np.isfinite(grid)]
    diverging = len(finite) and finite.min() < 0 < finite.max()
    bounds, _ = _classes(finite, bool(diverging))
    if diverging:
        colours = DIVERGING_COLOURS
    else:
        colours = sequential_colours(style)[:len(bounds) - 1] if len(bounds) - 1 < 5 else sequential_colours(style)
    cmap = ListedColormap(colours)
    cmap.set_bad(style.background)
    ax.pcolormesh(np.ma.masked_invalid(grid), cmap=cmap, norm=BoundaryNorm(bounds, cmap.N),
                  edgecolors=style.background, linewidth=1.0 if grid.shape[1] > 12 else 1.5)
    ax.set_aspect("equal")
    ax.set_anchor("W")
    ax.invert_yaxis()
    # Long row names go on two or more lines, so the cells keep most of the width.
    row_labels, row_need = _heatmap_rows(p["row_labels"], font_size)
    column_labels = list(p["column_labels"])
    char_in = font_size * 0.55 / 72
    label_in = max(len(line) for label in row_labels for line in label.split("\n")) * char_in
    cell_in = max(0.05, (ax.figure.get_size_inches()[0] - 0.5 - label_in) / max(grid.shape[1], 1))
    if row_need > cell_in:
        # Many columns and long row names: square cells would be shorter than the names, which then overlap.
        # Taller cells keep every name readable (legibility comes before the square-cell build rule).
        ax.set_aspect("auto")
    column_in = max((len(label) for label in column_labels), default=1) * char_in
    ticks = np.arange(grid.shape[1])
    rotate = False
    if column_in > cell_in * 0.9:
        if all(label.strip().lstrip("-").isdigit() for label in column_labels):
            # Numbers such as hours: show every 2nd, 3rd ... label (regular steps), never at an angle.
            step = next((s for s in (2, 3, 4, 6, 12) if s * cell_in >= column_in * 1.4), 24)
            ticks = ticks[::step]
        else:
            rotate = True                                     # the rules allow 90 degrees only, never an angle
    ax.set_xticks(ticks + 0.5, [column_labels[i] for i in ticks])
    ax.set_yticks(np.arange(grid.shape[0]) + 0.5, row_labels)
    for side in ("left", "bottom", "top", "right"):
        ax.spines[side].set_visible(False)
    ax.tick_params(length=0, pad=4)
    if rotate:
        ax.tick_params(axis="x", labelrotation=90)
    return [(f"{_fmt(lo)} to {_fmt(hi)}", c) for lo, hi, c in zip(bounds[:-1], bounds[1:], colours)], None


_DRAW = {
    "bar": _draw_bar, "lollipop": _draw_lollipop, "grouped_bar": _draw_grouped_bar,
    "stacked_bar": _draw_stacked_bar, "diverging_bar": _draw_diverging_bar, "line": _draw_line,
    "area": _draw_area, "slope": _draw_slope, "range_plot": _draw_range_plot, "histogram": _draw_histogram,
    "boxplot": _draw_boxplot, "scatter": _draw_scatter, "pie": _draw_pie, "heatmap": _draw_heatmap,
}


# ---------------------------------------------------------------------------
# The researcher's words
# ---------------------------------------------------------------------------

_CAUSAL = re.compile(r"\b(drives?|drove|driven|driving|proves?|proved|proven|because of|due to|leads? to|led to|"
                     r"caus(?:e|es|ed|ing)|results? in|resulted in|thanks to|owing to|triggers?|triggered|"
                     r"boosts?|boosted|impacts?|impacted)\b", re.IGNORECASE)
_RECORD = re.compile(r"\b(ever|all[- ]time|record|unprecedented|never before|first time)\b", re.IGNORECASE)


def check_headline(*texts) -> list:
    """Words in the headline or note that may claim more than a chart can show.

    Only flags; never rewrites. The researcher decides what to change.
    """
    flags = []
    for text in texts:
        for match in _CAUSAL.finditer(text or ""):
            flags.append(f"“{match.group(0)}” states a cause. A chart shows that things go together or change "
                         "together, not why; check that your evidence supports the cause.")
        for match in _RECORD.finditer(text or ""):
            flags.append(f"“{match.group(0)}” claims a record. The chart covers only the period and items in "
                         "your data; check the claim holds beyond them.")
    seen, unique = set(), []
    for flag in flags:
        if flag.lower() not in seen:
            seen.add(flag.lower())
            unique.append(flag)
    return unique


def file_stem(spec: ChartSpec, texts: dict, fallback: str) -> str:
    """A file name from the chart's content (headline, or the axis titles), never from personal data."""
    source = (texts.get("headline") or "").strip()
    if not source:
        titles = [t for t in (texts.get("y_title") or spec.y_title, texts.get("x_title") or spec.x_title) if t]
        source = " by ".join(titles[:2])
    stem = re.sub(r"[^a-z0-9]+", "-", source.lower()).strip("-")[:60].strip("-")
    return stem or fallback


# ---------------------------------------------------------------------------
# Outside the app (exported notebooks)
# ---------------------------------------------------------------------------

def notebook_charts(use: str = "report", pair: str = "orange-blue", shape: str = "square"):
    """Chart functions for an exported notebook: each chart is drawn and shown straight away."""
    style = Style(use=use, shape=shape, pair=pair)
    if not fonts_available():
        print("Familjen Grotesk was not found, so charts use a stand-in font. Put the FamiljenGrotesk .ttf files "
              "in a 'fonts' folder next to this file for the CEEW house style.")

    def show(spec: ChartSpec):
        for _level, text in spec.notes:
            print("Note:", text)
        if spec.kind == "table":
            try:
                display(spec.data)  # noqa: F821  (Jupyter)
            except NameError:
                print(spec.data)
            return
        png = render(spec, style)
        try:
            from IPython.display import Image, display as ipy_display
            ipy_display(Image(data=png, width=640))
        except ImportError:
            with open(file_stem(spec, {}, "chart") + ".png", "wb") as handle:
                handle.write(png)
    return Charts(show)
