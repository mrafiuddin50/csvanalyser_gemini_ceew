"""CEEW charts: the data viz skill's rules, the house style, and how charts reach the page."""
import importlib.util
import sys

import numpy as np
import pandas as pd
import pytest

from csvanalyser import charts
from csvanalyser.charts import (GREY_LIGHT, GREY_LIGHTEST, SHADES, ChartInputError, Charts, Style, categorical_colours,
                                check_headline, file_stem, natural_order, render)
from csvanalyser.export import build_notebook, charts_module_source, uses_ceew_charts
from csvanalyser.prompts import build_prompt, describe_schema
from csvanalyser.sandbox import run_generated_code


def draw(method, *args, **kwargs):
    specs = []
    getattr(Charts(specs.append), method)(*args, **kwargs)
    assert len(specs) == 1
    return specs[0]


def notes(spec, level=None):
    return " ".join(text for lvl, text in spec.notes if level is None or lvl == level)


STATES = pd.DataFrame({"state": [f"State {c}" for c in "ABCDE"], "capacity_mw": [10.0, 50, 30, 20, 40]})


# --- Colour rules ------------------------------------------------------------------------------

def test_colours_use_two_primaries_then_grey_and_never_the_third():
    colours, greyed = categorical_colours(list("ABCDEFGHIJ"), Style(pair="orange-green"))
    assert colours[:3] == [SHADES["orange"]["base"], SHADES["green"]["base"], GREY_LIGHT]
    assert not set(SHADES["blue"].values()) & set(colours)            # the third primary is never used
    assert len(set(colours[:8])) == 8 and colours[8:] == [GREY_LIGHTEST] * 2 and greyed == ["I", "J"]


def test_other_and_not_available_are_always_grey():
    colours, _ = categorical_colours(["Solar", "Other", "Wind", "Not available"], Style())
    assert colours[1] == colours[3] == GREY_LIGHT
    assert GREY_LIGHT not in (colours[0], colours[2])


def test_diverging_bars_never_use_green_for_below_or_orange_for_above():
    data = pd.DataFrame({"state": list("ABC"), "change": [3.0, -2, 1]})
    spec = draw("diverging_bar", data, "state", "change", above="orange", below="green")
    assert (spec.params["above"], spec.params["below"]) == ("blue", "orange")
    assert "never" in notes(spec)


# --- Chart menu guardrails ---------------------------------------------------------------------

def test_bar_sorts_largest_first_and_says_so():
    spec = draw("bar", STATES, "state", "capacity_mw")
    assert spec.data["capacity_mw"].tolist() == [50, 40, 30, 20, 10]
    assert spec.params["orientation"] == "horizontal" and spec.x_title == "capacity_mw"
    assert "largest to smallest" in notes(spec)
    asked = draw("bar", STATES, "state", "capacity_mw", sort="alphabetical")
    assert asked.data["state"].tolist() == sorted(STATES["state"]) and "largest" not in notes(asked)


def test_time_periods_stay_in_time_order_unless_asked_otherwise():
    data = pd.DataFrame({"year": [2022, 2020, 2021], "sales": [3.0, 1, 2]})
    spec = draw("bar", data, "year", "sales")
    assert spec.data["year"].tolist() == [2020, 2021, 2022] and spec.params["orientation"] == "vertical"
    asked = draw("bar", data, "year", "sales", sort="descending")            # the question asked: largest first
    assert asked.data["year"].tolist() == [2022, 2021, 2020]
    assert "time order" in notes(asked, "override")


def test_long_names_turn_bars_horizontal_unless_vertical_is_asked_for():
    data = pd.DataFrame({"district": ["A very long district name", "Another long district name"], "n": [1.0, 2]})
    auto = draw("bar", data, "district", "n")
    assert auto.params["orientation"] == "horizontal"
    asked = draw("bar", data, "district", "n", orientation="vertical")
    assert asked.params["orientation"] == "vertical" and "as you asked" in notes(asked, "override")
    assert "A very long\ndistrict name" in asked.params["tick_labels"]           # wrapped, not slanted
    turned = draw("bar", data, "district", "n", orientation="vertical", label_angle=90)
    assert turned.params["label_angle"] == 90 and "A very long district name" in turned.params["tick_labels"]
    assert "90 degrees" in notes(turned, "override")
    slanted = draw("bar", data, "district", "n", orientation="vertical", rotation=45)
    assert "slant" in notes(slanted, "override") and render(slanted, Style(), dpi=40)[:4] == b"\x89PNG"
    assert "Labels" not in notes(auto)                                     # no angle asked, no note


def test_more_than_fifteen_bars_is_refused_and_time_becomes_a_line():
    many = pd.DataFrame({"site": [f"S{i}" for i in range(20)], "v": np.arange(20.0)})
    spec = draw("bar", many, "site", "v")
    assert spec.kind == "table" and "at most 15" in notes(spec, "fallback")
    monthly = pd.DataFrame({"month": pd.date_range("2024-01-01", periods=20, freq="MS"), "v": np.arange(20.0)})
    assert draw("bar", monthly, "month", "v").kind == "line"


def test_asked_for_rules_step_back_with_a_note():
    many = pd.DataFrame({"site": [f"S{i}" for i in range(20)], "v": np.arange(20.0)})
    bars = draw("bar", many, "site", "v", as_asked=True)
    assert bars.kind == "bar" and len(bars.data) == 20 and "drawn as you asked" in notes(bars, "override")
    assert render(bars, Style(), dpi=40)[:4] == b"\x89PNG"
    monthly = pd.DataFrame({"month": pd.date_range("2024-01-01", periods=20, freq="MS"), "v": np.arange(20.0)})
    assert draw("bar", monthly, "month", "v", as_asked=True).kind == "bar"
    six = pd.DataFrame({"part": list("ABCDEF"), "pct": [20.0, 20, 20, 20, 10, 10]})
    pie = draw("pie", six, "part", "pct", as_asked=True)
    assert pie.kind == "pie" and "at most 5 slices" in notes(pie, "override")
    slope = draw("slope", pd.DataFrame({"s": list("ABCDEF"), "a": np.arange(6.0), "b": np.arange(6.0) + 1}),
                 "s", "a", "b", as_asked=True)
    assert slope.kind == "slope" and "at most 4" in notes(slope, "override")
    wide = pd.DataFrame({"g": list("AB"), "w": [1.0, 2], "x": [1.0, 2], "y": [1.0, 2], "z": [1.0, 2], "q": [1.0, 2]})
    grouped = draw("grouped_bar", wide, "g", ["w", "x", "y", "z", "q"], as_asked=True)
    assert grouped.kind == "grouped_bar" and "two to four series" in notes(grouped, "override")
    four = draw("grouped_bar", wide, "g", ["w", "x", "y", "z"], orientation="horizontal")
    assert four.params["orientation"] == "horizontal" and "at most three series" in notes(four, "override")
    falling = pd.DataFrame({"year": range(2000, 2004), "balance": [1.0, -2, 3, -1]})
    area = draw("area", falling, "year", ["balance"], as_asked=True)
    assert area.kind == "area" and "as you asked" in notes(area, "override")


def test_a_chart_named_in_the_question_counts_as_asked():
    # Seen from gemini-3.5-flash-lite: 'Bar chart of the daily average' without as_asked=True became a line.
    days = pd.DataFrame({"day": pd.date_range("2020-11-01", periods=30).date, "pm25": np.arange(30.0)})
    code = "result = days\nceew.bar(days, 'day', 'pm25')"
    _, outputs = run_generated_code(code.replace("days", "df"), days, {}, question="Bar chart of the daily average")
    assert outputs[0][1][0].kind == "bar" and len(outputs[0][1][0].data) == 30
    _, outputs = run_generated_code(code.replace("days", "df"), days, {}, question="Daily average PM2.5")
    assert outputs[0][1][0].kind == "line"                                  # not asked: the rules' default
    names = pd.DataFrame({"station": ["A very long station name", "Another long station name"], "v": [1.0, 2]})
    _, outputs = run_generated_code("ceew.bar(df, 'station', 'v')\nresult = df", names, {},
                                    question="Show a vertical chart of v by station")
    assert outputs[0][1][0].params["orientation"] == "vertical"             # 'vertical' in the question


def test_a_table_sorted_as_asked_keeps_its_order():
    # Seen from gemini-3.5-flash-lite: asked for hours largest first, it sorted the table but passed no sort=.
    hours = pd.DataFrame({"hour": range(24), "pm25": np.sin(np.arange(24) / 4) * 50 + 150})
    ranked = hours.sort_values("pm25", ascending=False)
    spec = draw("bar", ranked, "hour", "pm25", as_asked=True)
    assert spec.data["pm25"].is_monotonic_decreasing and "time order" in notes(spec, "override")
    plain = draw("bar", ranked.head(12), "hour", "pm25")                     # not asked: back in time order
    assert plain.data["hour"].is_monotonic_increasing


def test_accuracy_rules_hold_even_when_asked():
    double = pd.DataFrame({"answer": list("ABC"), "pct": [60.0, 50, 30]})
    assert draw("pie", double, "answer", "pct", as_asked=True).kind == "bar"           # R1: not one whole
    gaps = pd.DataFrame({"year": range(2000, 2003), "a": [1.0, None, 3], "b": [1.0, 2, 3]})
    assert draw("area", gaps, "year", ["a", "b"], as_asked=True).kind == "line"      # filling gaps invents values
    bars = draw("bar", STATES, "state", "capacity_mw", as_asked=True)
    assert render(bars, Style(), dpi=40)[:4] == b"\x89PNG"
    stacked = pd.DataFrame({"state": ["A", "B"], "lpg": [70.0, 60], "wood": [50.0, 40]})
    assert draw("stacked_bar", stacked, "state", ["lpg", "wood"], as_asked=True).kind == "grouped_bar"


def test_bars_need_one_row_per_category():
    data = pd.DataFrame({"state": ["A", "A"], "v": [1.0, 2]})
    with pytest.raises(ChartInputError, match="Aggregate"):
        draw("bar", data, "state", "v")


def test_many_categories_offer_the_grey_out_option_without_choosing():
    data = pd.DataFrame({"state": [f"S{i}" for i in range(12)], "v": np.arange(12.0)})
    spec = draw("bar", data, "state", "v")
    assert "name one to three" in notes(spec) and spec.params["highlight"] == []
    picked = draw("bar", data, "state", "v", highlight=["s3", "Nowhere"])
    assert picked.params["highlight"] == ["S3"] and "Nowhere" in notes(picked)


def test_pie_only_for_up_to_five_parts_that_make_one_whole():
    good = draw("pie", pd.DataFrame({"fuel": ["LPG", "Wood", "Other"], "share": [0.6, 0.3, 0.1]}), "fuel", "share")
    assert good.kind == "pie" and good.data["share"].sum() == pytest.approx(100)
    assert good.data["fuel"].tolist()[-1] == "Other"                    # 'Other' goes last
    double = draw("pie", pd.DataFrame({"answer": list("ABC"), "pct": [60.0, 50, 30]}), "answer", "pct")
    assert double.kind == "bar" and "R1" in notes(double, "fallback")
    six = draw("pie", pd.DataFrame({"part": list("ABCDEF"), "pct": [20.0, 20, 20, 20, 10, 10]}), "part", "pct")
    assert six.kind == "bar" and "at most 5" in notes(six, "fallback")


def test_pie_of_counts_shows_shares_of_the_whole():
    # Seen from gemini-3.5-flash-lite: households per income band passed as counts.
    counts = pd.DataFrame({"income_band": ["Low", "Middle", "High"], "household_count": [270, 240, 90]})
    spec = draw("pie", counts, "income_band", "household_count")
    assert spec.kind == "pie" and spec.data["household_count"].round(1).tolist() == [45.0, 40.0, 15.0]
    assert "counted once" in notes(spec)
    shares = pd.DataFrame({"answer": list("ABC"), "share": [60.0, 50.0, 30.0]})       # named as shares: refused
    assert draw("pie", shares, "answer", "share").kind == "bar"


def test_stacked_bars_that_do_not_add_up_become_grouped_bars():
    data = pd.DataFrame({"state": ["A", "B"], "lpg": [70.0, 60], "wood": [50.0, 40]})
    spec = draw("stacked_bar", data, "state", ["lpg", "wood"])
    assert spec.kind == "grouped_bar" and "120%" in notes(spec, "fallback")
    whole = draw("stacked_bar", data.assign(wood=[30.0, 40]), "state", ["lpg", "wood"])
    assert whole.kind == "stacked_bar"


def test_lines_break_at_gaps_instead_of_joining():
    data = pd.DataFrame({"year": [2000, 2001, 2002, 2006, 2007], "v": [1.0, 2, 3, 4, 5]})
    spec = draw("line", data, "year", "v")
    assert spec.data["v"].isna().sum() == 1 and "gap" in notes(spec)


def test_area_with_missing_values_becomes_a_line():
    data = pd.DataFrame({"year": [2000, 2001, 2002], "a": [1.0, None, 3], "b": [1.0, 2, 3]})
    assert draw("area", data, "year", ["a", "b"]).kind == "line"


def test_slope_with_many_categories_becomes_a_range_plot():
    data = pd.DataFrame({"state": list("ABCDEF"), "2019": np.arange(6.0), "2024": np.arange(6.0) + 1})
    assert draw("slope", data, "state", "2019", "2024").kind == "range_plot"


def test_histogram_states_automatic_bins_and_uses_given_ones():
    values = pd.DataFrame({"income": np.random.default_rng(0).gamma(4, 6, 300)})
    auto = draw("histogram", values, "income")
    assert "chosen automatically" in notes(auto)
    given = draw("histogram", values, "income", bin_width=5)
    assert np.allclose(np.diff(given.data["bin_start"]), 5) and given.data["count"].sum() == 300
    assert "chosen automatically" not in notes(given)


# --- Questions that broke: hour of day, vertical box plots, Gemini's own argument names ---------

HOURLY = pd.DataFrame({
    "observation_hour_start": pd.date_range("2024-01-01", periods=24 * 60, freq="h").strftime("%Y-%m-%d %H:%M:%S"),
    "location_name": np.tile(["Central School, Lucknow - CPCB", "Talkatora District Industries Center, Lucknow - CPCB"],
                             24 * 30),
    "average_value_in_hour": np.random.default_rng(3).gamma(4, 40, 24 * 60),
})


def test_hour_of_day_is_time_and_24_hours_become_a_line():
    times = pd.to_datetime(HOURLY["observation_hour_start"])
    by_hour = HOURLY.groupby(times.dt.hour)["average_value_in_hour"].mean().reset_index()
    assert charts.is_time(by_hour["observation_hour_start"])          # .dt.hour keeps the column's name
    spec = draw("bar", by_hour, "observation_hour_start", "average_value_in_hour")
    assert spec.kind == "line" and len(spec.data) == 24 and "line chart" in notes(spec, "fallback")
    assert spec.data["observation_hour_start"].tolist() == list(range(24))
    assert charts.is_time(pd.Series([1, 12], name="month")) and charts.is_time(pd.Series([0, 6], name="weekday"))
    assert not charts.is_time(pd.Series([0, 99], name="hour")) and not charts.is_time(pd.Series([3, 4], name="count"))


def test_hours_by_station_become_one_line_per_station():
    wide = pd.DataFrame({"hour": range(24), "a": np.arange(24.0), "b": np.arange(24.0) + 1})
    spec = draw("grouped_bar", wide, "hour", ["a", "b"])
    assert spec.kind == "line" and spec.params["ys"] == ["a", "b"]


def test_vertical_box_plot_wraps_long_names_instead_of_turning():
    spec = draw("boxplot", HOURLY, "location_name", "average_value_in_hour", orientation="vertical")
    assert spec.params["orientation"] == "vertical" and spec.y_title == "average_value_in_hour"
    assert all(max(len(line) for line in label.split("\n")) <= 14 for label in spec.params["tick_labels"])
    assert spec.params["labels"][0] == "Central School, Lucknow - CPCB"           # the data keeps the full name
    assert draw("boxplot", HOURLY, "location_name", "average_value_in_hour", vert=False).params["orientation"] \
        == "horizontal"
    assert render(spec, Style(), dpi=60)[:4] == b"\x89PNG"


@pytest.mark.parametrize("call", [
    lambda c: c.bar(STATES, x="state", y="capacity_mw"),
    lambda c: c.bar(STATES, "state", "capacity_mw", title="Capacity", color="red", figsize=(8, 4)),
    lambda c: c.boxplot(HOURLY, by="location_name", column="average_value_in_hour"),
    lambda c: c.boxplot(HOURLY, x="location_name", y="average_value_in_hour", vert=True),
    lambda c: c.scatter(HOURLY.assign(v=1.0), "v", "average_value_in_hour", color="red"),
    lambda c: c.line(STATES.assign(year=range(2000, 2005)), x="year", y="capacity_mw", xlabel="Year"),
    lambda c: c.box_plot(HOURLY, "location_name", "average_value_in_hour"),
    lambda c: c.column_chart(STATES, "state", "capacity_mw"),
    lambda c: c.bar(STATES.assign(b=1.0), "state", ["capacity_mw", "b"]),
])
def test_geminis_own_argument_names_still_draw_a_chart(call):
    specs = []
    call(Charts(specs.append))
    assert len(specs) == 1 and specs[0].kind != "table"


@pytest.mark.parametrize("code", [
    # Seen from gemini-3.5-flash-lite: the chart function treated like matplotlib.
    "fig = ceew.boxplot(df, 'location_name', 'average_value_in_hour', orientation='vertical')\nst.pyplot(fig)\n"
    "result = 1",
    "ax = ceew.bar(t, 'location_name', 'average_value_in_hour')\nax.set_title('PM2.5')\nax.set_xlabel('x')\n"
    "st.pyplot()\nresult = t",
    "fig, ax = ceew.bar(t, 'location_name', 'average_value_in_hour')\nst.write(fig)\nresult = ceew.line(t)",
])
def test_matplotlib_habits_around_chart_functions_do_no_harm(code):
    setup = "t = df.groupby('location_name', as_index=False)['average_value_in_hour'].mean()\n"
    result, outputs = run_generated_code(setup + code, HOURLY, {})
    assert [name for name, _, _ in outputs if name != "chart"] == []           # no stray images or text
    assert any(name == "chart" for name, _, _ in outputs)
    assert not isinstance(result, charts.ChartHandle)


def test_dates_as_python_date_objects_or_text_are_real_dates():
    # Seen from gemini-3.5-flash-lite: groupby(times.dt.date) gives date objects; each day became a label.
    days = pd.date_range("2020-11-01", periods=92, freq="D")
    daily = pd.DataFrame({"date": days.date, "pm25": np.arange(92.0)})
    spec = draw("line", daily, "date", "pm25")
    assert pd.api.types.is_datetime64_any_dtype(spec.data["date"]) and "evenly spaced" not in notes(spec)
    months = pd.DataFrame({"month": ["2020-11", "2020-12", "2021-01"], "pm25": [1.0, 2, 3]})
    assert pd.api.types.is_datetime64_any_dtype(draw("line", months, "month", "pm25").data["month"])
    words = pd.DataFrame({"day": ["Monday", "Tuesday"], "pm25": [1.0, 2]})
    assert draw("bar", words, "day", "pm25").data["day"].tolist() == ["Tuesday", "Monday"]   # still words
    by_day = draw("bar", daily, "date", "pm25")                       # 92 days: a line, not a refused bar chart
    assert by_day.kind == "line"


def test_ordered_categories_keep_their_order():
    aqi = pd.DataFrame({"category": pd.Categorical(["Good", "Moderate", "Severe"],
                                                   categories=["Good", "Moderate", "Severe"], ordered=True),
                        "share": [5.0, 60, 35]})
    assert draw("bar", aqi, "category", "share").params["labels"] == ["Good", "Moderate", "Severe"]
    # Seen from gemini-3.5-flash-lite: AQI bands as plain text, sorted by size (Very poor first).
    text = pd.DataFrame({"band": ["Very poor", "Severe", "Poor", "Moderate", "Satisfactory", "Good"],
                         "share": [51.0, 17, 15, 11, 5, 1]})
    spec = draw("bar", text, "band", "share")
    assert spec.params["labels"] == ["Good", "Satisfactory", "Moderate", "Poor", "Very poor", "Severe"]
    assert "order of the categories" in notes(spec)
    assert draw("bar", text, "band", "share", sort="descending").params["labels"][0] == "Very poor"
    bins = pd.DataFrame({"pm25": ["61-90", "0-30", "(30, 60]", "121-250"], "hours": [3.0, 9, 1, 4]})
    assert draw("bar", bins, "pm25", "hours").params["labels"] == ["0-30", "(30, 60]", "61-90", "121-250"]
    assert natural_order(pd.Series(["Lucknow", "Delhi"])) is None


def test_wide_heatmap_thins_hour_labels_and_turns_long_names():
    hours = pd.DataFrame(np.arange(96.0).reshape(4, 24), index=[f"Long station name number {i}" for i in range(4)])
    spec = draw("heatmap", hours)
    from matplotlib.figure import Figure
    for case, expect in ((spec, "thinned"), (draw("heatmap", pd.DataFrame(np.eye(5), columns=[
            f"relative_humidity_{i}" for i in range(5)], index=list("abcde"))), "turned")):
        png = render(case, Style(), dpi=40)
        assert png[:4] == b"\x89PNG", expect


def test_heatmap_takes_a_correlation_matrix():
    # Seen from gemini-3.5-flash-lite: a correlation heatmap drawn with plain matplotlib, because
    # ceew.heatmap only took long tables.
    numbers = pd.DataFrame({"a": [1.0, 2, 3, 4], "b": [2.0, 1, 4, 3], "c": [4.0, 3, 2, 1]})
    spec = draw("heatmap", numbers.corr())
    assert spec.kind == "heatmap" and spec.params["grid"].shape == (3, 3)
    assert spec.params["row_labels"] == ["a", "b", "c"]
    named = numbers.corr().rename_axis("variable").reset_index()            # labels in a column, not the index
    assert draw("heatmap", named).params["grid"].shape == (3, 3)
    long = pd.DataFrame({"station": ["A", "A", "B", "B"], "hour": [0, 1, 0, 1], "pm25": [1.0, 2, 3, 4]})
    assert draw("heatmap", long).params["grid"].shape == (2, 2)            # long tables still work


def test_empty_figures_after_a_ceew_chart_are_not_shown():
    code = ("ceew.scatter(df, 'average_value_in_hour', 'average_value_in_hour')\nst.pyplot(plt.gcf())\n"
            "fig, ax = plt.subplots()\nresult = 1")
    _, outputs = run_generated_code(code, HOURLY, {})
    assert [name for name, _, _ in outputs] == ["chart"]


@pytest.mark.parametrize("code", [
    "fig, ax = plt.subplots()\nax.plot([1, 2])\nax2 = ax.twinx()\nresult = 1",
    "df.assign(v=df['average_value_in_hour'], w=1.0)[['v', 'w']].plot(secondary_y='w')\nresult = 1",
])
def test_two_y_axes_are_refused_even_with_plain_matplotlib(code):
    # Seen from gemini-3.5-flash-lite: asked for a second y-axis, it drew one with ax.twinx().
    from csvanalyser.sandbox import ChartRuleError
    with pytest.raises(ChartRuleError, match="R4"):
        run_generated_code(code, HOURLY, {})
    from matplotlib.axes import Axes
    assert Axes.twinx.__name__ == "twinx"                                  # put back after the run


def test_unused_arguments_are_named_in_a_note():
    spec = draw("bar", STATES, "state", "capacity_mw", title="Capacity", color="red")
    assert "Not used: title, color" in notes(spec)
    assert draw("column_chart", STATES, "state", "capacity_mw").params["orientation"] == "vertical"
    assert "Not used" not in notes(draw("bar", STATES, "state", "capacity_mw"))


def test_prompt_explains_hour_of_day_and_box_plot_orientation():
    prompt = build_prompt(describe_schema(HOURLY), "average PM2.5 by hour", "")
    assert "hour of the day (0 to 23)" in prompt and "pd.to_datetime" in prompt
    assert "ceew.boxplot(data, group, value, orientation='auto'" in prompt


# --- Drawing -----------------------------------------------------------------------------------

def test_every_chart_kind_renders_in_every_use():
    rng = np.random.default_rng(0)
    specs = []
    ceew = Charts(specs.append)
    ceew.bar(STATES, "state", "capacity_mw")
    ceew.lollipop(STATES, "state", "capacity_mw")
    ceew.grouped_bar(STATES.assign(b=STATES["capacity_mw"] / 2), "state", ["capacity_mw", "b"])
    ceew.diverging_bar(STATES.assign(capacity_mw=STATES["capacity_mw"] - 30), "state", "capacity_mw")
    ceew.stacked_bar(pd.DataFrame({"s": ["A", "B"], "x": [40.0, 50], "y": [60.0, 50]}), "s", ["x", "y"])
    ceew.pie(pd.DataFrame({"p": ["A", "B"], "v": [70.0, 30]}), "p", "v")
    ceew.line(pd.DataFrame({"d": pd.date_range("2024-01-01", periods=30), "a": rng.normal(size=30),
                            "b": rng.normal(size=30)}), "d", ["a", "b"])
    ceew.area(pd.DataFrame({"year": range(2000, 2010), "a": np.arange(10.0), "b": np.ones(10)}), "year", ["a", "b"])
    ceew.slope(pd.DataFrame({"s": list("AB"), "2019": [1.0, 2], "2024": [2.0, 1]}), "s", "2019", "2024")
    ceew.range_plot(pd.DataFrame({"s": list("AB"), "2019": [1.0, 2], "2024": [2.0, 1]}), "s", "2019", "2024")
    ceew.histogram(pd.DataFrame({"v": rng.normal(size=100)}), "v")
    ceew.boxplot(pd.DataFrame({"g": list("AB") * 20, "v": rng.normal(size=40)}), "g", "v")
    ceew.scatter(pd.DataFrame({"x": rng.normal(size=50), "y": rng.normal(size=50)}), "x", "y")
    ceew.heatmap(pd.DataFrame([(r, c, rng.uniform()) for r in "AB" for c in "XYZ"], columns=["r", "c", "v"]),
                 "r", "c", "v")
    assert len(specs) == len(charts.CHART_FUNCTIONS) - 1            # donut is the pie with a hole
    for use in charts.USES:
        for spec in specs:
            assert render(spec, Style(use=use), dpi=60)[:8] == b"\x89PNG\r\n\x1a\n", (spec.kind, use)


def test_headline_source_and_note_appear_only_once_written():
    from PIL import Image
    import io
    spec = draw("bar", STATES, "state", "capacity_mw")
    svg = render(spec, Style(), fmt="svg").decode()
    for absent in ("headline", "Source", "Note", "[", "..."):
        assert absent not in svg.split("</metadata>")[-1].replace("[Axis", ""), absent   # no placeholders at all
    assert "Familjen Grotesk" in svg                                 # editable text in the CEEW font
    texts = {"headline": "State B leads", "source": "CEA", "note": "Provisional figures", "x_title": "Capacity (MW)"}
    svg = render(spec, Style(), texts, fmt="svg").decode()
    assert "State B leads" in svg and "Source: CEA" in svg and "Note: Provisional figures" in svg
    assert "Capacity (MW)" in svg
    only_source = render(spec, Style(), {"source": "CEA"}, fmt="svg").decode()
    assert "Source: CEA" in only_source and "Note:" not in only_source
    # No space is kept for words that are not there: the bare chart is shorter than the one with all three.
    bare = Image.open(io.BytesIO(render(spec, Style(), dpi=50)))
    worded = Image.open(io.BytesIO(render(spec, Style(), texts, dpi=50)))
    assert bare.height < worded.height and bare.width == worded.width


def test_outreach_is_1080_pixels_on_cream():
    from PIL import Image
    import io
    spec = draw("bar", STATES, "state", "capacity_mw")
    square = Image.open(io.BytesIO(render(spec, Style(use="outreach")))).convert("RGB")
    portrait = Image.open(io.BytesIO(render(spec, Style(use="outreach", shape="portrait"))))
    assert square.size == (1080, 1080) and portrait.size == (1080, 1350)
    assert square.getpixel((2, 2)) == (0xF5, 0xF0, 0xE7)


def test_headline_check_flags_causes_and_records_without_rewriting():
    flags = check_headline("Traffic drives pollution to its highest ever level")
    assert any("“drives”" in f for f in flags) and any("“ever”" in f for f in flags)
    assert check_headline("Delhi's PM2.5 was higher in winter than in summer") == []


def test_file_names_come_from_the_content():
    spec = draw("bar", STATES, "state", "capacity_mw")
    assert file_stem(spec, {"headline": "Rooftop solar by state!"}, "x") == "rooftop-solar-by-state"
    assert file_stem(spec, {}, "x") == "capacity-mw"


# --- In the app --------------------------------------------------------------------------------

def test_generated_code_can_draw_ceew_charts_with_the_real_column_names():
    df = pd.DataFrame({"site_name": ["Anand Vihar", "Punjabi Bagh"], "pm25": [123.45, 678.91]})
    aliases = {"site_name": "col_1", "pm25": "col_2"}
    code = ("t = df.groupby('col_1', as_index=False)['col_2'].mean()\n"
            "ceew.bar(t, 'col_1', 'col_2', why='Compares col_2 across col_1 (magnitude).')\nresult = t")
    result, outputs = run_generated_code(code, df.rename(columns=aliases), {v: k for k, v in aliases.items()})
    (name, (spec,), _), = outputs
    assert name == "chart" and spec.kind == "bar"
    assert list(spec.data.columns) == ["site_name", "pm25"] and spec.x_title == "pm25"
    assert spec.params["category"] == "site_name" and spec.why == "Compares pm25 across site_name (magnitude)."
    assert list(result.columns) == ["site_name", "pm25"]


def test_chart_input_errors_do_not_carry_data_values():
    df = pd.DataFrame({"site": ["Anand Vihar", "Anand Vihar"], "pm25": [123.45, 678.91]})
    with pytest.raises(ChartInputError) as info:
        run_generated_code("ceew.bar(df, 'site', 'pm25')\nresult = 1", df, {})
    assert "Anand" not in str(info.value) and "123" not in str(info.value)


def test_plain_matplotlib_charts_get_the_house_style():
    import matplotlib
    code = ("fig, ax = plt.subplots()\n"
            "result = (list(plt.rcParams['font.family']), plt.rcParams['axes.prop_cycle'].by_key()['color'][0])")
    result, _ = run_generated_code(code, pd.DataFrame({"a": [1]}), {}, Style(pair="green-blue"))
    assert result[0][0] == "Familjen Grotesk" and result[1] == SHADES["green"]["base"]
    assert matplotlib.rcParams["font.family"] != result[0]           # not left behind for the rest of the app


def test_prompt_lists_the_chart_functions_and_rules_but_no_data():
    df = pd.DataFrame({"site": ["Anand Vihar"], "pm25": [123.45]})
    prompt = build_prompt(describe_schema(df), "Show pm25 by site", "")
    assert "ceew.bar(" in prompt and "ceew.histogram(" in prompt
    assert "Do NOT write titles, headlines" in prompt and "highlight: only items the question names" in prompt
    assert "Anand" not in prompt and "123.45" not in prompt


def test_exported_notebook_loads_the_chart_functions(tmp_path):
    turns = [{"question": "q", "code": "ceew.bar(df, 'a', 'b')\nresult = df", "error": None}]
    assert uses_ceew_charts(turns)
    notebook = build_notebook(turns, "pd.read_csv('data.csv')", ["a", "b"])
    assert "from ceew_charts import notebook_charts" in notebook
    # The saved file works on its own, outside the app.
    path = tmp_path / "ceew_charts.py"
    path.write_text(charts_module_source(), encoding="utf-8")
    spec = importlib.util.spec_from_file_location("ceew_charts_test", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules["ceew_charts_test"] = module
    try:
        spec.loader.exec_module(module)
        drawn = []
        module.Charts(drawn.append).bar(STATES, "state", "capacity_mw")
        assert module.render(drawn[0], module.Style(), dpi=50)[:4] == b"\x89PNG"
    finally:
        sys.modules.pop("ceew_charts_test", None)
