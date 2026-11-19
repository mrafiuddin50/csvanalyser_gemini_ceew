"""CEEW charts: the data viz skill's rules, the house style, and how charts reach the page."""
import importlib.util
import sys

import numpy as np
import pandas as pd
import pytest

from csvanalyser import charts
from csvanalyser.charts import (GREY_LIGHT, GREY_LIGHTEST, SHADES, ChartInputError, Charts, Style, categorical_colours,
                                check_headline, file_stem, render)
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


def test_time_periods_stay_in_time_order_as_vertical_bars():
    data = pd.DataFrame({"year": [2022, 2020, 2021], "sales": [3.0, 1, 2]})
    spec = draw("bar", data, "year", "sales", sort="descending")
    assert spec.data["year"].tolist() == [2020, 2021, 2022]
    assert spec.params["orientation"] == "vertical" and "time order" in notes(spec)


def test_long_names_switch_to_horizontal_bars():
    data = pd.DataFrame({"district": ["A very long district name", "Another long district name"], "n": [1.0, 2]})
    spec = draw("bar", data, "district", "n", orientation="vertical")
    assert spec.params["orientation"] == "horizontal" and "rotated" in notes(spec)


def test_more_than_fifteen_bars_is_refused_and_time_becomes_a_line():
    many = pd.DataFrame({"site": [f"S{i}" for i in range(20)], "v": np.arange(20.0)})
    spec = draw("bar", many, "site", "v")
    assert spec.kind == "table" and "at most 15" in notes(spec, "fallback")
    monthly = pd.DataFrame({"month": pd.date_range("2024-01-01", periods=20, freq="MS"), "v": np.arange(20.0)})
    assert draw("bar", monthly, "month", "v").kind == "line"


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


def test_placeholders_until_the_researcher_writes_the_words():
    spec = draw("bar", STATES, "state", "capacity_mw")
    svg = render(spec, Style(), fmt="svg").decode()
    for placeholder in ("[Your headline here]", "[Source: ...]", "[Note: ...]"):
        assert placeholder in svg
    assert "Familjen Grotesk" in svg                                 # editable text in the CEEW font
    texts = {"headline": "State B leads", "source": "CEA", "note": "Provisional figures", "x_title": "Capacity (MW)"}
    svg = render(spec, Style(), texts, fmt="svg").decode()
    assert "State B leads" in svg and "Source: CEA" in svg and "Note: Provisional figures" in svg
    assert "Capacity (MW)" in svg and "[Your headline here]" not in svg


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
