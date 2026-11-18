"""The guided tour (tour.js) and the page (stlite_app.py) must agree on section names.

tour.js finds each section by the CSS class Streamlit gives a keyed container:
st.container(key="tour-x") -> class "st-key-tour-x". If a key is renamed or removed
in the app without updating tour.js, the tour would silently skip that step.
"""
import pathlib
import re

ROOT = pathlib.Path(__file__).resolve().parents[1]
TOUR_JS = (ROOT / "tour.js").read_text(encoding="utf-8")
APP = (ROOT / "stlite_app.py").read_text(encoding="utf-8")
INDEX = (ROOT / "index.html").read_text(encoding="utf-8")


def tour_keys():
    return re.findall(r'key:\s*"(tour-[a-z-]+)"', TOUR_JS)


def test_every_tour_step_has_an_anchor_in_the_app():
    keys = tour_keys()
    assert len(keys) >= 12
    for key in keys:
        assert f'key="{key}"' in APP, f"tour.js step '{key}' has no st.container(key=...) in stlite_app.py"


def test_tour_uses_existing_sample_buttons_and_data_marker():
    for key in ["sample_sidebar", "sample_welcome"]:
        assert f".st-key-{key} button" in TOUR_JS
        assert f'key="{key}"' in APP
    assert 'section("tour-tabs")' in TOUR_JS and 'key="tour-tabs"' in APP


def test_tab_labels_match():
    labels = re.search(r"MAIN_TABS = \[(.+?)\]", APP).group(1)
    for label in re.findall(r'"([^"]+)"', labels):
        assert f'"{label}"' in TOUR_JS, label
    assert '"Data quality audit"' in TOUR_JS and '"Data quality audit"' in APP


def test_tour_buttons_and_assets():
    assert 'class="{style}">🧭 Take the tour</button>' in APP
    assert ".tda-tour-start" in TOUR_JS
    assert "driver.js@1.10.2/dist/driver.js.iife.js" in INDEX
    assert "driver.js@1.10.2/dist/driver.css" in INDEX
    assert './tour.js"' in INDEX and './tour.css"' in INDEX


def test_tour_has_a_notebook_download_step():
    keys = tour_keys()
    assert "tour-export-session" in keys and "tour-export-report" in keys
    assert "Jupyter notebook (.ipynb)" in TOUR_JS


def test_connect_step_points_to_ceew_gemini_pro_access():
    assert "CEEW has <b>Gemini Pro</b>." in TOUR_JS
    assert "Gemini Enterprise" not in TOUR_JS
    assert 'href=\\"https://aistudio.google.com/\\" target=\\"_blank\\" rel=\\"noopener\\"' in TOUR_JS


def test_sample_button_has_its_own_row():
    assert "popover.wrapper.insertBefore(row, popover.footer)" in TOUR_JS
    css = (ROOT / "tour.css").read_text(encoding="utf-8")
    assert ".tda-tour-sample-row" in css and "width: 100%" in css


def test_only_one_tour_button_on_the_page():
    assert APP.count("tour_button(") == 2      # the function definition and one call (in the header)


def test_tour_text_covers_the_main_features():
    text = TOUR_JS
    for feature in ["API key", "synthetic sample", "What you can analyse", "Example questions", "Analyse",
                    "Run my version", "Explain this code", "Jupyter notebook", "Anonymise column names",
                    "printable guide"]:
        assert feature in text, feature
