"""Page assets: mobile styles, toolbar settings and the CEEW logo hook."""
import pathlib

ROOT = pathlib.Path(__file__).resolve().parents[1]
INDEX = (ROOT / "index.html").read_text(encoding="utf-8")
APP = (ROOT / "stlite_app.py").read_text(encoding="utf-8")
APP_CSS = (ROOT / "app.css").read_text(encoding="utf-8")
TOUR_JS = (ROOT / "tour.js").read_text(encoding="utf-8")
GUIDE = (ROOT / "guide.html").read_text(encoding="utf-8")


def test_mobile_styles_are_loaded():
    assert './app.css"' in INDEX
    assert "@media (max-width: 640px)" in APP_CSS
    assert ".st-key-tour-header" in APP_CSS and ".st-key-tour-examples" in APP_CSS
    assert 'name="viewport"' in INDEX


def test_developer_toolbar_hidden():
    assert '"client.toolbarMode": "minimal"' in INDEX


def test_tour_handles_the_phone_sidebar():
    assert "function closeSidebar()" in TOUR_JS and "isNarrow()" in TOUR_JS
    assert 'getAttribute("aria-expanded") === "true"' in TOUR_JS     # state, not visibility
    assert "scrollIntoView" in TOUR_JS


def test_logo_is_shown_only_when_the_file_exists():
    assert 'LOGO_PATH = "assets/ceew-logo.png"' in APP
    assert "os.path.exists(LOGO_PATH)" in APP and "st.logo(" in APP
    if (ROOT / "assets" / "ceew-logo.png").exists():
        assert '"assets/ceew-logo.png"' in INDEX, "logo file exists but is not listed in index.html"


def test_guide_has_phone_styles():
    assert "@media (max-width: 600px)" in GUIDE
