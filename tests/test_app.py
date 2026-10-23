from pathlib import Path

from streamlit.testing.v1 import AppTest

APP = Path(__file__).resolve().parents[1] / "app" / "streamlit_app.py"


def test_every_tab_renders_without_exceptions():
    at = AppTest.from_file(str(APP), default_timeout=30).run()
    assert not at.exception
    assert len(at.tabs) == 4
    assert len(at.markdown) > 5 and len(at.table) >= 2
    at.selectbox[0].select_index(3).run()  # pick another user in the recommendation viewer
    assert not at.exception
    at.slider[-1].set_value(0.8).run()  # CUPED what-if
    assert not at.exception
