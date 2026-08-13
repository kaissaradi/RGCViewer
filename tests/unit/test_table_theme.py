import pandas as pd
from qtpy.QtCore import Qt

from src.gui.theme import DARK_COLORS, LIGHT_COLORS
from src.gui.widgets.widgets import (
    HiddenIdFilterProxyModel,
    HighlightStatusPandasModel,
)


def _model():
    df = pd.DataFrame(
        {
            "cluster_id": [1, 2],
            "n_spikes": [10, 20],
            "status": ["Original", "Clean"],
        }
    )
    return HighlightStatusPandasModel(df)


def test_table_body_text_follows_light_theme():
    model = _model()
    model.update_colors(LIGHT_COLORS)
    color = model.data(model.index(0, 0), Qt.ForegroundRole)
    assert color.name().lower() == LIGHT_COLORS["text_primary"].lower()


def test_table_body_text_follows_dark_theme():
    model = _model()
    model.update_colors(DARK_COLORS)
    color = model.data(model.index(0, 0), Qt.ForegroundRole)
    assert color.name().lower() == DARK_COLORS["text_primary"].lower()


def test_table_status_uses_theme_good_color():
    model = _model()
    model.update_colors(LIGHT_COLORS)
    status_col = list(model._dataframe.columns).index("status")
    color = model.data(model.index(1, status_col), Qt.ForegroundRole)
    assert color.name().lower() == LIGHT_COLORS["status_good_text"].lower()


def test_proxy_update_colors_forwards_to_source():
    model = _model()
    proxy = HiddenIdFilterProxyModel(lambda: (), None)
    proxy.setSourceModel(model)
    proxy.update_colors(LIGHT_COLORS)
    color = model.data(model.index(0, 0), Qt.ForegroundRole)
    assert color.name().lower() == LIGHT_COLORS["text_primary"].lower()
