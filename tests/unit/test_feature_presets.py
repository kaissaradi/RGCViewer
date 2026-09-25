"""Feature Extraction presets are saved and remembered (PLAN.md Q15)."""

import pytest
from qtpy.QtCore import QSettings

from src.gui.panels import feature_presets as fp


@pytest.fixture
def settings(tmp_path):
    return QSettings(str(tmp_path / "t.ini"), QSettings.Format.IniFormat)


def test_default_until_something_is_saved(settings):
    assert fp.current(settings) == fp.DEFAULT_NAME
    assert fp.panels_for(fp.DEFAULT_NAME, settings) is None
    assert fp.load_presets(settings) == {}


def test_save_select_and_delete(settings):
    pairs = [("ACG PC1", "RF area"), ("Time to peak", "Grating OSI")]
    fp.save_preset("OFF types", pairs, settings)
    assert fp.load_presets(settings) == {"OFF types": pairs}
    assert fp.current(settings) == "OFF types"           # saving selects it
    assert fp.panels_for("OFF types", settings) == pairs
    fp.set_current(fp.DEFAULT_NAME, settings)
    assert fp.current(settings) == fp.DEFAULT_NAME
    fp.set_current("OFF types", settings)
    fp.delete_preset("OFF types", settings)
    assert fp.load_presets(settings) == {} and fp.current(settings) == fp.DEFAULT_NAME


def test_survives_a_new_settings_object(settings, tmp_path):
    fp.save_preset("mine", [("a", "b")], settings)
    settings.sync()
    again = QSettings(str(tmp_path / "t.ini"), QSettings.Format.IniFormat)
    assert fp.load_presets(again) == {"mine": [("a", "b")]} and fp.current(again) == "mine"


def test_default_name_is_reserved_and_bad_json_is_ignored(settings):
    with pytest.raises(ValueError):
        fp.save_preset(fp.DEFAULT_NAME, [("a", "b")], settings)
    settings.setValue("feature_extraction/presets", "{not json")
    assert fp.load_presets(settings) == {}
