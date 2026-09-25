"""Hand-typed Vision classes map to one name per type (PLAN.md Q44).

Every string below occurs in the lab's .params files (scan 2026-09-25).
"""

import pytest

from src.analysis.class_names import canonical_type


@pytest.mark.parametrize("class_id, expected", [
    ("All/on/brisk sustained", ("ON brisk sustained", False)),
    ("All/ON/brisk sustained", ("ON brisk sustained", False)),
    ("All/on/brisk-sustained", ("ON brisk sustained", False)),
    ("All/OFF/BRISK SUSTAINED", ("OFF brisk sustained", False)),
    ("All/off/brisk sustained/nc21/nc22/nc23", ("OFF brisk sustained", False)),
    ("All/OFF/BRISK TRANSIENT", ("OFF brisk transient", False)),
    ("All/on/brisk-transient/nc33", ("ON brisk transient", False)),
    ("All/OFF/TRANSIENT", ("OFF transient", False)),
    ("All/off/small transient", ("OFF small transient", False)),
    ("All/off/transient?", ("OFF transient", True)),
    ("All/ON/maybe b-sus", ("ON brisk sustained", True)),
    ("All/OFF/unclassified/brisk sustained", ("OFF brisk sustained", True)),
    ("All/on/brisksustained", ("ON brisk sustained", False)),     # run together
    ("All/Off/Brisk_Transient", ("OFF brisk transient", False)),  # underscore, title case
])
def test_type_names(class_id, expected):
    assert canonical_type(class_id) == expected


@pytest.mark.parametrize("class_id", [
    "All", "All/ON/unclassified", "All/off/unclassified/nc107", "All/nc3", "All/weak",
    "All/huge/misfit", "All/off?", "All/class-2/nc12", "All/on/weird",
    "All/on/brisk-sustained/removed-brisk-sustained",   # taken out of the type
    "All/on/off-center OBS",                             # cannot tell which type
    "", None,
])
def test_not_a_type(class_id):
    assert canonical_type(class_id) is None
