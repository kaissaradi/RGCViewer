"""Type barcode (Q40) and mosaic numbers (Q42) on made-up data with known answers."""

import numpy as np

from src.analysis import barcode, mosaic_stats as ms
from src.analysis.rf_geometry import RFFit


def test_barcode_groups_sorts_and_flags_the_misfit():
    t = np.linspace(0, 1, 31)
    on = np.sin(np.pi * t)
    rows = {1: on, 2: on * 3, 3: on + 0.01 * np.cos(9 * t), 4: -on,   # 4 is OFF in an ON group
            10: -on, 11: -on * 0.5, 12: -on}
    group = {1: "ON", 2: "ON", 3: "ON", 4: "ON", 10: "OFF", 11: "OFF", 12: "OFF", 99: "ON"}
    bc = barcode.build_barcode(rows, group, ["ON", "OFF"])
    assert [b[0] for b in bc.bands] == ["ON", "OFF"]
    assert bc.cells[:4][-1] == 4                    # least typical last in its band
    assert bc.misfits() == [4]
    np.testing.assert_allclose(np.abs(bc.matrix).max(axis=1), 1.0)   # scaled per row
    assert 99 not in bc.cells                       # no vector, no row
    assert bc.row_of(11) >= 4 and bc.row_of(12345) == -1


def test_radius_follows_the_ellipse():
    f = RFFit(0, 0, 3.0, 1.0, 0.0)                  # SigmaX along +x when theta = 0
    assert np.isclose(ms.radius_toward(f, 0.0), 3.0)
    assert np.isclose(ms.radius_toward(f, np.pi / 2), 1.0)
    g = RFFit(0, 0, 3.0, 1.0, -np.pi / 2)           # y-up: SigmaX axis at -theta = +90°
    assert np.isclose(ms.radius_toward(g, np.pi / 2), 3.0)


def test_a_tiling_grid_scores_2_and_a_duplicate_is_a_close_pair():
    fits = {i * 10 + j: RFFit(4.0 * i, 4.0 * j, 1.0, 1.0 + 1e-6, 0.0)
            for i in range(5) for j in range(5)}
    st = ms.class_stats(fits)
    assert np.isclose(st.median_nnnd, 4.0, atol=1e-3)   # centres 4 apart, radius 1: 2*4/2
    assert st.close_pairs == []
    fits[999] = RFFit(0.3, 0.2, 1.0, 1.0 + 1e-6, 0.0)  # sits on top of cell 0
    st = ms.class_stats(fits)
    assert st.close_pairs and set(st.close_pairs[0][:2]) == {0, 999}
    assert "close pair" in st.verdict


def test_hole_candidates_are_inside_and_far():
    grid = {i * 10 + j: RFFit(4.0 * i, 4.0 * j, 1.0, 1.0 + 1e-6, 0.0)
            for i in range(5) for j in range(5) if (i, j) != (2, 2)}
    others = {500: RFFit(8.0, 8.0, 1.0, 1.0 + 1e-6, 0.0),     # in the hole
              501: RFFit(1.0, 0.5, 1.0, 1.0 + 1e-6, 0.0),     # next to a member
              502: RFFit(40.0, 40.0, 1.0, 1.0 + 1e-6, 0.0)}   # outside the footprint
    assert [c for c, _v in ms.hole_candidates(grid, others)] == [500]


def test_a_cell_that_fits_another_type_better_is_flagged_and_bins_are_not_judged():
    t = np.linspace(0, 1, 31)
    sustained = np.exp(-((t - 0.8) / 0.12) ** 2)                     # one lobe
    transient = sustained - 0.7 * np.exp(-((t - 0.55) / 0.12) ** 2)  # biphasic
    rng = np.random.default_rng(0)
    rows, group = {}, {}
    for i in range(8):
        rows[i] = sustained + 0.02 * rng.standard_normal(31); group[i] = "ON brisk sustained"
        rows[20 + i] = transient + 0.02 * rng.standard_normal(31); group[20 + i] = "ON brisk transient"
    rows[99] = transient.copy(); group[99] = "ON brisk sustained"           # mislabelled
    for i in range(5):
        rows[50 + i] = rng.standard_normal(31); group[50 + i] = "unclassified"  # a bin
    code = barcode.build_barcode(rows, group, ["ON brisk sustained", "ON brisk transient", "unclassified"],
                                 type_groups=["ON brisk sustained", "ON brisk transient"])
    assert code.misfits() == [99]
    assert code.elsewhere[code.row_of(99)][0] == "ON brisk transient"
    assert np.isnan(code.typicality[code.row_of(52)])
