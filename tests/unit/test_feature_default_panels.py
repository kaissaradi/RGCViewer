"""Feature Extraction default panels (PLAN.md Q14; user's choice 2026-09-24)."""

import numpy as np

from src.analysis import feature_catalog as fc


def _catalog(names):
    return {n: fc.FeatureEntry(n, "g", np.arange(5.0)) for n in names}


ALL = ["Temporal STA PC1", "Temporal STA PC2", "Temporal STA PC3", "ACG PC1",
       "ACG PC2", "ACG PC3", "RF area", "RF diameter (short)", "Time to peak",
       "Firing rate", "Grating DSI"]


def test_defaults_are_the_four_chosen_pairs_then_two_random():
    panels = fc.resolve_panels(_catalog(ALL), rng=np.random.default_rng(0))
    assert panels[:4] == [
        ("Temporal STA PC1", "Temporal STA PC2"),
        ("ACG PC1", "ACG PC2"),
        ("Temporal STA PC1", "ACG PC1"),
        ("Temporal STA PC1", "RF area"),
    ]
    keys = [tuple(sorted(p)) for p in panels]
    assert len(panels) == 6 and len(set(keys)) == 6
    assert all(a in ALL and b in ALL and a != b for a, b in panels[4:])


def test_random_pairs_change_between_openings():
    seen = {tuple(fc.resolve_panels(_catalog(ALL), rng=np.random.default_rng(s))[4:])
            for s in range(10)}
    assert len(seen) > 1


def test_a_run_without_an_sta_gets_the_ranked_substitutes():
    names = [n for n in ALL if not n.startswith("Temporal STA")]
    panels = fc.resolve_panels(_catalog(names), rng=np.random.default_rng(0))
    assert panels[1] == ("ACG PC1", "ACG PC2")
    assert panels[0] == ("ACG PC1", "RF area")          # first usable fallback
    assert all(a in names and b in names for a, b in panels)
    assert len({tuple(sorted(p)) for p in panels}) == 6


def test_empty_catalog_gives_no_panels():
    assert fc.resolve_panels({}) == []
