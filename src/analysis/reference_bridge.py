"""
reference_bridge.py
===================
Lightweight adapter that provides analysis products (STAs, RF params) from a
matched reference run, keyed by the *current* run's cell IDs.

The bridge lazily loads only the vision data it needs from the reference run
and remaps IDs transparently so panels never need to know the data came from
elsewhere.

Typical lifecycle:
    1. User triggers "Map Reference Run..." from the menu
    2. CrossRunMatcher produces a MatchingReport (or loads from JSON)
    3. ReferenceBridge is created with the report + reference vision dir
    4. DataManager holds the bridge and queries it as fallback when the
       current run lacks STAs/RFs

Usage:
    bridge = ReferenceBridge.from_matching_report(report, ref_dir, ref_dataset)
    sta = bridge.get_sta(current_cell_id)        # returns STA or None
    rf  = bridge.get_rf_params(current_cell_id)  # returns stafit or None
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, Optional, Set

logger = logging.getLogger(__name__)


class ReferenceBridge:
    """
    Serves analysis products from a reference run using a pre-computed
    ID mapping {current_run_id → reference_run_id}.
    """

    def __init__(
        self,
        mapping: Dict[int, int],
        ref_stas=None,           # LazySTADict or dict-like
        ref_params=None,         # VisionCellDataTable or None
        ref_eis: Optional[dict] = None,
        ref_run_path: str = "",
        confidence_scores: Optional[Dict[int, float]] = None,
        match_statuses: Optional[Dict[int, str]] = None,
    ):
        self._mapping = mapping                  # {current_id: ref_id}
        self._reverse = {v: k for k, v in mapping.items()}  # {ref_id: current_id}
        self._ref_stas = ref_stas
        self._ref_params = ref_params
        self._ref_eis = ref_eis
        self.ref_run_path = ref_run_path
        self._confidence = confidence_scores or {}
        self._statuses = match_statuses or {}

    # ------------------------------------------------------------------
    # Construction helpers
    # ------------------------------------------------------------------

    @classmethod
    def from_matching_report(
        cls,
        report,  # MatchingReport
        ref_dir: str | Path,
        ref_dataset: str,
        load_stas: bool = True,
        load_params: bool = True,
        load_eis: bool = False,
    ) -> "ReferenceBridge":
        """
        Build a bridge from a MatchingReport + reference Vision directory.

        Only loads the data types requested (STAs and params by default).
        EIs are not loaded by default since they're only needed for matching,
        not for display.
        """
        from .vision_integration import (
            load_sta_data,
            load_params_data,
            load_ei_data,
            VISION_LOADER_AVAILABLE,
        )

        ref_dir = Path(ref_dir)

        ref_stas = None
        ref_params = None
        ref_eis = None

        if not VISION_LOADER_AVAILABLE:
            logger.warning("visionloader not available; ReferenceBridge will be empty")
        else:
            if load_stas:
                try:
                    ref_stas = load_sta_data(ref_dir, ref_dataset)
                    logger.info("Reference STAs loaded: %d cells",
                                len(ref_stas) if ref_stas else 0)
                except Exception as e:
                    logger.warning("Failed to load reference STAs: %s", e)

            if load_params:
                try:
                    ref_params = load_params_data(ref_dir, ref_dataset)
                    logger.info("Reference params loaded")
                except Exception as e:
                    logger.warning("Failed to load reference params: %s", e)

            if load_eis:
                try:
                    ei_bundle = load_ei_data(ref_dir, ref_dataset)
                    ref_eis = ei_bundle.get("ei_data") if ei_bundle else None
                except Exception as e:
                    logger.warning("Failed to load reference EIs: %s", e)

        # Extract confidence/status metadata from report
        confidence = {}
        statuses = {}
        for m in report.matches:
            if m.reference_id is not None:
                confidence[m.current_id] = m.confidence
                statuses[m.current_id] = m.status

        return cls(
            mapping=report.mapping,
            ref_stas=ref_stas,
            ref_params=ref_params,
            ref_eis=ref_eis,
            ref_run_path=str(ref_dir),
            confidence_scores=confidence,
            match_statuses=statuses,
        )

    # ------------------------------------------------------------------
    # Public API — called by panels / DataManager
    # ------------------------------------------------------------------

    def has_match(self, current_id: int) -> bool:
        """Does this current-run cell have a matched reference cell?"""
        return current_id in self._mapping

    def get_reference_id(self, current_id: int) -> Optional[int]:
        """Return the matched reference-run Vision ID, or None."""
        return self._mapping.get(current_id)

    def get_confidence(self, current_id: int) -> float:
        """Return match confidence (correlation) for this cell."""
        return self._confidence.get(current_id, 0.0)

    def get_status(self, current_id: int) -> str:
        """Return match status: 'high', 'marginal', or ''."""
        return self._statuses.get(current_id, "")

    @property
    def matched_current_ids(self) -> Set[int]:
        """All current-run IDs that have a reference match."""
        return set(self._mapping.keys())

    @property
    def mapping(self) -> Dict[int, int]:
        return dict(self._mapping)

    # --- STA access ---

    def get_sta(self, current_id: int):
        """
        Get STA data for a current-run cell from the reference run.

        Returns the STA object (with .red, .green, .blue arrays) or None.
        This is the same object type that LazySTADict returns.
        """
        ref_id = self._mapping.get(current_id)
        if ref_id is None or self._ref_stas is None:
            return None

        if ref_id not in self._ref_stas:
            return None

        try:
            return self._ref_stas[ref_id]
        except Exception as e:
            logger.debug("Failed to get reference STA for %d→%d: %s",
                         current_id, ref_id, e)
            return None

    def has_sta(self, current_id: int) -> bool:
        """Check if a reference STA is available without loading it."""
        ref_id = self._mapping.get(current_id)
        if ref_id is None or self._ref_stas is None:
            return False
        return ref_id in self._ref_stas

    # --- RF params access ---

    def get_stafit(self, current_id: int):
        """
        Get the Vision STAFit (RF Gaussian fit) for a matched cell.

        Returns the stafit object or None.
        """
        ref_id = self._mapping.get(current_id)
        if ref_id is None or self._ref_params is None:
            return None

        try:
            return self._ref_params.get_stafit_for_cell(ref_id)
        except Exception:
            return None

    def get_rf_center(self, current_id: int):
        """
        Get RF center (x0, y0) for a matched cell.

        Returns (x0, y0) tuple or None.
        """
        stafit = self.get_stafit(current_id)
        if stafit is None:
            return None
        try:
            import numpy as np
            x, y = stafit.center_x, stafit.center_y
            if np.isfinite(x) and np.isfinite(y):
                return (x, y)
        except AttributeError:
            pass
        return None

    def get_rf_ellipse_params(self, current_id: int):
        """
        Get RF ellipse parameters for population plot overlay.

        Returns dict with keys: x0, y0, std_x, std_y, angle
        or None if unavailable.
        """
        stafit = self.get_stafit(current_id)
        if stafit is None:
            return None

        try:
            import numpy as np
            params = {
                "x0": stafit.center_x,
                "y0": stafit.center_y,
                "std_x": stafit.std_x,
                "std_y": stafit.std_y,
                "angle": getattr(stafit, "angle", 0.0),
            }
            # Validate
            if all(np.isfinite(v) for v in params.values()):
                return params
        except (AttributeError, TypeError):
            pass
        return None

    # --- EI access (optional) ---

    def get_ei(self, current_id: int):
        """Get reference EI for a matched cell. Only available if load_eis=True."""
        ref_id = self._mapping.get(current_id)
        if ref_id is None or self._ref_eis is None:
            return None
        return self._ref_eis.get(ref_id)

    # --- Bulk accessors for population panels ---

    def get_all_rf_ellipses(self) -> Dict[int, dict]:
        """
        Return RF ellipse params for all matched cells.

        Returns {current_id: {x0, y0, std_x, std_y, angle}}
        For use by population_panel to overlay borrowed RFs.
        """
        result = {}
        for current_id in self._mapping:
            params = self.get_rf_ellipse_params(current_id)
            if params is not None:
                result[current_id] = params
        return result

    def get_all_stas_available(self) -> Set[int]:
        """Return set of current-run IDs for which a reference STA exists."""
        if self._ref_stas is None:
            return set()
        return {
            cid for cid, rid in self._mapping.items()
            if rid in self._ref_stas
        }

    # ------------------------------------------------------------------
    # Metadata
    # ------------------------------------------------------------------

    def summary(self) -> str:
        n_total = len(self._mapping)
        n_high = sum(1 for s in self._statuses.values() if s == "high")
        n_marginal = sum(1 for s in self._statuses.values() if s == "marginal")
        has_stas = self._ref_stas is not None
        has_params = self._ref_params is not None
        return (
            f"ReferenceBridge: {n_total} mapped cells "
            f"({n_high} high, {n_marginal} marginal), "
            f"STAs={'yes' if has_stas else 'no'}, "
            f"params={'yes' if has_params else 'no'}, "
            f"ref={self.ref_run_path}"
        )

    def __repr__(self):
        return f"<ReferenceBridge {len(self._mapping)} cells → {self.ref_run_path}>"