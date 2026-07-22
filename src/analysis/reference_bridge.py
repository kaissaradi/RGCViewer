"""
reference_bridge.py
===================
Lightweight adapter that provides analysis products (STAs, RF params, chirp,
grating) from a matched reference run, keyed by the *current* run's Vision IDs.

MatchingReport.mapping is always {current_vision_id → reference_vision_id}.
UI / feature_cache / UMAP use cluster_id (KS 0-indexed in hybrid mode). Use
``CellMatchCaveat`` / ``build_ui_caveats`` for per-original-run cell caveats.

Typical lifecycle:
    1. User triggers "Map Reference Run..." from the menu
    2. CrossRunMatcher produces a MatchingReport (or loads from JSON)
    3. ReferenceBridge is created with the report + reference vision dir
    4. DataManager.install_reference_bridge() installs it, builds caveats,
       invalidates physics that can now be filled from the reference
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Per-cell caveats (original-run UI IDs)
# ---------------------------------------------------------------------------

@dataclass
class CellMatchCaveat:
    """Match quality + data provenance for one current-run cell (UI id space)."""

    cluster_id: int
    current_vision_id: int
    reference_id: Optional[int]
    status: str  # "high" | "marginal" | "unmatched" | "conflict" | ""
    confidence: float
    next_best_corr: float = 0.0
    next_best_id: Optional[int] = None
    tier: str = "ei"
    provenance: Dict[str, Optional[str]] = field(default_factory=dict)
    # provenance keys: "timecourse", "rf_geometry", "chirp", "grating"
    # values: "current" | "reference" | None
    reference_run_path: str = ""
    stimuli_available_on_ref: Tuple[str, ...] = ()

    def to_dict(self) -> dict:
        return {
            "cluster_id": self.cluster_id,
            "current_vision_id": self.current_vision_id,
            "reference_id": self.reference_id,
            "status": self.status,
            "confidence": float(self.confidence),
            "next_best_corr": float(self.next_best_corr),
            "next_best_id": self.next_best_id,
            "tier": self.tier,
            "provenance": dict(self.provenance),
            "reference_run_path": self.reference_run_path,
            "stimuli_available_on_ref": list(self.stimuli_available_on_ref),
        }


def vision_id_to_cluster_id(vision_id: int, is_vision_only: bool) -> int:
    """Vision file key → UI cluster_id."""
    return int(vision_id) if is_vision_only else int(vision_id) - 1


def cluster_id_to_vision_id(cluster_id: int, is_vision_only: bool) -> int:
    """UI cluster_id → Vision file key."""
    return int(cluster_id) if is_vision_only else int(cluster_id) + 1


class ReferenceBridge:
    """
    Serves analysis products from a reference run using a pre-computed
    ID mapping {current_vision_id → reference_vision_id}.
    """

    def __init__(
        self,
        mapping: Dict[int, int],
        ref_stas=None,
        ref_params=None,
        ref_eis: Optional[dict] = None,
        ref_run_path: str = "",
        confidence_scores: Optional[Dict[int, float]] = None,
        match_statuses: Optional[Dict[int, str]] = None,
        match_meta: Optional[Dict[int, dict]] = None,
        ref_chirp_data=None,
        ref_chirp_id_to_row: Optional[Dict[int, int]] = None,
        ref_grating_data: Optional[dict] = None,
        stimuli_loaded: Optional[Tuple[str, ...]] = None,
        full_matches=None,
    ):
        self._mapping = mapping  # {current_vision_id: ref_vision_id}
        self._reverse = {v: k for k, v in mapping.items()}
        self._ref_stas = ref_stas
        self._ref_params = ref_params
        self._ref_eis = ref_eis
        self.ref_run_path = ref_run_path
        self._confidence = confidence_scores or {}
        self._statuses = match_statuses or {}
        # Full match rows keyed by current_vision_id (includes unmatched)
        self._match_meta = match_meta or {}
        self._full_matches = full_matches  # optional list of MatchResult

        # Chirp: same schema as DataManager.chirp_data; rows keyed by
        # reference-run *cluster_id* (KS space after load convention).
        self._ref_chirp_data = ref_chirp_data
        self._ref_chirp_id_to_row = ref_chirp_id_to_row or {}

        # Grating: analyzed per-cell dict keyed by reference cluster_id
        self._ref_grating_data = ref_grating_data

        self.stimuli_loaded: Tuple[str, ...] = stimuli_loaded or ()

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
        load_chirp: bool = True,
        load_grating: bool = True,
        ref_is_vision_only: bool = False,
    ) -> "ReferenceBridge":
        """
        Build a bridge from a MatchingReport + reference Vision directory.

        Loads STAs/params by default, plus chirp/grating npy files if present
        in *ref_dir* (MVP: only the selected directory is searched).
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
        stimuli: List[str] = []

        if not VISION_LOADER_AVAILABLE:
            logger.warning("visionloader not available; ReferenceBridge STA/params empty")
        else:
            if load_stas:
                try:
                    ref_stas = load_sta_data(ref_dir, ref_dataset)
                    if ref_stas is not None:
                        stimuli.append("sta")
                        logger.info(
                            "Reference STAs loaded: %d cells",
                            len(ref_stas) if ref_stas else 0,
                        )
                except Exception as e:
                    logger.warning("Failed to load reference STAs: %s", e)

            if load_params:
                try:
                    ref_params = load_params_data(ref_dir, ref_dataset)
                    if ref_params is not None:
                        if "params" not in stimuli:
                            stimuli.append("params")
                        logger.info("Reference params loaded")
                except Exception as e:
                    logger.warning("Failed to load reference params: %s", e)

            if load_eis:
                try:
                    ei_bundle = load_ei_data(ref_dir, ref_dataset)
                    ref_eis = ei_bundle.get("ei_data") if ei_bundle else None
                except Exception as e:
                    logger.warning("Failed to load reference EIs: %s", e)

        ref_chirp_data = None
        ref_chirp_id_to_row = None
        if load_chirp:
            ref_chirp_data, ref_chirp_id_to_row = cls._try_load_chirp(
                ref_dir, ref_is_vision_only=ref_is_vision_only
            )
            if ref_chirp_data is not None:
                stimuli.append("chirp")

        ref_grating_data = None
        if load_grating:
            ref_grating_data = cls._try_load_grating(
                ref_dir, ref_is_vision_only=ref_is_vision_only
            )
            if ref_grating_data is not None:
                stimuli.append("grating")

        confidence: Dict[int, float] = {}
        statuses: Dict[int, str] = {}
        match_meta: Dict[int, dict] = {}
        for m in report.matches:
            match_meta[m.current_id] = {
                "reference_id": m.reference_id,
                "confidence": float(m.confidence),
                "status": m.status,
                "next_best_corr": float(getattr(m, "next_best_corr", 0.0) or 0.0),
                "next_best_id": getattr(m, "next_best_id", None),
                "tier": getattr(m, "tier", "ei") or "ei",
            }
            if m.reference_id is not None and m.status in ("high", "marginal"):
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
            match_meta=match_meta,
            ref_chirp_data=ref_chirp_data,
            ref_chirp_id_to_row=ref_chirp_id_to_row,
            ref_grating_data=ref_grating_data,
            stimuli_loaded=tuple(stimuli),
            full_matches=list(report.matches),
        )

    @staticmethod
    def _try_load_chirp(
        ref_dir: Path, ref_is_vision_only: bool = False
    ) -> Tuple[Optional[dict], Optional[Dict[int, int]]]:
        """Load first *Chirp*.npy in ref_dir. Returns (mdic, id_to_row) or (None, None)."""
        try:
            candidates = sorted(ref_dir.glob("*Chirp*.npy"))
            if not candidates:
                logger.info("No chirp file in reference dir %s", ref_dir)
                return None, None

            path = candidates[0]
            mdic = np.load(path, allow_pickle=True).item()
            required = ("psth_mean", "cluster_id", "quality_index", "bin_size_ms")
            missing = [k for k in required if k not in mdic]
            if missing:
                logger.warning(
                    "Reference chirp %s missing keys %s", path.name, missing
                )
                return None, None

            id_to_row: Dict[int, int] = {}
            for i, cid in enumerate(mdic["cluster_id"]):
                # Same convention as DataManager.load_chirp_data: file IDs are
                # Vision-keyed unless the reference run is vision-only.
                ks_id = int(cid) if ref_is_vision_only else int(cid) - 1
                id_to_row[ks_id] = i

            logger.info(
                "Reference chirp loaded from %s (%d cells)", path.name, len(id_to_row)
            )
            return mdic, id_to_row
        except Exception as e:
            logger.warning("Failed to load reference chirp: %s", e)
            return None, None

    @staticmethod
    def _try_load_grating(
        ref_dir: Path, ref_is_vision_only: bool = False
    ) -> Optional[dict]:
        """Load analyzed *Grating*/*DSOS* npy if present. Raw-only skipped for MVP."""
        try:
            candidates = sorted(set(
                list(ref_dir.glob("*Grating*.npy")) + list(ref_dir.glob("*DSOS*.npy"))
            ))
            if not candidates:
                logger.info("No grating file in reference dir %s", ref_dir)
                return None

            analyzed = None
            for p in candidates:
                try:
                    mdic = np.load(p, allow_pickle=True).item()
                except Exception:
                    continue
                if not ReferenceBridge._grating_looks_analyzed(mdic):
                    continue
                if analyzed is None or "_combined" in p.name:
                    analyzed = (p, mdic)

            if analyzed is None:
                logger.info(
                    "Grating file(s) in %s but none analyzed (raw_only not bridged)",
                    ref_dir,
                )
                return None

            path, mdic = analyzed
            out = {}
            for k, v in mdic.items():
                if isinstance(k, (int, np.integer)):
                    ks_id = int(k) if ref_is_vision_only else int(k) - 1
                    out[ks_id] = v

            logger.info(
                "Reference grating loaded from %s (%d cells)", path.name, len(out)
            )
            return out
        except Exception as e:
            logger.warning("Failed to load reference grating: %s", e)
            return None

    @staticmethod
    def _grating_looks_analyzed(mdic) -> bool:
        sample_vals = [v for k, v in mdic.items() if isinstance(k, (int, np.integer))]
        if not sample_vals:
            return False
        first = sample_vals[0]
        if not isinstance(first, dict):
            return False
        return any(
            isinstance(k, tuple) and isinstance(v, dict) and "condition_type" in v
            for k, v in first.items()
        )

    # ------------------------------------------------------------------
    # Match / mapping API (Vision IDs)
    # ------------------------------------------------------------------

    def has_match(self, current_vision_id: int) -> bool:
        return int(current_vision_id) in self._mapping

    def get_reference_id(self, current_vision_id: int) -> Optional[int]:
        return self._mapping.get(int(current_vision_id))

    def get_confidence(self, current_vision_id: int) -> float:
        return float(self._confidence.get(int(current_vision_id), 0.0))

    def get_status(self, current_vision_id: int) -> str:
        return self._statuses.get(int(current_vision_id), "")

    @property
    def matched_current_ids(self) -> Set[int]:
        """Current-run Vision IDs with an accepted match."""
        return set(self._mapping.keys())

    @property
    def mapping(self) -> Dict[int, int]:
        return dict(self._mapping)

    # ------------------------------------------------------------------
    # STA / RF (Vision IDs)
    # ------------------------------------------------------------------

    def get_sta(self, current_vision_id: int):
        ref_id = self._mapping.get(int(current_vision_id))
        if ref_id is None or self._ref_stas is None:
            return None
        if ref_id not in self._ref_stas:
            return None
        try:
            return self._ref_stas[ref_id]
        except Exception as e:
            logger.debug(
                "Failed to get reference STA for %d→%d: %s",
                current_vision_id,
                ref_id,
                e,
            )
            return None

    def has_sta(self, current_vision_id: int) -> bool:
        ref_id = self._mapping.get(int(current_vision_id))
        if ref_id is None or self._ref_stas is None:
            return False
        return ref_id in self._ref_stas

    def get_stafit(self, current_vision_id: int):
        ref_id = self._mapping.get(int(current_vision_id))
        if ref_id is None or self._ref_params is None:
            return None
        try:
            return self._ref_params.get_stafit_for_cell(ref_id)
        except Exception:
            return None

    def has_rf(self, current_vision_id: int) -> bool:
        stafit = self.get_stafit(current_vision_id)
        if stafit is None:
            return False
        try:
            vals = (stafit.center_x, stafit.center_y, stafit.std_x, stafit.std_y)
            return all(np.isfinite(v) for v in vals) and stafit.std_x > 0 and stafit.std_y > 0
        except Exception:
            return False

    def get_rf_center(self, current_vision_id: int):
        stafit = self.get_stafit(current_vision_id)
        if stafit is None:
            return None
        try:
            x, y = stafit.center_x, stafit.center_y
            if np.isfinite(x) and np.isfinite(y):
                return (x, y)
        except AttributeError:
            pass
        return None

    def get_rf_ellipse_params(self, current_vision_id: int):
        """
        RF ellipse params for population overlay.

        Returns dict: x0, y0, std_x, std_y, angle (radians, Vision rot)
        or None.
        """
        stafit = self.get_stafit(current_vision_id)
        if stafit is None:
            return None
        try:
            rot = getattr(stafit, "rot", getattr(stafit, "angle", 0.0))
            params = {
                "x0": stafit.center_x,
                "y0": stafit.center_y,
                "std_x": stafit.std_x,
                "std_y": stafit.std_y,
                "angle": rot,
            }
            if all(np.isfinite(v) for v in params.values()):
                if params["std_x"] > 0 and params["std_y"] > 0:
                    return params
        except (AttributeError, TypeError):
            pass
        return None

    def get_ei(self, current_vision_id: int):
        ref_id = self._mapping.get(int(current_vision_id))
        if ref_id is None or self._ref_eis is None:
            return None
        return self._ref_eis.get(ref_id)

    # ------------------------------------------------------------------
    # Chirp / grating (Vision IDs for match; data keyed by ref cluster_id)
    # ------------------------------------------------------------------

    def has_any_chirp(self) -> bool:
        return (
            self._ref_chirp_data is not None
            and bool(self._ref_chirp_id_to_row)
            and len(self._ref_chirp_id_to_row) > 0
        )

    def has_any_grating(self) -> bool:
        return bool(self._ref_grating_data)

    def _ref_cluster_id(self, current_vision_id: int, ref_is_vision_only: bool = False) -> Optional[int]:
        """Map current vision id → reference cluster_id used in chirp/grating maps."""
        ref_vid = self._mapping.get(int(current_vision_id))
        if ref_vid is None:
            return None
        return int(ref_vid) if ref_is_vision_only else int(ref_vid) - 1

    def has_chirp(self, current_vision_id: int, ref_is_vision_only: bool = False) -> bool:
        if not self.has_any_chirp():
            return False
        ref_cid = self._ref_cluster_id(current_vision_id, ref_is_vision_only)
        if ref_cid is None:
            return False
        return ref_cid in self._ref_chirp_id_to_row

    def get_chirp_row(self, current_vision_id: int, ref_is_vision_only: bool = False):
        """
        Return (psth_mean 1d array, quality_index float) or None.
        """
        if not self.has_any_chirp():
            return None
        ref_cid = self._ref_cluster_id(current_vision_id, ref_is_vision_only)
        if ref_cid is None:
            return None
        row = self._ref_chirp_id_to_row.get(ref_cid)
        if row is None:
            return None
        try:
            psth = np.asarray(self._ref_chirp_data["psth_mean"][row], dtype=np.float64)
            qi = float(self._ref_chirp_data["quality_index"][row])
            return psth, qi
        except Exception as e:
            logger.debug("get_chirp_row failed for %s: %s", current_vision_id, e)
            return None

    def has_grating(self, current_vision_id: int, ref_is_vision_only: bool = False) -> bool:
        if not self.has_any_grating():
            return False
        ref_cid = self._ref_cluster_id(current_vision_id, ref_is_vision_only)
        if ref_cid is None:
            return False
        return ref_cid in self._ref_grating_data

    def get_grating_entry(self, current_vision_id: int, ref_is_vision_only: bool = False):
        """Return analyzed grating per-condition dict for matched cell, or None."""
        if not self.has_any_grating():
            return None
        ref_cid = self._ref_cluster_id(current_vision_id, ref_is_vision_only)
        if ref_cid is None:
            return None
        return self._ref_grating_data.get(ref_cid)

    # ------------------------------------------------------------------
    # Bulk accessors
    # ------------------------------------------------------------------

    def get_all_rf_ellipses(self) -> Dict[int, dict]:
        """{current_vision_id: {x0, y0, std_x, std_y, angle}}"""
        result = {}
        for current_id in self._mapping:
            params = self.get_rf_ellipse_params(current_id)
            if params is not None:
                result[current_id] = params
        return result

    def get_all_stas_available(self) -> Set[int]:
        if self._ref_stas is None:
            return set()
        return {
            cid for cid, rid in self._mapping.items() if rid in self._ref_stas
        }

    # ------------------------------------------------------------------
    # Caveats for original-run UI IDs
    # ------------------------------------------------------------------

    def build_ui_caveats(self, is_vision_only: bool = False) -> Dict[int, CellMatchCaveat]:
        """
        Build {cluster_id: CellMatchCaveat} for every cell in the matching report
        (including unmatched/conflict). Keys are original-run UI IDs.
        """
        stimuli = self.stimuli_loaded
        caveats: Dict[int, CellMatchCaveat] = {}

        # Prefer full match list so unmatched cells are included
        if self._full_matches is not None:
            rows = [
                {
                    "current_id": m.current_id,
                    "reference_id": m.reference_id,
                    "confidence": float(m.confidence),
                    "status": m.status,
                    "next_best_corr": float(getattr(m, "next_best_corr", 0.0) or 0.0),
                    "next_best_id": getattr(m, "next_best_id", None),
                    "tier": getattr(m, "tier", "ei") or "ei",
                }
                for m in self._full_matches
            ]
        else:
            rows = []
            for vid, meta in self._match_meta.items():
                rows.append({"current_id": vid, **meta})
            for vid in self._mapping:
                if not any(r["current_id"] == vid for r in rows):
                    rows.append(
                        {
                            "current_id": vid,
                            "reference_id": self._mapping[vid],
                            "confidence": self._confidence.get(vid, 0.0),
                            "status": self._statuses.get(vid, "high"),
                            "next_best_corr": 0.0,
                            "next_best_id": None,
                            "tier": "ei",
                        }
                    )

        for r in rows:
            vid = int(r["current_id"])
            cid = vision_id_to_cluster_id(vid, is_vision_only)
            caveats[cid] = CellMatchCaveat(
                cluster_id=cid,
                current_vision_id=vid,
                reference_id=r.get("reference_id"),
                status=r.get("status") or "",
                confidence=float(r.get("confidence") or 0.0),
                next_best_corr=float(r.get("next_best_corr") or 0.0),
                next_best_id=r.get("next_best_id"),
                tier=r.get("tier") or "ei",
                provenance={
                    "timecourse": None,
                    "rf_geometry": None,
                    "chirp": None,
                    "grating": None,
                },
                reference_run_path=self.ref_run_path,
                stimuli_available_on_ref=stimuli,
            )
        return caveats

    def get_caveat(
        self, cluster_id: int, is_vision_only: bool = False
    ) -> Optional[CellMatchCaveat]:
        """Convenience single-cell caveat (rebuilds dict — prefer dm.match_caveats)."""
        return self.build_ui_caveats(is_vision_only=is_vision_only).get(int(cluster_id))

    # ------------------------------------------------------------------
    # Metadata
    # ------------------------------------------------------------------

    def summary(self) -> str:
        n_total = len(self._mapping)
        n_high = sum(1 for s in self._statuses.values() if s == "high")
        n_marginal = sum(1 for s in self._statuses.values() if s == "marginal")
        stim = ",".join(self.stimuli_loaded) if self.stimuli_loaded else "none"
        return (
            f"ReferenceBridge: {n_total} mapped cells "
            f"({n_high} high, {n_marginal} marginal), "
            f"stimuli=[{stim}], ref={self.ref_run_path}"
        )

    def __repr__(self):
        return (
            f"<ReferenceBridge {len(self._mapping)} cells "
            f"stimuli={self.stimuli_loaded} → {self.ref_run_path}>"
        )
