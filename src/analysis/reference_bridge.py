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
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import numpy as np

from . import rf_geometry

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
        ref_grating_raw: Optional[dict] = None,
        stimuli_loaded: Optional[Tuple[str, ...]] = None,
        full_matches=None,
        ref_id_shift: int = 1,
    ):
        self._mapping = mapping  # {current_vision_id: ref_vision_id}
        # The chirp/grating maps are keyed by reference Vision ID − this shift:
        # 1 (Kilosort cluster IDs) or 0 (a Vision-only session keeps Vision IDs).
        # Lookups must use the shift the maps were loaded with; a per-call
        # default of 1 served the neighbouring cell in Vision-only sessions.
        self._ref_id_shift = int(ref_id_shift)
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
        # Or the raw trials (every lab grating file is raw, 2026-09-25):
        # {"spike_times_by_trial": {ref cluster_id: trials}, "trial_parameters",
        # "source"}. A matched cell's DS/OS is computed once, on first use,
        # with the Grating tab's own test (grating_calc).
        self._ref_grating_raw = ref_grating_raw
        self._grating_computed: Dict[int, Optional[dict]] = {}
        self._grating_lock = threading.Lock()

        self.stimuli_loaded: Tuple[str, ...] = stimuli_loaded or ()
        # Set when the reference gratings came from a Lisp file (Q61).
        self.grating_source = None
        self.grating_spatial_label = None

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

        ref_grating_data = ref_grating_raw = None
        lisp_summary = None
        if load_grating:
            ref_grating_data, ref_grating_raw = cls._try_load_grating_files(
                ref_dir, ref_is_vision_only=ref_is_vision_only
            )
            if ref_grating_data is None and ref_grating_raw is None:
                # An older run: its gratings are in a Lisp sequence file (Q61).
                ref_grating_raw, lisp_summary = cls._try_load_lisp_grating(
                    ref_dir, ref_dataset, ref_is_vision_only=ref_is_vision_only)
            if ref_grating_data is not None or ref_grating_raw is not None:
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

        bridge = cls(
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
            ref_grating_raw=ref_grating_raw,
            stimuli_loaded=tuple(stimuli),
            full_matches=list(report.matches),
            ref_id_shift=0 if ref_is_vision_only else 1,
        )
        if lisp_summary is not None:
            bridge.grating_source = lisp_summary
            bridge.grating_spatial_label = ("period", "px")
        return bridge

    @staticmethod
    def _try_load_lisp_grating(ref_dir: Path, ref_dataset: str, ref_is_vision_only: bool = False):
        """(raw trials, summary) from the reference run's Lisp sequence file, or (None, None).

        Only a file named after the run next to it (``stimuli/s02`` for data002,
        lisp_stimulus.find_sequence_files); triggers and spikes from its .neurons.
        """
        from . import lisp_stimulus as ls
        try:
            found = ls.find_sequence_files(ref_dir, ref_dataset)
            if not found:
                return None, None
            import src.analysis.visionloader as vl
            seq = ls.read_sequence(found[0])
            with vl.NeuronsReader(str(ref_dir), ref_dataset) as nr:
                ttl = nr.get_TTL_times()
                spikes = nr.get_spike_sample_nums_for_all_real_neurons()
                fs = float(nr.sample_freq)
            shift = 0 if ref_is_vision_only else 1
            ends = [int(v[-1]) for v in spikes.values() if len(v)]
            g = ls.build_grating_trials(seq, ttl, {int(k) - shift: v for k, v in spikes.items()},
                                        fs, recording_end_samples=max(ends) if ends else None)
        except Exception as exc:
            logger.info("No Lisp grating for reference %s: %s", ref_dir, exc)
            return None, None
        raw = g.as_raw()
        raw["source"] = Path(found[0]).name
        logger.info("Reference grating trials from Lisp %s: %s", found[0], ls.describe(g.summary))
        return raw, g.summary

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
        """The analysed grating dict only (see _try_load_grating_files)."""
        return ReferenceBridge._try_load_grating_files(ref_dir, ref_is_vision_only)[0]

    @staticmethod
    def _try_load_grating_files(ref_dir: Path, ref_is_vision_only: bool = False):
        """(analysed, raw) from the reference run's *Grating*/*DSOS* npy files, one read each.

        analysed: {ref cluster_id: per-condition dict}. raw: {"spike_times_by_trial":
        {ref cluster_id: trials}, "trial_parameters", "source"}. Either may be None.
        The files are keyed by Vision ID; cluster_id = Vision ID − 1 unless the
        reference was opened Vision-only (CLAUDE.md trap 1).
        """
        ref_dir = Path(ref_dir)
        shift = 0 if ref_is_vision_only else 1
        try:
            candidates = sorted(set(
                list(ref_dir.glob("*Grating*.npy")) + list(ref_dir.glob("*DSOS*.npy"))
            ))
        except OSError:
            return None, None
        if not candidates:
            logger.info("No grating file in reference dir %s", ref_dir)
        analyzed = raw = None
        for p in candidates:
            try:
                mdic = np.load(p, allow_pickle=True).item()
            except Exception:
                continue
            if not isinstance(mdic, dict):
                continue
            if ReferenceBridge._grating_looks_analyzed(mdic):
                if analyzed is None or "_combined" in p.name:
                    analyzed = {int(k) - shift: v for k, v in mdic.items()
                                if isinstance(k, (int, np.integer))}
                    logger.info("Reference grating loaded from %s (%d cells)", p.name, len(analyzed))
            elif raw is None and "trial_parameters" in mdic and "spike_times_by_trial" in mdic:
                raw = {"spike_times_by_trial": {int(k) - shift: v
                                                for k, v in mdic["spike_times_by_trial"].items()},
                       "trial_parameters": mdic["trial_parameters"], "source": p.name}
                logger.info("Reference grating trials from %s (%d cells)", p.name,
                            len(raw["spike_times_by_trial"]))
        return analyzed, raw

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

    def pick(self, current_vision_id: int, kind: str = "match") -> Optional["ReferenceBridge"]:
        """This bridge if it has ``kind`` for the cell, else None (see MultiBridge.pick).

        ``kind``: "match", "sta", "rf", "chirp" or "grating".
        """
        vid = int(current_vision_id)
        if not self.has_match(vid):
            return None
        ok = {"match": lambda: True,
              "sta": lambda: self.has_sta(vid),
              "rf": lambda: self.has_rf(vid),
              "chirp": lambda: self.has_chirp(vid),
              "grating": lambda: self.has_grating(vid)}[kind]()
        return self if ok else None

    @property
    def bridges(self) -> List["ReferenceBridge"]:
        return [self]

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

    def get_rf_fit(self, current_vision_id: int):
        """The reference run's stored RF fit (y-up, radians) or None.

        Read through rf_geometry so the centre is in Vision's stored frame
        whatever the reference table was loaded with.
        """
        ref_id = self._mapping.get(int(current_vision_id))
        if ref_id is None or self._ref_params is None:
            return None
        return rf_geometry.raw_rf_fit(self._ref_params, ref_id)

    def get_rf_ellipse_params(self, current_vision_id: int):
        """
        RF ellipse params for population overlay.

        Returns dict: x0, y0 (Vision's stored y-up frame), std_x, std_y,
        angle (radians, Vision Theta) or None. Draw it with
        rf_geometry.mosaic_ellipse / image_ellipse, not by hand.
        """
        fit = self.get_rf_fit(current_vision_id)
        if fit is None:
            return None
        return {"x0": fit.x0, "y0": fit.y0, "std_x": fit.std_x,
                "std_y": fit.std_y, "angle": fit.theta}

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
        return bool(self._ref_grating_data) or bool(self._ref_grating_raw)

    def _ref_cluster_id(self, current_vision_id: int, ref_is_vision_only=None) -> Optional[int]:
        """Map current vision id → reference key used in the chirp/grating maps.

        ``ref_is_vision_only`` None (the default) uses the shift the maps were
        loaded with; True/False force 0/1.
        """
        ref_vid = self._mapping.get(int(current_vision_id))
        if ref_vid is None:
            return None
        shift = self._ref_id_shift if ref_is_vision_only is None else (0 if ref_is_vision_only else 1)
        return int(ref_vid) - shift

    def has_chirp(self, current_vision_id: int, ref_is_vision_only=None) -> bool:
        if not self.has_any_chirp():
            return False
        ref_cid = self._ref_cluster_id(current_vision_id, ref_is_vision_only)
        if ref_cid is None:
            return False
        return ref_cid in self._ref_chirp_id_to_row

    def get_chirp_row(self, current_vision_id: int, ref_is_vision_only=None):
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

    def has_grating(self, current_vision_id: int, ref_is_vision_only=None) -> bool:
        if not self.has_any_grating():
            return False
        ref_cid = self._ref_cluster_id(current_vision_id, ref_is_vision_only)
        if ref_cid is None:
            return False
        if self._ref_grating_data and ref_cid in self._ref_grating_data:
            return True
        raw = self._ref_grating_raw
        return bool(raw) and ref_cid in raw["spike_times_by_trial"]

    def get_grating_entry(self, current_vision_id: int, ref_is_vision_only=None):
        """The matched cell's per-condition grating dict (analysed, or computed from trials)."""
        if not self.has_any_grating():
            return None
        ref_cid = self._ref_cluster_id(current_vision_id, ref_is_vision_only)
        if ref_cid is None:
            return None
        if self._ref_grating_data and ref_cid in self._ref_grating_data:
            return self._ref_grating_data[ref_cid]
        raw = self._ref_grating_raw
        if not raw or ref_cid not in raw["spike_times_by_trial"]:
            return None
        with self._grating_lock:
            if ref_cid in self._grating_computed:
                return self._grating_computed[ref_cid]
        from . import grating_calc
        entry = grating_calc.compute_grating_response(
            ref_cid, raw["spike_times_by_trial"], raw["trial_parameters"])
        with self._grating_lock:
            self._grating_computed[ref_cid] = entry
        return entry

    def get_grating_entry_if_ready(self, current_vision_id: int, ref_is_vision_only=None):
        """The matched cell's grating dict if analysed or already scored; never computes.

        For drawing many cells on the GUI thread (population arrows, table
        columns): scoring is ~10 ms a cell, so it happens once, in the
        background, in ``precompute_gratings``.
        """
        if not self.has_any_grating():
            return None
        ref_cid = self._ref_cluster_id(current_vision_id, ref_is_vision_only)
        if ref_cid is None:
            return None
        if self._ref_grating_data and ref_cid in self._ref_grating_data:
            return self._ref_grating_data[ref_cid]
        with self._grating_lock:
            return self._grating_computed.get(ref_cid)

    def precompute_gratings(self, cancelled=None) -> int:
        """Score every matched cell's raw grating trials once. Returns the number scored."""
        n = 0
        if not self._ref_grating_raw:
            return n
        for vid in list(self._mapping):
            if cancelled is not None and cancelled():
                break
            if self.has_grating(vid) and self.get_grating_entry(vid) is not None:
                n += 1
        return n

    def get_grating_trials(self, current_vision_id: int, ref_is_vision_only=None):
        """(trials, trial_parameters) of the matched cell, for rasters; None without raw trials."""
        raw = self._ref_grating_raw
        ref_cid = self._ref_cluster_id(current_vision_id, ref_is_vision_only)
        if not raw or ref_cid is None or ref_cid not in raw["spike_times_by_trial"]:
            return None
        return raw["spike_times_by_trial"][ref_cid], raw["trial_parameters"]

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


class MultiBridge:
    """Several matched runs behind the ReferenceBridge interface (PLAN.md Q62).

    File ▸ Match Runs can match this run to more than one other run. Each
    question is answered by the first run, in the order given, that has the
    answer for that cell: its chirp from one run, its grating from another.
    Callers that report where a value came from ask ``pick(vid, kind)`` for
    that run's bridge and read the run, reference ID and confidence from it.
    """

    def __init__(self, bridges):
        self._bridges = [b for b in bridges if b is not None]

    @property
    def bridges(self) -> List[ReferenceBridge]:
        return list(self._bridges)

    def pick(self, current_vision_id: int, kind: str = "match") -> Optional[ReferenceBridge]:
        for b in self._bridges:
            got = b.pick(current_vision_id, kind)
            if got is not None:
                return got
        return None

    # -- match ---------------------------------------------------------------
    def has_match(self, vid) -> bool:
        return any(b.has_match(vid) for b in self._bridges)

    def _first(self, vid, kind="match"):
        return self.pick(vid, kind)

    def get_reference_id(self, vid):
        b = self._first(vid)
        return None if b is None else b.get_reference_id(vid)

    def get_confidence(self, vid) -> float:
        b = self._first(vid)
        return 0.0 if b is None else b.get_confidence(vid)

    def get_status(self, vid) -> str:
        b = self._first(vid)
        return "" if b is None else b.get_status(vid)

    @property
    def matched_current_ids(self) -> Set[int]:
        out: Set[int] = set()
        for b in self._bridges:
            out |= b.matched_current_ids
        return out

    @property
    def mapping(self) -> Dict[int, int]:
        """First run's reference ID per cell (for display; per-run IDs via pick)."""
        out: Dict[int, int] = {}
        for b in reversed(self._bridges):
            out.update(b.mapping)
        return out

    @property
    def ref_run_path(self) -> str:
        return " + ".join(str(b.ref_run_path) for b in self._bridges)

    @property
    def stimuli_loaded(self) -> Tuple[str, ...]:
        seen: List[str] = []
        for b in self._bridges:
            seen += [s for s in b.stimuli_loaded if s not in seen]
        return tuple(seen)

    # -- per kind ------------------------------------------------------------
    def _call(self, vid, kind, name, default=None, *args):
        b = self.pick(vid, kind)
        return default if b is None else getattr(b, name)(vid, *args)

    def has_sta(self, vid) -> bool:
        return self.pick(vid, "sta") is not None

    def get_sta(self, vid):
        return self._call(vid, "sta", "get_sta")

    def has_rf(self, vid) -> bool:
        return self.pick(vid, "rf") is not None

    def get_stafit(self, vid):
        b = self.pick(vid, "rf") or self.pick(vid, "sta")
        return None if b is None else b.get_stafit(vid)

    def get_rf_center(self, vid):
        return self._call(vid, "rf", "get_rf_center")

    def get_rf_fit(self, vid):
        return self._call(vid, "rf", "get_rf_fit")

    def get_rf_ellipse_params(self, vid):
        return self._call(vid, "rf", "get_rf_ellipse_params")

    def get_ei(self, vid):
        return self._call(vid, "match", "get_ei")

    def has_any_chirp(self) -> bool:
        return any(b.has_any_chirp() for b in self._bridges)

    def has_any_grating(self) -> bool:
        return any(b.has_any_grating() for b in self._bridges)

    def has_chirp(self, vid, ref_is_vision_only=False) -> bool:
        return self.pick(vid, "chirp") is not None

    def get_chirp_row(self, vid, ref_is_vision_only=False):
        return self._call(vid, "chirp", "get_chirp_row")

    def has_grating(self, vid, ref_is_vision_only=False) -> bool:
        return self.pick(vid, "grating") is not None

    def get_grating_entry(self, vid, ref_is_vision_only=False):
        return self._call(vid, "grating", "get_grating_entry")

    def get_grating_entry_if_ready(self, vid, ref_is_vision_only=False):
        return self._call(vid, "grating", "get_grating_entry_if_ready")

    def get_grating_trials(self, vid, ref_is_vision_only=False):
        return self._call(vid, "grating", "get_grating_trials")

    def precompute_gratings(self, cancelled=None) -> int:
        return sum(b.precompute_gratings(cancelled) for b in self._bridges)

    def first_with(self, kind: str) -> Optional[ReferenceBridge]:
        """The first run that has any ``kind`` ("chirp" or "grating") at all."""
        test = {"chirp": "has_any_chirp", "grating": "has_any_grating"}[kind]
        return next((b for b in self._bridges if getattr(b, test)()), None)

    # -- bulk ----------------------------------------------------------------
    def get_all_rf_ellipses(self) -> Dict[int, dict]:
        out: Dict[int, dict] = {}
        for b in reversed(self._bridges):
            out.update(b.get_all_rf_ellipses())
        return out

    def get_all_stas_available(self) -> Set[int]:
        out: Set[int] = set()
        for b in self._bridges:
            out |= b.get_all_stas_available()
        return out

    def build_ui_caveats(self, is_vision_only: bool = False) -> Dict[int, CellMatchCaveat]:
        """Per cell, the first run's caveat that is a real match, else the first one."""
        out: Dict[int, CellMatchCaveat] = {}
        for b in self._bridges:
            for cid, cav in b.build_ui_caveats(is_vision_only).items():
                held = out.get(cid)
                if held is None or (held.status not in ("high", "marginal")
                                    and cav.status in ("high", "marginal")):
                    out[cid] = cav
        return out

    def get_caveat(self, cluster_id: int, is_vision_only: bool = False):
        return self.build_ui_caveats(is_vision_only).get(int(cluster_id))

    def summary(self) -> str:
        return " | ".join(b.summary() for b in self._bridges)

    def __repr__(self):
        return f"<MultiBridge {[Path(str(b.ref_run_path)).name for b in self._bridges]}>"
