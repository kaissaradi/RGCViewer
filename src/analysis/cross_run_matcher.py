"""
cross_run_matcher.py
====================
Cross-run neuron identity matching for Encore.

Matches neurons between two independent Kilosort sorts of the same retina/MEA
by comparing Electrical Images (EIs), with fallback to Kilosort templates and
optional RF-position validation.

The output is a mapping dict {current_run_id: reference_run_id} plus per-cell
confidence metadata, saved as a JSON sidecar for persistence across sessions.

Usage (standalone):
    from cross_run_matcher import CrossRunMatcher
    matcher = CrossRunMatcher(current_eis, reference_eis)
    result = matcher.run()
    result.save("mapping.json")

Usage (from GUI):
    matcher = CrossRunMatcher.from_vision_dirs(current_dir, current_ds,
                                                ref_dir, ref_ds)
    result = matcher.run()
"""

from __future__ import annotations

import json
import hashlib
import logging
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration defaults — mirrors the MATLAB map_ei defaults where applicable
# ---------------------------------------------------------------------------
DEFAULT_ELECTRODE_THRESHOLD = 5.0       # zero out electrodes below this amplitude
DEFAULT_SIGNIFICANT_ELECTRODES = 10     # min electrodes above threshold to include cell
DEFAULT_CORR_THRESHOLD = 0.85           # accept match (MATLAB uses 0.95; relaxed for cross-stim)
DEFAULT_MARGINAL_THRESHOLD = 0.70       # below this → unmatched; between this and corr_threshold → marginal
DEFAULT_METHOD = "space"                # "space" | "full" | "power" — mirrors ei_corr methods
DEFAULT_N_REMOVED_CHANNELS = 1          # remove top-N channels (axon cap artifact)


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------
@dataclass
class MatchResult:
    """One matched pair (or failed match) between current and reference run."""
    current_id: int
    reference_id: Optional[int]     # None if unmatched
    confidence: float               # correlation value
    tier: str                       # "ei" | "template" | "rf_validated"
    status: str                     # "high" | "marginal" | "unmatched" | "conflict"
    next_best_corr: float = 0.0     # second-best correlation (for margin info)
    next_best_id: Optional[int] = None


@dataclass
class MatchingReport:
    """Full output of a matching run."""
    matches: List[MatchResult] = field(default_factory=list)
    current_run_path: str = ""
    reference_run_path: str = ""
    method: str = DEFAULT_METHOD
    corr_threshold: float = DEFAULT_CORR_THRESHOLD
    marginal_threshold: float = DEFAULT_MARGINAL_THRESHOLD

    # --- Convenience accessors ------------------------------------------------

    @property
    def mapping(self) -> Dict[int, int]:
        """Return {current_id: reference_id} for accepted matches only."""
        return {
            m.current_id: m.reference_id
            for m in self.matches
            if m.reference_id is not None and m.status in ("high", "marginal")
        }

    @property
    def high_confidence(self) -> List[MatchResult]:
        return [m for m in self.matches if m.status == "high"]

    @property
    def marginal(self) -> List[MatchResult]:
        return [m for m in self.matches if m.status == "marginal"]

    @property
    def unmatched(self) -> List[MatchResult]:
        return [m for m in self.matches if m.status == "unmatched"]

    @property
    def conflicts(self) -> List[MatchResult]:
        return [m for m in self.matches if m.status == "conflict"]

    def summary(self) -> str:
        n = len(self.matches)
        h = len(self.high_confidence)
        mg = len(self.marginal)
        um = len(self.unmatched)
        cf = len(self.conflicts)
        return (
            f"Matched {h + mg}/{n} cells "
            f"({h} high-confidence, {mg} marginal, {cf} conflicts, {um} unmatched)"
        )

    # --- Persistence ----------------------------------------------------------

    def save(self, path: str | Path):
        """Save mapping + metadata to JSON."""
        path = Path(path)
        data = {
            "current_run": self.current_run_path,
            "reference_run": self.reference_run_path,
            "method": self.method,
            "corr_threshold": self.corr_threshold,
            "marginal_threshold": self.marginal_threshold,
            "summary": self.summary(),
            "mapping": {
                str(m.current_id): {
                    "reference_id": m.reference_id,
                    "confidence": round(float(m.confidence), 4),
                    "tier": m.tier,
                    "status": m.status,
                    "next_best_corr": round(float(m.next_best_corr), 4),
                    "next_best_id": m.next_best_id,
                }
                for m in self.matches
            },
        }
        path.write_text(json.dumps(data, indent=2))
        logger.info("Saved matching report to %s", path)

    @classmethod
    def load(cls, path: str | Path) -> "MatchingReport":
        """Load a previously saved mapping."""
        path = Path(path)
        data = json.loads(path.read_text())
        report = cls(
            current_run_path=data.get("current_run", ""),
            reference_run_path=data.get("reference_run", ""),
            method=data.get("method", DEFAULT_METHOD),
            corr_threshold=data.get("corr_threshold", DEFAULT_CORR_THRESHOLD),
            marginal_threshold=data.get("marginal_threshold", DEFAULT_MARGINAL_THRESHOLD),
        )
        for cid_str, info in data.get("mapping", {}).items():
            report.matches.append(MatchResult(
                current_id=int(cid_str),
                reference_id=info.get("reference_id"),
                confidence=info.get("confidence", 0.0),
                tier=info.get("tier", "ei"),
                status=info.get("status", "unmatched"),
                next_best_corr=info.get("next_best_corr", 0.0),
                next_best_id=info.get("next_best_id"),
            ))
        return report


# ---------------------------------------------------------------------------
# Core EI extraction helpers
# ---------------------------------------------------------------------------

def _extract_ei_matrix(
    ei_dict: dict,
    method: str = DEFAULT_METHOD,
    electrode_threshold: float = DEFAULT_ELECTRODE_THRESHOLD,
    significant_electrodes: int = DEFAULT_SIGNIFICANT_ELECTRODES,
    n_removed_channels: int = DEFAULT_N_REMOVED_CHANNELS,
) -> Tuple[np.ndarray, List[int], List[int]]:
    """
    Extract a (n_cells, n_features) matrix from a Vision EI dict.

    Returns:
        matrix:       (n_valid_cells, n_features)
        valid_ids:     cell IDs with sufficient signal
        excluded_ids:  cell IDs excluded for weak signal
    """
    valid_ids = []
    excluded_ids = []
    vectors = []

    for cell_id, entry in ei_dict.items():
        ei_arr = getattr(entry, "ei", None)
        if not isinstance(ei_arr, np.ndarray) or ei_arr.size == 0:
            excluded_ids.append(int(cell_id))
            continue

        ei = ei_arr.copy().astype(np.float64)

        # Remove top-N channels (axon cap artifact, same as ei_corr)
        if n_removed_channels > 0:
            max_per_ch = np.max(ei, axis=1)
            remove_idx = np.argsort(max_per_ch)[-n_removed_channels:]
            ei = np.delete(ei, remove_idx, axis=0)

        # Threshold: zero out sub-threshold electrodes (MATLAB-style)
        max_per_ch = np.max(ei, axis=1)
        ei[max_per_ch < electrode_threshold] = 0

        # Check significant electrode count
        n_sig = np.sum(max_per_ch >= electrode_threshold)
        if n_sig < significant_electrodes:
            excluded_ids.append(int(cell_id))
            continue

        # Noise floor: zero values below 1.5*std (same as ei_corr)
        ei_std = ei.std()
        if ei_std > 0:
            ei[np.abs(ei) < (ei_std * 1.5)] = 0

        # Reduce to feature vector based on method
        if method == "space":
            vec = np.max(np.abs(ei), axis=1)  # spatial footprint
        elif method == "full":
            vec = ei.flatten()
        elif method == "power":
            vec = np.mean(ei ** 2, axis=1)
        else:
            raise ValueError(f"Unknown method: {method!r}")

        vectors.append(vec)
        valid_ids.append(int(cell_id))

    if not vectors:
        return np.empty((0, 0)), [], excluded_ids

    # Ensure all vectors have same length (they should, but guard)
    min_len = min(v.shape[0] for v in vectors)
    matrix = np.array([v[:min_len] for v in vectors], dtype=np.float64)

    return matrix, valid_ids, excluded_ids


def _correlation_matrix(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """
    Compute correlation between rows of A and rows of B.
    Returns shape (n_A, n_B).
    """
    # Center
    A_centered = A - A.mean(axis=1, keepdims=True)
    B_centered = B - B.mean(axis=1, keepdims=True)

    # Norms
    A_norms = np.linalg.norm(A_centered, axis=1, keepdims=True)
    B_norms = np.linalg.norm(B_centered, axis=1, keepdims=True)

    # Avoid division by zero
    A_norms = np.where(A_norms == 0, 1.0, A_norms)
    B_norms = np.where(B_norms == 0, 1.0, B_norms)

    corr = (A_centered / A_norms) @ (B_centered / B_norms).T
    np.nan_to_num(corr, copy=False, nan=0.0, posinf=0.0, neginf=0.0)

    return corr


# ---------------------------------------------------------------------------
# Kilosort template extraction
# ---------------------------------------------------------------------------

def _extract_template_matrix(
    templates_path: str | Path,
    cluster_ids: Optional[List[int]] = None,
) -> Tuple[np.ndarray, List[int]]:
    """
    Load Kilosort templates.npy and build feature matrix.

    templates.npy shape: (n_clusters, n_timepoints, n_channels)

    Returns:
        matrix:  (n_clusters, n_features) — spatial footprint (max abs over time)
        ids:     cluster IDs (0-indexed Kilosort IDs)
    """
    templates = np.load(templates_path)  # (n_clusters, n_time, n_channels)

    if cluster_ids is None:
        cluster_ids = list(range(templates.shape[0]))

    vectors = []
    valid_ids = []
    for cid in cluster_ids:
        if cid >= templates.shape[0]:
            continue
        t = templates[cid]  # (n_time, n_channels)
        vec = np.max(np.abs(t), axis=0)  # spatial footprint across channels
        vectors.append(vec)
        valid_ids.append(cid)

    if not vectors:
        return np.empty((0, 0)), []

    return np.array(vectors, dtype=np.float64), valid_ids


# ---------------------------------------------------------------------------
# Assignment: mutual-best with thresholds (MATLAB map_ei style)
# ---------------------------------------------------------------------------

def _assign_mutual_best(
    corr_matrix: np.ndarray,
    current_ids: List[int],
    reference_ids: List[int],
    corr_threshold: float = DEFAULT_CORR_THRESHOLD,
    marginal_threshold: float = DEFAULT_MARGINAL_THRESHOLD,
    tier: str = "ei",
) -> List[MatchResult]:
    """
    Greedy mutual-best assignment from a correlation matrix.

    For each current cell, find its best reference match. Accept only if
    that reference cell's best current match is the same current cell
    (mutual best). This is the same logic as the MATLAB map_ei.

    Returns list of MatchResult.
    """
    n_curr, n_ref = corr_matrix.shape
    results = []

    for i in range(n_curr):
        row = corr_matrix[i]
        sorted_idx = np.argsort(row)
        best_j = sorted_idx[-1]
        best_corr = row[best_j]

        # Second-best for margin info
        next_best_corr = row[sorted_idx[-2]] if n_ref > 1 else 0.0
        next_best_id = reference_ids[sorted_idx[-2]] if n_ref > 1 else None

        if best_corr < marginal_threshold:
            results.append(MatchResult(
                current_id=current_ids[i],
                reference_id=None,
                confidence=float(best_corr),
                tier=tier,
                status="unmatched",
                next_best_corr=float(next_best_corr),
                next_best_id=next_best_id,
            ))
            continue

        # Mutual-best check: is current cell i also the best match for ref cell best_j?
        col = corr_matrix[:, best_j]
        best_i_for_j = np.argmax(col)

        if best_i_for_j != i:
            # Conflict: another current cell has a stronger claim on this ref cell
            results.append(MatchResult(
                current_id=current_ids[i],
                reference_id=reference_ids[best_j],
                confidence=float(best_corr),
                tier=tier,
                status="conflict",
                next_best_corr=float(next_best_corr),
                next_best_id=next_best_id,
            ))
            continue

        # Mutual best — classify by threshold
        status = "high" if best_corr >= corr_threshold else "marginal"
        results.append(MatchResult(
            current_id=current_ids[i],
            reference_id=reference_ids[best_j],
            confidence=float(best_corr),
            tier=tier,
            status=status,
            next_best_corr=float(next_best_corr),
            next_best_id=next_best_id,
        ))

    return results


# ---------------------------------------------------------------------------
# Main matcher class
# ---------------------------------------------------------------------------

class CrossRunMatcher:
    """
    Match neurons across two independent sorts of the same retina.

    Tier 1: EI correlation (primary)
    Tier 2: Kilosort templates (fallback)
    Tier 3: RF position validation (post-hoc check)
    """

    def __init__(
        self,
        current_eis: Optional[dict] = None,
        reference_eis: Optional[dict] = None,
        current_templates_path: Optional[str | Path] = None,
        reference_templates_path: Optional[str | Path] = None,
        current_cluster_ids: Optional[List[int]] = None,
        reference_cluster_ids: Optional[List[int]] = None,
        method: str = DEFAULT_METHOD,
        corr_threshold: float = DEFAULT_CORR_THRESHOLD,
        marginal_threshold: float = DEFAULT_MARGINAL_THRESHOLD,
        electrode_threshold: float = DEFAULT_ELECTRODE_THRESHOLD,
        significant_electrodes: int = DEFAULT_SIGNIFICANT_ELECTRODES,
        n_removed_channels: int = DEFAULT_N_REMOVED_CHANNELS,
        current_run_path: str = "",
        reference_run_path: str = "",
        vision_id_offset: int = 1,  # vision_id = cluster_id + 1
    ):
        self.current_eis = current_eis
        self.reference_eis = reference_eis
        self.current_templates_path = current_templates_path
        self.reference_templates_path = reference_templates_path
        self.current_cluster_ids = current_cluster_ids
        self.reference_cluster_ids = reference_cluster_ids
        self.method = method
        self.corr_threshold = corr_threshold
        self.marginal_threshold = marginal_threshold
        self.electrode_threshold = electrode_threshold
        self.significant_electrodes = significant_electrodes
        self.n_removed_channels = n_removed_channels
        self.current_run_path = current_run_path
        self.reference_run_path = reference_run_path
        self.vision_id_offset = vision_id_offset

    @classmethod
    def from_vision_dirs(
        cls,
        current_dir: str | Path,
        current_dataset: str,
        ref_dir: str | Path,
        ref_dataset: str,
        **kwargs,
    ) -> "CrossRunMatcher":
        """
        Convenience constructor that loads EIs from Vision directories.
        Requires visionloader.
        """
        from .vision_integration import load_ei_data

        current_ei_bundle = load_ei_data(Path(current_dir), current_dataset)
        ref_ei_bundle = load_ei_data(Path(ref_dir), ref_dataset)

        current_eis = current_ei_bundle.get("ei_data") if current_ei_bundle else None
        ref_eis = ref_ei_bundle.get("ei_data") if ref_ei_bundle else None

        # Check for templates.npy as fallback
        curr_templates = Path(current_dir) / "templates.npy"
        ref_templates = Path(ref_dir) / "templates.npy"

        return cls(
            current_eis=current_eis,
            reference_eis=ref_eis,
            current_templates_path=str(curr_templates) if curr_templates.exists() else None,
            reference_templates_path=str(ref_templates) if ref_templates.exists() else None,
            current_run_path=str(current_dir),
            reference_run_path=str(ref_dir),
            **kwargs,
        )

    def run(self) -> MatchingReport:
        """
        Execute hierarchical matching.

        Tier 1: EI matching (if both runs have EIs)
        Tier 2: Template matching for any cells unmatched by Tier 1
                 (or as primary if no EIs)
        """
        report = MatchingReport(
            current_run_path=self.current_run_path,
            reference_run_path=self.reference_run_path,
            method=self.method,
            corr_threshold=self.corr_threshold,
            marginal_threshold=self.marginal_threshold,
        )

        matched_current_ids = set()
        matched_reference_ids = set()

        # ---- Tier 1: EI matching ----
        if self.current_eis and self.reference_eis:
            logger.info("Tier 1: EI-based matching (method=%s)", self.method)
            ei_results = self._match_eis()
            for m in ei_results:
                report.matches.append(m)
                if m.reference_id is not None and m.status in ("high", "marginal"):
                    matched_current_ids.add(m.current_id)
                    matched_reference_ids.add(m.reference_id)
            logger.info(
                "Tier 1 complete: %d high, %d marginal, %d unmatched, %d conflicts",
                len([m for m in ei_results if m.status == "high"]),
                len([m for m in ei_results if m.status == "marginal"]),
                len([m for m in ei_results if m.status == "unmatched"]),
                len([m for m in ei_results if m.status == "conflict"]),
            )

        # ---- Tier 2: Template matching for remaining cells ----
        if (self.current_templates_path and self.reference_templates_path):
            unmatched_ids = [
                m.current_id for m in report.matches
                if m.status in ("unmatched", "conflict")
            ]
            # If no EI matching happened, try templates for all cells
            if not self.current_eis or not self.reference_eis:
                unmatched_ids = None  # match all available

            if unmatched_ids is None or len(unmatched_ids) > 0:
                logger.info("Tier 2: Template-based matching for remaining cells")
                template_results = self._match_templates(
                    restrict_current_ids=unmatched_ids,
                    exclude_reference_ids=matched_reference_ids,
                )
                # Replace unmatched/conflict entries with template matches
                if template_results:
                    template_map = {m.current_id: m for m in template_results}
                    for i, m in enumerate(report.matches):
                        if m.current_id in template_map and m.status in ("unmatched", "conflict"):
                            t = template_map[m.current_id]
                            if t.status in ("high", "marginal"):
                                report.matches[i] = t

                    # Add any entirely new cells (no EI entry existed)
                    existing_ids = {m.current_id for m in report.matches}
                    for t in template_results:
                        if t.current_id not in existing_ids:
                            report.matches.append(t)

        logger.info("Matching complete: %s", report.summary())
        return report

    def _match_eis(self) -> List[MatchResult]:
        """Tier 1: Match using EI correlation."""
        curr_matrix, curr_ids, curr_excluded = _extract_ei_matrix(
            self.current_eis,
            method=self.method,
            electrode_threshold=self.electrode_threshold,
            significant_electrodes=self.significant_electrodes,
            n_removed_channels=self.n_removed_channels,
        )
        ref_matrix, ref_ids, ref_excluded = _extract_ei_matrix(
            self.reference_eis,
            method=self.method,
            electrode_threshold=self.electrode_threshold,
            significant_electrodes=self.significant_electrodes,
            n_removed_channels=self.n_removed_channels,
        )

        if curr_matrix.size == 0 or ref_matrix.size == 0:
            logger.warning("EI matrices empty — cannot match")
            return []

        # Ensure feature dimensions match (trim to shorter if different sample counts)
        min_feat = min(curr_matrix.shape[1], ref_matrix.shape[1])
        curr_matrix = curr_matrix[:, :min_feat]
        ref_matrix = ref_matrix[:, :min_feat]

        corr = _correlation_matrix(curr_matrix, ref_matrix)

        results = _assign_mutual_best(
            corr, curr_ids, ref_ids,
            corr_threshold=self.corr_threshold,
            marginal_threshold=self.marginal_threshold,
            tier="ei",
        )

        # Add excluded cells as unmatched
        for eid in curr_excluded:
            results.append(MatchResult(
                current_id=eid,
                reference_id=None,
                confidence=0.0,
                tier="ei",
                status="unmatched",
            ))

        return results

    def _match_templates(
        self,
        restrict_current_ids: Optional[List[int]] = None,
        exclude_reference_ids: Optional[set] = None,
    ) -> List[MatchResult]:
        """Tier 2: Match using Kilosort templates.npy."""
        try:
            curr_matrix, curr_ks_ids = _extract_template_matrix(
                self.current_templates_path,
                cluster_ids=self.current_cluster_ids,
            )
            ref_matrix, ref_ks_ids = _extract_template_matrix(
                self.reference_templates_path,
                cluster_ids=self.reference_cluster_ids,
            )
        except Exception as e:
            logger.warning("Failed to load templates: %s", e)
            return []

        if curr_matrix.size == 0 or ref_matrix.size == 0:
            return []

        # Convert Kilosort IDs to Vision IDs (cluster_id + offset)
        curr_vision_ids = [cid + self.vision_id_offset for cid in curr_ks_ids]
        ref_vision_ids = [cid + self.vision_id_offset for cid in ref_ks_ids]

        # Filter to requested subsets
        if restrict_current_ids is not None:
            restrict_set = set(restrict_current_ids)
            mask = [vid in restrict_set for vid in curr_vision_ids]
            curr_matrix = curr_matrix[mask]
            curr_vision_ids = [v for v, m in zip(curr_vision_ids, mask) if m]

        if exclude_reference_ids:
            mask = [vid not in exclude_reference_ids for vid in ref_vision_ids]
            ref_matrix = ref_matrix[mask]
            ref_vision_ids = [v for v, m in zip(ref_vision_ids, mask) if m]

        if curr_matrix.size == 0 or ref_matrix.size == 0:
            return []

        # Templates may have different channel counts — align
        min_ch = min(curr_matrix.shape[1], ref_matrix.shape[1])
        curr_matrix = curr_matrix[:, :min_ch]
        ref_matrix = ref_matrix[:, :min_ch]

        corr = _correlation_matrix(curr_matrix, ref_matrix)

        return _assign_mutual_best(
            corr, curr_vision_ids, ref_vision_ids,
            corr_threshold=self.corr_threshold,
            marginal_threshold=self.marginal_threshold,
            tier="template",
        )


# ---------------------------------------------------------------------------
# RF validation (post-hoc, Tier 3)
# ---------------------------------------------------------------------------

def validate_matches_with_rf(
    report: MatchingReport,
    current_params,   # VisionCellDataTable or similar
    reference_params,
    max_distance_um: float = 100.0,
) -> MatchingReport:
    """
    Post-hoc validation: flag matches where RF centers are too far apart.

    Only applicable when both runs have RF params (both ran white noise).
    Doesn't change match assignments — just downgrades status to "marginal"
    for spatially inconsistent matches.
    """
    if current_params is None or reference_params is None:
        logger.info("RF validation skipped: params not available for both runs")
        return report

    for match in report.matches:
        if match.reference_id is None or match.status == "unmatched":
            continue

        try:
            curr_rf = _get_rf_center(current_params, match.current_id)
            ref_rf = _get_rf_center(reference_params, match.reference_id)
        except (KeyError, AttributeError):
            continue

        if curr_rf is None or ref_rf is None:
            continue

        dist = np.sqrt((curr_rf[0] - ref_rf[0]) ** 2 + (curr_rf[1] - ref_rf[1]) ** 2)
        if dist > max_distance_um:
            logger.info(
                "RF mismatch: cell %d→%d distance=%.1f um (threshold=%.1f), "
                "downgrading to marginal",
                match.current_id, match.reference_id, dist, max_distance_um,
            )
            match.status = "marginal"
            match.tier = "rf_flagged"

    return report


def _get_rf_center(params, cell_id):
    """Extract RF center (x, y) from VisionCellDataTable. Returns None if unavailable."""
    try:
        x = params.getDoubleField(cell_id, "x0")
        y = params.getDoubleField(cell_id, "y0")
        if np.isfinite(x) and np.isfinite(y):
            return (x, y)
    except Exception:
        pass
    return None


# ---------------------------------------------------------------------------
# JSON sidecar naming convention
# ---------------------------------------------------------------------------

def get_mapping_path(current_run_path: str | Path, ref_run_path: str | Path) -> Path:
    """
    Deterministic filename for the mapping JSON sidecar.
    Placed alongside the current run's data directory.
    """
    current = Path(current_run_path)
    ref_name = Path(ref_run_path).name
    return current / f"_mapping_to_{ref_name}.json"