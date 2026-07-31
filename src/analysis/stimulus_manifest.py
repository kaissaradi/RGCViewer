"""Reader for the per-preparation stimulus manifest, ``<prep>/kilosort25/stimuli/<prep>.json``.

Symphony writes one JSON per preparation describing every run: its protocol,
full parameter set, epoch boundaries, array id and the experimenter's epoch-group
label. Nothing in the app read it before this module — the sorted files alone
cannot say which light level a run was recorded at.

**Light level is not recorded numerically.** ``properties.ndf`` is null for
every block in every preparation checked, and ``backgroundIntensity`` is 0.5
throughout, so the stimulus itself does not encode it either. The only record is
the free-text epoch-group label the experimenter typed — "NDF5wheel", "NDF3ish",
"ndf0", "NDF -1.3", "ndf1-again". This module therefore reports the label
verbatim and makes no attempt to parse a number out of it; ordering light levels
is a decision for the user, not a regex.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

__all__ = ["RunInfo", "StimulusManifest", "find_manifest"]


class RunInfo:
    """What the manifest knows about one run."""

    __slots__ = ("run", "protocol", "group_label", "parameters", "n_samples",
                 "array_id", "epoch_starts", "epoch_ends")

    def __init__(self, run, protocol, group_label, parameters, n_samples,
                 array_id, epoch_starts, epoch_ends):
        self.run = run
        self.protocol = protocol
        self.group_label = group_label
        self.parameters = parameters or {}
        self.n_samples = n_samples
        self.array_id = array_id
        self.epoch_starts = epoch_starts or []
        self.epoch_ends = epoch_ends or []

    @property
    def protocol_short(self) -> str:
        return (self.protocol or "").rsplit(".", 1)[-1]

    @property
    def n_epochs(self) -> int:
        return len(self.epoch_starts)

    def __repr__(self):
        return (f"RunInfo({self.run}, {self.protocol_short}, "
                f"group={self.group_label!r}, epochs={self.n_epochs})")


def find_manifest(kilosort_dir) -> Path | None:
    """Locate ``<prep>.json`` given a run directory.

    Layout is ``<prep>/kilosort25/<run>/``, so the manifest sits beside the run
    folders in ``<prep>/kilosort25/stimuli/``. Also checks the run directory
    itself and one level further up, since concatenated-sort folders are
    sometimes nested differently.
    """
    if kilosort_dir is None:
        return None
    d = Path(kilosort_dir)
    for base in (d.parent / "stimuli", d / "stimuli", d.parent.parent / "stimuli"):
        try:
            if not base.is_dir():
                continue
        except OSError:
            continue
        candidates = sorted(base.glob("*.json"))
        if candidates:
            return candidates[0]
    return None


class StimulusManifest:
    """Parsed ``<prep>.json``, indexed by run name.

    Never raises on malformed input: a manifest that cannot be read yields an
    empty instance, and callers degrade to "light level unknown" rather than
    failing to load a dataset.
    """

    def __init__(self, path=None):
        self.path = Path(path) if path else None
        self.experiment = None
        self.runs: dict[str, RunInfo] = {}
        self.animal: dict = {}
        if self.path is not None:
            self._load()

    @classmethod
    def for_kilosort_dir(cls, kilosort_dir) -> "StimulusManifest":
        return cls(find_manifest(kilosort_dir))

    def _load(self):
        try:
            with open(self.path, "r", encoding="utf-8") as fh:
                doc = json.load(fh)
        except Exception as e:
            logger.warning("Could not read stimulus manifest %s: %s", self.path, e)
            return

        self.experiment = self.path.stem
        try:
            self._read_animal(doc)
        except Exception:
            logger.debug("Manifest animal metadata unreadable", exc_info=True)

        for proto in doc.get("protocol", []) or []:
            label = proto.get("label")
            for group in proto.get("group", []) or []:
                group_label = group.get("label")
                for blk in group.get("block", []) or []:
                    # dataFile looks like "20260715A/data000/"
                    raw = (blk.get("dataFile") or "").strip("/")
                    run = raw.rsplit("/", 1)[-1] if raw else None
                    if not run:
                        continue
                    props = blk.get("properties") or {}
                    self.runs[run] = RunInfo(
                        run=run,
                        protocol=label,
                        group_label=group_label,
                        parameters=blk.get("parameters") or {},
                        n_samples=blk.get("n_samples") or props.get("n_samples"),
                        array_id=blk.get("array_id") or props.get("array_id"),
                        epoch_starts=blk.get("epochStarts"),
                        epoch_ends=blk.get("epochEnds"),
                    )
        logger.debug("Stimulus manifest %s: %d runs", self.path.name, len(self.runs))

    def _read_animal(self, doc):
        node = (doc.get("sources") or [{}])[0]
        self.animal = {
            "label": node.get("label"),
            "keywords": (node.get("attributes") or {}).get("keywords"),
            "properties": node.get("properties") or {},
        }

    # ── lookups ──────────────────────────────────────────────────────────────

    def get(self, run) -> RunInfo | None:
        return self.runs.get(str(run))

    def group_label(self, run, default=None):
        """The experimenter's epoch-group label for *run* — the only record of
        light level. Free text; do not parse it."""
        info = self.runs.get(str(run))
        return info.group_label if info is not None else default

    def group_labels(self, runs, default="unknown") -> list:
        return [self.group_label(r, default) or default for r in runs]

    def runs_for_protocol(self, needle) -> list:
        needle = needle.lower()
        return sorted(r for r, i in self.runs.items()
                      if needle in (i.protocol or "").lower())

    def __len__(self):
        return len(self.runs)

    def __bool__(self):
        return bool(self.runs)
