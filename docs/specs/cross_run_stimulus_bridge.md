# SPEC.md — Cross-Run Stimulus Bridge (map any run → physics cache, RF mosaic, UMAP)

> Reading order: **AGENTS.md (full) → PLAN.md (Fragile Zones) → this spec → write failing tests → implement.**
> Do **not** start implementation until this spec is marked **Ready for Dev** and the open design decisions in Block 11 are resolved.

---

## Progress Log (resume point)

> Update after every stage so a fresh session can pick up here.

- [x] **Stage 0 — Spec + context.** This document + PLAN.md Active Work entry.
- [ ] **Stage 1 — ReferenceBridge stimulus load + per-cell caveats.** Extend bridge to load chirp/grating when present; expose remapped accessors and per-current-id caveat records.
- [ ] **Stage 2 — Physics / feature pipeline integration.** `get_cell_physics` + `get_raw_feature_blocks` fill-gaps from bridge; provenance fields; invalidation on map install.
- [ ] **Stage 3 — RF mosaic + GUI wiring.** Population panel draws borrowed ellipses; `map_reference_run` loads stimuli, invalidates caches, re-gates UMAP.
- [ ] **Stage 4 — Tests + docs.** Unit tests for ID remap, fill-gap precedence, caveats; PLAN.md Completed Fix Registry.

---

## Block 0 — Metadata

| Field | Value |
|---|---|
| **Date created** | 2026-07-16 |
| **Last updated** | 2026-07-16 |
| **Commit hash when spec was written** | `7fb457a` |
| **Branch** | `feat/cross-run-stimulus-bridge` (create from current work branch before coding) |
| **Author** | Kais + agent |
| **Spec status** | Draft — awaiting design decisions (Block 11) |

---

## Block 1 — Problem Statement

**Symptom:** A scientist loads a run that has EIs (and maybe spikes) but is missing one or more stimulus products — white-noise STAs/RFs, chirp PSTHs, and/or grating DSOS curves. Another run of the *same retina* has those products. Today "Map Reference Run..." can EI-match cells and load reference STAs into a `ReferenceBridge`, but:

1. Chirp and grating from the reference run are **never** loaded or remapped.
2. Borrowed STA/RF geometry is **not** reliably reflected in the population RF mosaic (panel only reads `dm.vision_params`).
3. UMAP feature blocks (`get_raw_feature_blocks`) only see **current-run** chirp/grating; borrowed white-noise features only appear if `get_cell_physics` is re-run *after* the bridge is installed — and a warm `feature_cache` with `_computed: True` **never re-borrows**.
4. Match quality (high / marginal / confidence / next-best) is not stored as a **per-current-cell caveat** that UMAP, panels, or export can read.

**Root cause (gaps in the existing partial implementation):**

| Gap | Where |
|---|---|
| Bridge only loads STAs + params | `ReferenceBridge.from_matching_report(load_stas=True, load_params=True)` — no chirp/grating |
| Timecourse borrow may use wrong ID for params lookup | `get_cell_physics()` passes current `vid` into `get_sta_timecourse_data(..., ref_params, vid)` instead of the **reference** Vision ID |
| Warm physics cache never re-evaluates after map | `get_cell_physics` returns any `_computed` entry immediately |
| Feature matrix ignores bridge for chirp/grating | `get_raw_feature_blocks` only reads `chirp_id_to_row` / `grating_computed_cache` on the current DM |
| RF mosaic ignores bridge | `plot_population_rfs_background` iterates `vision_params.get_cell_ids()` only |
| Caveats not first-class | Bridge has `_confidence` / `_statuses` but nothing writes them into `feature_cache` or a stable per-cell caveat map for the original run's IDs |

**User story:**  
"As a scientist, after I map another run of the same retina, I want every stimulus product available on that run (white noise STAs/RFs, chirp, grating) remapped onto **my current cell IDs**, filled into the physics/feature pipeline so UMAP and the RF mosaic use them, with an explicit **caveat per current cell** about match quality and data provenance — so I never mistake borrowed responses for native ones."

---

## Block 2 — Vision ID Contract

| Question | Answer |
|---|---|
| Does this spec access Vision file data? | **Yes** — reference STAs/params/EIs via existing loaders; current EIs for matching |
| ID space this spec operates in | **Both** — matching report keys are **Vision 1-indexed** IDs (EI dict keys). Chirp/grating on disk are often Vision-keyed and re-keyed to **Kilosort 0-indexed** at load (same as `load_chirp_data` / `load_grating_data`). UI / `feature_cache` / UMAP always use **cluster_id** (KS space in hybrid, Vision space when `is_vision_only`). |
| Reads `is_vision_only` flag? | **Yes** |
| Translation used | Canonical: `vid = dm.get_vision_id_for_cluster(cluster_id)` (or `cluster_id if is_vision_only else cluster_id + 1`). Bridge public API for panels/physics accepts **cluster_id** (UI space) and translates internally. MatchingReport continues to store **Vision IDs** (do not change JSON schema keys without a migration note). |
| Safe access pattern used | Never `vision_stas[cluster_id]`. Physics path: `get_cell_physics(cluster_id)` owns offset. Bridge STA access: map `current_vid → ref_vid`, then `ref_stas[ref_vid]`. Chirp/grating bridge maps: current `cluster_id → ref cluster_id → row/entry`. |

### ID space diagram (hybrid mode)

```
Current run UI / feature_cache / UMAP:   cluster_id  (0-indexed KS)
Current Vision files / MatchingReport:   current_vid = cluster_id + 1
Reference Vision files:                  reference_vid
Reference chirp/grating after load:      ref_cluster_id = reference_vid - 1  (unless ref is vision-only)

Bridge mapping (MatchingReport.mapping): { current_vid: reference_vid }
Caveats keyed for the scientist:         { cluster_id: CaveatRecord }   # original-run UI IDs
```

**Law 1 regressions to watch:**  
- `get_cell_physics` reference timecourse must call `get_sta_timecourse_data(..., ref_id)` not current `vid`.  
- Chirp row maps from a reference file must apply the **reference** run's vision-only flag (or a documented convention: always treat analysis npy IDs as Vision-keyed unless marked otherwise — same as current loaders).

---

## Block 3 — Affected Files

| File path | Function(s) added or modified | Change type | Touches DataManager? |
|---|---|---|---|
| `src/analysis/reference_bridge.py` | `ReferenceBridge`, `from_matching_report`, new stimulus loaders + caveat API | Extend | No |
| `src/analysis/cross_run_matcher.py` | (optional) document ID space; maybe attach stimulus inventory to report metadata | Light / docs-only unless needed | No |
| `src/analysis/data_manager.py` | `get_cell_physics`, `get_raw_feature_blocks`, new `install_reference_bridge` / `invalidate_physics_for_borrow`, caveat accessors | Modify + add | **Yes** |
| `src/gui/callbacks.py` | `map_reference_run` | Extend post-accept path | Indirect |
| `src/gui/panels/population_panel.py` | `plot_population_rfs_background`, `_update_highlight_patch` | Borrowed ellipse layer | No |
| `src/gui/panels/umap_panel.py` | `refresh_feature_availability` | Gate chirp (and grating if gated) when bridge supplies them | No |
| `tests/unit/test_reference_bridge.py` | New tests (Block 9) | Add | No |
| `tests/unit/test_cross_run_stimulus_bridge.py` | Physics + feature-block fill-gap tests | Add | No |
| `docs/PLAN.md` | Active Work + later Completed Fix Registry | Update | No |
| `docs/AGENTS.md` | Short pointer under architecture / caches if caveats become a first-class cache field | Optional light update | No |

> **DataManager is touched.** Rebase from main before every push. Run Vision ID offset tests after any change to `get_cell_physics`.

### Explicit non-touch (unless a later stage expands scope)

- `cross_run_matcher` mutual-best algorithm thresholds (keep defaults; no retuning in this PR).
- Chirp panel / grating panel single-cell **display** of borrowed curves (optional Stage 5 — out of scope for MVP; physics + UMAP first).
- Auto-discovery of "sibling" runs under a parent date folder (user still picks the reference directory).

---

## Block 4 — Qt Threading Contract

| Operation | Runs on | Worker / mechanism | Signal | Slot | Tier |
|---|---|---|---|---|---|
| Directory pick + confirm dialogs | Main | N/A | N/A | `map_reference_run` | User action (blocking dialogs OK) |
| EI matching (existing) | Main today | Keep as-is for MVP; optional future worker | N/A | N/A | User action — may freeze briefly on large EI sets (pre-existing) |
| Load reference STAs/params/chirp/grating after Accept | Main for MVP *or* background `threading.Thread` if load > ~1s | Prefer: short `QThread` worker `ReferenceStimulusLoadWorker` if chirp/grating npy + STA open is heavy | `bridge_ready = Signal(object)` / `error = Signal(str)` | `_on_reference_bridge_ready` | User action — **not** inside Tier 1 scroll path |
| Physics recompute after install | Background | Reuse `ensure_physics_cache` via existing pattern (daemon pool / UMAP path) | None required if only cache fill | N/A | Not on scroll Tier 1 |
| Population RF redraw | Main | Direct call after bridge installed | N/A | N/A | After map — not per-keypress |
| UMAP checkbox re-gate | Main | `umap_panel.refresh_feature_availability()` | N/A | N/A | After map |

**Hard rule:** Installing a bridge must **not** add disk I/O or bridge lookups to Tier 1 of `update_cluster_views()`.

**Stale result guard:** Any new result slot for a load worker must discard if the user has already mapped a *different* reference or cleared the bridge:

```python
if getattr(self.data_manager, "reference_bridge", None) is not bridge_instance_expected:
    return
```

---

## Block 5 — Cache Contract

| Question | Answer |
|---|---|
| Which cache(s) does this spec read? | `feature_cache` (cluster_id), `grating_computed_cache`, `chirp_data` / `chirp_id_to_row`, bridge-internal ref STAs/params/chirp/grating |
| Which cache(s) does this spec write? | `feature_cache` (physics + provenance + optional caveat mirror); possibly a new `dm.match_caveats: dict[cluster_id, CaveatRecord]` (authoritative for match metadata — not necessarily inside every physics entry) |
| What triggers invalidation? | **On successful Accept of a map:** for every `cluster_id` that gains a match **or** already had `_computed` with missing timecourse/rf while bridge has STA, clear `_computed` (or delete entry) so `get_cell_physics` re-runs with fill-gap. On bridge clear/replace: same. Do **not** wipe ACG values unnecessarily — prefer targeted recompute of STA-derived fields. |
| Is data persisted to disk? | Mapping JSON already: `_mapping_to_{ref_name}.json`. **MVP:** do **not** persist borrowed chirp/grating into `feature_cache.pkl` as if native without provenance (if feature_cache is saved, provenance keys must round-trip). Prefer: on load of pkl, if provenance missing and bridge not installed, treat as current-only. |
| Which lock must be held? | `_feature_lock` when mutating `feature_cache`; `_grating_cache_lock` if writing remapped grating entries into a borrow cache |
| Must tests bypass the cache? | **Yes** for physics recompute tests — use `tmp_path` / empty `feature_cache` |

### Precedence (fill-gap, never clobber native)

For each product:

1. **Current run has usable product** → use current; provenance = `"current"`.
2. **Else** if bridge has match + reference product → use remapped reference; provenance = `"reference"`.
3. **Else** → missing / zero sentinel (existing UMAP behavior); provenance = `None`.

"Usable" means:

- STA/RF: `vision_stas` has vid **or** valid stafit geometry (existing path).
- Chirp: row in `chirp_id_to_row` with QI ≥ `CHIRP_MIN_QI` (same gate as today).
- Grating: entry in analyzed data or `grating_computed_cache` with a non-None pooled curve.

Marginal matches: still fill-gap (include in UMAP) but caveat `status="marginal"` so UI can warn. Conflicts/unmatched: no borrow.

---

## Block 6 — DataManager Attributes Used

| Attribute | Type | Can be `None`? | This spec | Safe access |
|---|---|---|---|---|
| `reference_bridge` | `ReferenceBridge` | Yes | Read + write (install) | `if self.reference_bridge:` |
| `vision_stas` / `vision_params` / `vision_eis` | various | Yes | Read (current preferred) | Existing null checks + Law 1 |
| `feature_cache` | `dict` | No (may be empty) | Read + write | `_feature_lock` |
| `chirp_data`, `chirp_id_to_row`, `chirp_available` | … | Yes / bool | Read; availability OR with bridge | Existing |
| `grating_data`, `grating_computed_cache`, `grating_available`, `grating_status` | … | Yes | Read; fill-gap from bridge | Locks for computed cache |
| `is_vision_only` | `bool` | No | Read | Direct |
| `match_caveats` (**new**) | `dict[int, CaveatRecord]` | Empty dict default | Write on install; read by panels/export | Main-thread install; read-only from workers after install |

### New types (live in `reference_bridge.py` or small dataclass module)

```python
@dataclass
class CellMatchCaveat:
    """Per original-run cluster_id (UI space)."""
    cluster_id: int                 # current run UI ID
    current_vision_id: int          # matching report key
    reference_id: Optional[int]     # Vision ID on reference run
    status: str                     # "high" | "marginal" | "unmatched" | "conflict" | ""
    confidence: float               # EI/template correlation
    next_best_corr: float = 0.0
    next_best_id: Optional[int] = None
    tier: str = "ei"                # "ei" | "template" | "rf_flagged" | ...
    provenance: Dict[str, Optional[str]] = field(default_factory=dict)
    # provenance keys: "timecourse", "rf_geometry", "chirp", "grating"
    # values: "current" | "reference" | None
    reference_run_path: str = ""
    stimuli_available_on_ref: Tuple[str, ...] = ()  # e.g. ("sta", "chirp")
```

Authoritative map: `DataManager.match_caveats[cluster_id] = CellMatchCaveat(...)`.  
Also keep `ReferenceBridge.get_caveat(cluster_id)` for convenience that translates Vision→UI if needed.

---

## Block 7 — Acceptance Criteria

### AC1 — Map any reference Vision directory

- **Setup:** Current run has EIs loaded. Reference directory has at least a `.ei` file.
- **Action:** File → Map Reference Run… → select dir → Accept.
- **Expected:** MatchingReport produced or loaded from sidecar; `dm.reference_bridge` non-None; status bar shows match summary; mapping JSON written if new.
- **Test type:** Unit (matcher + bridge construction with synthetic EI dicts) + Manual GUI

### AC2 — Stimulus inventory on reference

- **Setup:** Reference dir contains some subset of: `.sta`+`.params`, `*Chirp*.npy`, `*Grating*.npy` / `*DSOS*.npy`.
- **Action:** Accept map.
- **Expected:** Bridge reports which products loaded (e.g. `stimuli=("sta","chirp")`). Missing products do not fail the whole map. Current-run-only products remain current-only.
- **Test type:** Unit with `tmp_path` fake files / mocked loaders

### AC3 — Fill-gap into physics cache (white noise)

- **Setup:** Current cell has no STA; bridge has high-confidence match with STA+params on reference. `feature_cache` empty for that cell.
- **Action:** `dm.get_cell_physics(cluster_id)`.
- **Expected:** Non-None `timecourse` and non-zero RF geometry when reference has them; `provenance["timecourse"]=="reference"`; `match_caveats[cluster_id].status=="high"`. Params timecourse lookup uses **reference_id**, not current vid.
- **Test type:** Unit

### AC4 — Native preferred over reference

- **Setup:** Current cell has own STA; reference also has STA for match.
- **Action:** `get_cell_physics`.
- **Expected:** Timecourse/geometry from current; provenance `"current"`. Reference not used for those fields.
- **Test type:** Unit

### AC5 — Warm cache recompute after map

- **Setup:** Cell physics computed before map with `timecourse=None`, `_computed=True`. Then install bridge that can supply STA.
- **Action:** `install_reference_bridge` (or map Accept path).
- **Expected:** That cell's cache entry is invalidated for STA-derived fields; subsequent `get_cell_physics` returns borrowed timecourse. ACG not needlessly wiped if already present.
- **Test type:** Unit

### AC6 — Chirp / grating remapped for UMAP

- **Setup:** Current run `chirp_available=False`; reference has chirp npy covering matched ref IDs. Grating analogous.
- **Action:** `get_raw_feature_blocks(ids, filter_config)` with chirp/grating weights enabled in UMAP config path.
- **Expected:** Non-zero chirp/grating rows for matched cells that have QI/curve on reference; unmatched cells still zero sentinels; `chirp_available` effective for UMAP gate becomes True if either current **or** bridge has chirp.
- **Test type:** Unit

### AC7 — Caveats for every current cell in the report

- **Setup:** MatchingReport with high, marginal, unmatched, conflict entries.
- **Action:** Install bridge.
- **Expected:** `match_caveats` has an entry for every current-run cell that appeared in the report (including unmatched), keyed by **UI cluster_id**. High/marginal include `reference_id` + confidence; unmatched have `reference_id is None`.
- **Test type:** Unit

### AC8 — RF mosaic shows borrowed ellipses

- **Setup:** Current `vision_params` missing fits for matched cells; bridge has RF params.
- **Action:** Redraw population RF mosaic after map.
- **Expected:** Borrowed ellipses appear for those current cells (UI IDs in labels). Styling distinguishes borrowed from native (e.g. dashed edge or lower alpha — exact style in Block 11). No crash when only borrowed RFs exist.
- **Test type:** Unit (ellipse list builder) + Manual visual

### AC9 — UMAP checkbox re-gate

- **Setup:** Chirp checkbox disabled after load (no current chirp). Map supplies chirp via bridge.
- **Action:** Accept map → `refresh_feature_availability`.
- **Expected:** Chirp feature row enabled; tooltip clear. Clearing bridge (if implemented) re-disables when no current chirp.
- **Test type:** Unit/integration light + Manual

### AC10 — No Tier 1 regression

- **Setup:** Bridge installed.
- **Action:** Rapid cluster scroll.
- **Expected:** No disk I/O or full bridge STA loads in Tier 1. STA panel may still load on Tier 2 as today.
- **Test type:** Manual / code review checklist

### AC11 — Law 1 hybrid vs vision-only

- **Setup:** Parametrize hybrid and vision-only DataManagers with synthetic bridge mapping.
- **Action:** `get_cell_physics` / chirp fill-gap for a known pair.
- **Expected:** Correct cell's reference product; no off-by-one silent swap.
- **Test type:** Unit (parametrize)

---

## Block 8 — Regression Guard

| Prior fix | Files overlap | Regression test to run | When |
|---|---|---|---|
| Vision ID offset in `get_cell_physics` | `data_manager.py` | `test_get_cell_physics_vision_id_offset` (both branches) | After Stage 2, before PR |
| Physics cache unified / LazySTADict | `data_manager.py`, bridge STA loads | `test_lazy_sta_dict_*`, `test_ensure_physics_cache_*` | Before PR |
| Chirp UMAP feature block | `get_raw_feature_blocks` | `TestGetRawFeatureBlocksChirp`, `TestBuildFeatureMatrixChirp` | After Stage 2 |
| Chirp checkbox re-gate | `umap_panel.refresh_feature_availability` | Existing chirp checkbox tests if any; manual | After Stage 3 |
| Population RF mosaic | `population_panel.py` | `test_population_rf_*` | After Stage 3 |
| ACG full recording | `get_cell_physics` ACG path | `test_acg_includes_late_spike_trains` | If ACG invalidation touched |

---

## Block 9 — Test Plan

### Unit — `tests/unit/test_reference_bridge.py`

| Test function | Fixture | Asserts | Cache bypass |
|---|---|---|---|
| `test_from_report_mapping_keys_vision_ids` | synthetic report | mapping and reverse map correct | N/A |
| `test_get_sta_uses_reference_id` | mock LazySTADict | `get_sta(current)` returns ref entry only | N/A |
| `test_load_chirp_remapped_to_current_cluster_ids` | tmp npy + report | bridge `has_chirp(cid)` / `get_chirp_psth(cid)` | N/A |
| `test_load_grating_remapped` | mock analyzed dict | remapped by mapping | N/A |
| `test_caveats_for_all_report_cells` | report with all statuses | len(caveats) covers high+marginal+unmatched+conflict | N/A |
| `test_missing_stimulus_does_not_fail_bridge` | ref without chirp | bridge still has STAs | N/A |

### Unit — `tests/unit/test_cross_run_stimulus_bridge.py`

| Test function | Fixture | Asserts | Cache bypass |
|---|---|---|---|
| `test_physics_borrows_timecourse_with_ref_params_id` | mock_dm + bridge | timecourse non-None; spy that params lookup used ref_id | Yes |
| `test_physics_prefers_current_sta` | mock_dm both present | provenance current | Yes |
| `test_install_bridge_invalidates_computed_without_sta` | warm poisoned cache | after install, recompute fills | Yes |
| `test_raw_blocks_chirp_from_bridge` | mock_dm no local chirp | chirp row non-zero for matched | Yes |
| `test_raw_blocks_grating_from_bridge` | mock_dm | grating row non-zero | Yes |
| `test_unmatched_cell_zero_sentinel` | bridge without that id | zeros + caveat unmatched | Yes |
| `test_vision_id_offset_hybrid_and_vision_only` | parametrize | correct product | Yes |
| `test_population_ellipse_list_includes_borrowed` | pure function or light mock | borrowed ellipses in list | N/A |

### Integration (optional MVP)

| Test | Fixture | Exercises |
|---|---|---|
| `test_map_reference_accept_installs_bridge` | make_main_window + qtbot + mocks | Accept path sets `dm.reference_bridge` |

---

## Block 10 — Out of Scope

- Retuning EI correlation thresholds or mutual-best logic.
- Automatic folder crawling for "all stimulus runs under date X".
- Merging **spike trains** across runs (identity only for analysis products).
- Editing/correcting matches in a GUI table (JSON + re-run is enough for MVP).
- Single-cell ChirpPanel / GratingPanel rendering of borrowed curves (follow-up; physics+UMAP+mosaic first).
- STA panel visual "borrowed" badge (nice-to-have; caveats API enables it later).
- Persisting a second full `feature_cache` of only borrowed features.
- Multi-reference simultaneous bridges (MVP: one `reference_bridge` at a time; installing a new one replaces the old).

---

## Block 11 — Open Design Decisions (resolve before Ready for Dev)

These need an explicit yes/no from the owner before coding Stage 1.

### D1 — Borrowed RF mosaic styling

| Option | Pros | Cons |
|---|---|---|
| **A.** Dashed edge, same color family as native | Clear "not native" | Slightly busier |
| **B.** Same solid style as native, lower alpha | Minimal code | Easy to miss that RF is borrowed |
| **C.** Distinct hue (e.g. teal for borrowed) | Very obvious | Theme interaction |

**Recommendation:** **A** (dashed + slightly lower alpha). Labels still show **current** UI IDs.

### D2 — Where authoritative caveats live

| Option | Pros | Cons |
|---|---|---|
| **A.** `dm.match_caveats[cluster_id]` only | Clean separation from physics | Two lookups |
| **B.** Mirror into every `feature_cache` entry | One dict for UMAP export | Pollution; invalidation complexity |
| **C.** Both: authoritative on DM, slim copy on physics entry (`match_status`, `match_confidence`, `provenance`) | Best for UMAP hover + QC | Slight duplication |

**Recommendation:** **C**.

### D3 — Chirp/grating load path on reference

Reference dirs may be pure Vision (`*.sta`, `*.ei`) while chirp/grating npy often sit in the **Kilosort analysis directory**.  

| Option | Pros | Cons |
|---|---|---|
| **A.** Glob only inside the selected reference directory | Simple | Misses files one level up / sibling |
| **B.** Glob selected dir + parent + common sibling patterns (`*Chirp*.npy`) | More hits | Surprise loads |
| **C.** A — plus optional second file dialog "Load chirp/grating from…" if not found | Explicit | Extra click |

**Recommendation:** **A** for MVP; log a clear status message if EI matched but no chirp/grating found ("No *Chirp*.npy in reference dir — map STA only"). Expand to C if lab layout requires it.

### D4 — EI matching thread

Keep matching on main thread (current) vs always background worker.

**Recommendation:** Keep main-thread matching for MVP (existing behavior); only background the **stimulus load** if profiling shows STA/chirp open is slow.

### D5 — Effective `chirp_available` for UMAP

| Option | Pros |
|---|---|
| **A.** `dm.chirp_available or bridge.has_any_chirp()` | Checkboxes enable when only bridge has chirp |
| **B.** Leave `chirp_available` false; special-case UMAP only | Confusing dual flags |

**Recommendation:** **A** — introduce `dm.effective_chirp_available()` (and grating analogue) used by `refresh_feature_availability` and workers. Do not flip the raw `chirp_available` flag (that remains "current run file loaded").

---

## Block 12 — Implementation sketch (not code yet)

### Stage 1 — Bridge

1. Extend `ReferenceBridge.from_matching_report` to accept `load_chirp=True`, `load_grating=True`.
2. On construction, glob reference dir for stimulus products; load via the same schema logic as `DataManager.load_chirp_data` / `load_grating_data` but **do not** write into current DM caches directly — store on bridge, remapped:
   - Internal ref structures keyed by **reference** IDs.
   - Public getters take **current cluster_id**, translate via `get_vision_id` helper passed in **or** dual maps built at install time from report + `is_vision_only`.
3. Build `list[CellMatchCaveat]` / dict for all report rows (UI keys).
4. Fix any getter that still confuses current vs ref ID.

### Stage 2 — DataManager

1. `install_reference_bridge(bridge, report)`:
   - set `reference_bridge`
   - build `match_caveats`
   - invalidate physics for cells that need re-borrow
2. `get_cell_physics`: after current STA path fails, borrow with **ref_id** for params; set provenance; update caveat provenance fields.
3. `get_raw_feature_blocks`: if current chirp/grating missing for cid, ask bridge; same QI / pooled-curve rules.
4. `effective_chirp_available()` / `effective_grating_available()`.

### Stage 3 — GUI

1. `map_reference_run` after Accept: full install path; inventory message; population redraw; `umap_panel.refresh_feature_availability()`.
2. Population: when native stafit missing, use `bridge.get_rf_ellipse_params(cluster_id)` (API should accept UI id); draw dashed.
3. Optional status: "Mapped N cells (H high, M marginal); borrowed STA=… chirp=… grating=…"

### Stage 4 — Tests + PLAN

Per Block 9; then move Active Work → Completed Fix Registry.

---

## Block 13 — Known bugs this spec should fix along the way

1. **Wrong params cell_id on borrow** — `get_cell_physics` passes current `vid` into `get_sta_timecourse_data` with `_ref_params` (looks up wrong row or fails silently into cube fallback).
2. **RF mosaic never uses bridge** — despite `ReferenceBridge.get_all_rf_ellipses()` existing.
3. **Post-map physics sticky** — poisoned/warm `_computed` entries without STA never re-borrow.

---

## Block 14 — Manual verification script (lab data)

When `/mnt/lab/Array-data/` is mounted:

1. Load a run that is EI-rich but STA-poor (or chirp-less).
2. Map a white-noise (or multi-stimulus) sibling run of the same date/retina.
3. Confirm: match dialog numbers look sane; RF mosaic fills in; UMAP chirp checkbox enables if ref has chirp; re-run UMAP; hover/caveat export (if exposed) shows marginal flags.
4. Spot-check 3 high + 3 marginal cells: EI overlay / RF position reasonableness.
5. Dark + light mode mosaic styling.

---

## Block 15 — Definition of Ready / Done

**Ready for Dev:** Block 11 decisions recorded (owner initials + date).  
**Done:** All AC1–AC11 pass or waived in writing; Stages 1–4 checked; PLAN.md updated; no new failures in regression guard table.
