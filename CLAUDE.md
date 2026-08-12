# RGCViewer — orientation

RGCViewer (internal name Axolotl) is a PyQt GUI. Use it to curate
spike-sorted multi-electrode array recordings from ex vivo retina and to
sort units into retinal ganglion cell (RGC) types.

## Start

1. Activate conda environment `rgcviewer` (Python 3.10, PyQt5 via qtpy).
2. From the repository root, run `python main.py`.
3. Use File → Open to load a run. The window does not reopen the last run.

Tests live in `tests/`. Documents live in `docs/`. Both are in the
repository. Run tests with:

```bash
conda activate rgcviewer
python -m pytest tests/unit/ -v
```

Read next: `docs/AGENTS.md`, then `HANDOFF.md`.

## The experiment

A piece of mouse retina sits on a Litke 512-electrode array. Pitch is
60 µm. Area is 1890 × 900 µm (~1.7 mm²). Sample rate is 20 kHz.

Visual stimuli come from Symphony (`manookinlab.protocols.*`,
`fieldlab.protocols.*`). Light level changes between recordings with an
NDF wheel. Runs are grouped as NDF3ish, NDF2wheel, and similar labels.

Each recording is a numbered run: `data000`, `data001`, … . Each run is
one protocol:

| Protocol | Result |
|---|---|
| `SpatialNoise` | White-noise checkerboard. STA and receptive-field fits. |
| `ChirpStimulus` | ON/OFF steps plus frequency and contrast sweeps. Response shape. |
| `GratingDSOS_ks` | Drifting gratings. Direction and orientation selectivity. |
| `ContrastResponseGrating` | Contrast series. Contrast gain, F1/F2. |
| `PresentMovies` | Repeated natural movies. |

Every run is spike-sorted independently. Cell IDs in `data006` have no
relation to cell IDs in `data007`.

Some preparations have a concatenated sort folder such as `data010-013`.
Only inside that folder do stimuli share one ID space.
`CrossRunMatcher` (EI correlation) and `ReferenceBridge` map separately
sorted runs. They are a lossy substitute for a concatenated sort.

## Files

Layout: `<prep>/kilosort25/<run>/`. Example: `20260721A/kilosort25/data006/`.

| File | Contents |
|---|---|
| `.neurons` | Spike times per cell. Big-endian header: `[header_size, n_cells, n_samples, sampling_rate]` |
| `.ei` | Electrical image. Mean waveform on all electrodes per cell. Often >500 MB. |
| `.sta` | Spike-triggered average. Present on noise runs. Can be stale. |
| `.params` | Vision parameters, including Gaussian RF fits (`stafit`). |
| `.globals` | Run metadata and electrode map. Can disagree with `.ei` width. |
| `.noise` | Spike-sorting noise estimate. |
| `ksfiles/` | Kilosort 2.5 output: `spike_times.npy`, `spike_clusters.npy`, `cluster_*.tsv`, `params.py`. |
| `*_ChirpStimulus.npy`, `*_GratingDSOS.npy`, `*_contrastResponse_unified.npy` | Stimulus analyses computed offline. The app reads them. It does not create them. |
| `*.classification_MC.txt` | Manual classification: `<visionID>  All/Path/To/Group/` |

Two files sit in `<prep>/kilosort25/stimuli/`:

- `<prep>.json` — authoritative stimulus manifest. Protocol, parameters,
  `epochStarts` / `epochEnds`, `frameTimesMs`, array ID, strain, injection,
  age. The app does not read this file.
- `<prep>.txt` — human summary. Incomplete. Prefer the JSON.

## The analysis

The goal is to partition several hundred sorted units into RGC types.

1. `DataManager.get_cell_physics` assembles per-cell features from Vision data.
2. `analysis_core.build_feature_matrix` builds a weighted matrix: PCA blocks
   for temporal STA, autocorrelogram, grating direction curve, and chirp
   PSTH, plus RF long/short diameter. Weights are sliders.
   `constants.py` holds the defaults.
3. UMAP embeds the matrix. Ward or K-Means clusters the embedding.
4. Clusters become groups in the tree. The user curates them and exports a
   classification file.

The validation criterion is the mosaic. A real RGC type tiles the retina.
Receptive fields cover the array and do not overlap. A cluster whose RFs
sit on top of each other is contaminated.

`RFMapWidget` exists for this check. Lasso a group. See if the RFs tile.

Only about 10–15% of RGCs under the array are sorted. Types look sparse.
Nearest-neighbour regularity and coverage degrade under that subsample.
RF overlap does not: removal of a cell cannot create overlap. Overlap is
the trustworthy mosaic statistic.

## Traps

**Vision ID = Kilosort `cluster_id` + 1**, except in vision-only mode,
where they are equal. Always call `DataManager.get_vision_id_for_cluster`.
An off-by-one error is silent.

**Cells with no STA collapse into a fake cluster.**
`get_raw_feature_blocks` gives a cell with no RF fit an all-zero temporal
row. `build_feature_matrix` guards the all-missing case, not the mixed
case. Those cells share one point in the temporal PCA block. Temporal
default weight is 10, so that point dominates. The tight group is
"cells with no receptive field". The same applies to grating and chirp
zero-sentinels. Not fixed.

**Feature weights are not what they look like.** A block's contribution
to Euclidean distance scales with `n_columns × weight²`. At defaults,
temporal STA (4 PCs × 10) is ~400 against ACG (4 PCs × 1) ~4. The
default embedding is almost STA timecourse alone.

**Stixel size varies between preparations.** A 200 µm stixel against a
100–300 µm mouse RF leaves the Gaussian fit unconstrained. RF diameters
from that run are not comparable to a 40–50 µm run.

**Check the animal before you assume wild-type.** The JSON records strain
and injection. An `rd10` retina with an optogenetic rescue (for example
`Grm6-waChR` in ON bipolar cells) has no photoreceptor-driven vision.
The ON/OFF axis partly collapses. rd10 retina also shows a strong
~5–15 Hz network oscillation that dominates the ACG as a network
property, not a cell-type signature.

**Array IDs 500–1500 are the Litke 512**, resolved by
`electrode_map.determine_array_type`. Recorded `arrayPitch` of 60 µm is
correct.

**A stale `.sta` attaches RFs to the wrong units.** The load dialog
reports this. Example: 20251204, "159 of 310 cells in the .sta do not
exist in this sort". Use the noise-run STA or Map Reference. Spike
analyses are still valid.

**Some older kilosort4 conversions have a broken EI.** The `.ei` payload
can be 519 channels while `.globals` is a 512-row map. The loader reads
the payload width from the `.ei` file. Plots that still disagree stay
blank. Do not invent an electrode map to force a draw. The user accepts
missing EI plots on those runs.

## Code layout

- `src/analysis/data_manager.py` — loading, caching, per-cell feature source
- `src/analysis/analysis_core.py` — STA metrics, EI computation, feature matrix
- `src/analysis/cross_run_matcher.py`, `reference_bridge.py` — bridging runs
- `src/analysis/visionloader.py` — Vision file readers; EI stride from `.ei`
- `src/gui/main_window.py` — shell, tree, table, context menus
- `src/gui/callbacks.py` — tree manipulation, loading, saving
- `src/gui/panels/` — one module per analysis tab
- `src/gui/panels/live_selectors.py` — rectangle and lasso tools
- `src/gui/recent_paths.py` — file-dialog folder memory (QSettings)
