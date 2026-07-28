# RGCViewer — orientation for agents

RGCViewer (internally "Axolotl") is a PyQt GUI for curating spike-sorted
multi-electrode array recordings from ex vivo retina and sorting the resulting
units into retinal ganglion cell (RGC) types.

Run it with `python main.py` from the `rgcviewer` conda environment
(Python 3.10, PyQt5 via qtpy). There is no test suite in the repo; `docs/` and
`tests/` are gitignored.

## The experiment

A piece of mouse retina is laid on a **Litke 512-electrode array** — 60 µm
pitch, 1890 × 900 µm, ~1.7 mm², sampled at 20 kHz. Visual stimuli are
projected onto it and driven by Symphony (`manookinlab.protocols.*`,
`fieldlab.protocols.*`). Light level is stepped between recordings with an NDF
wheel, which is why runs are grouped as "NDF3ish", "NDF2wheel" and so on.

Each recording is a numbered run — `data000`, `data001`, … — and each run is a
separate protocol:

| Protocol | What it gives you |
|---|---|
| `SpatialNoise` | White-noise checkerboard → STA, receptive field fits |
| `ChirpStimulus` | ON/OFF steps + frequency and contrast sweeps → response shape |
| `GratingDSOS_ks` | Drifting gratings → direction/orientation selectivity |
| `ContrastResponseGrating` | Contrast series → contrast gain, F1/F2 |
| `PresentMovies` | Repeated natural movies |

**Every run is spike-sorted independently.** This is the single most important
structural fact about the data: cell IDs from `data006` have nothing to do with
cell IDs from `data007`. Some preparations also have a *concatenated* sort — a
folder like `data010-013` or `data007-010` where several runs were sorted
together — and only within one of those do all stimuli share a single ID space.
`CrossRunMatcher` (EI correlation) and `ReferenceBridge` exist to bridge
separately sorted runs; they are a lossy substitute for a concatenated sort.

## Files

Layout is `<prep>/kilosort25/<run>/`, e.g. `20260721A/kilosort25/data006/`.

| File | Contents |
|---|---|
| `.neurons` | Spike times per cell. Big-endian header: `[header_size, n_cells, n_samples, sampling_rate]` |
| `.ei` | Electrical image — mean waveform across all 512 electrodes per cell. The largest file, often >500 MB |
| `.sta` | Spike-triggered average. Only present for noise runs |
| `.params` | Vision parameters, including the Gaussian RF fits (`stafit`) the mosaics are drawn from |
| `.globals` | Run metadata |
| `.noise` | Spike-sorting noise estimate |
| `ksfiles/` | Raw Kilosort 2.5 output: `spike_times.npy`, `spike_clusters.npy`, `cluster_*.tsv`, and `params.py` whose `dat_path` records the machine that did the sorting |
| `*_ChirpStimulus.npy`, `*_GratingDSOS.npy`, `*_contrastResponse_unified.npy` | Stimulus analyses precomputed **offline**. The app reads these; it does not generate them, so a run without them cannot be analysed for that stimulus |
| `*.classification_MC.txt` | Manual classification: `<visionID>  All/Path/To/Group/` |

Two files sit in `<prep>/kilosort25/stimuli/`:

- **`<prep>.json`** — the authoritative stimulus manifest. Per run it holds the
  protocol, full parameter set, `epochStarts`/`epochEnds` in samples,
  `frameTimesMs`, array ID, and the experimenter's notes and animal metadata
  (strain, injection, age). **The app does not read it.** It is the best source
  for trial alignment and for knowing what a run actually was.
- **`<prep>.txt`** — a human-readable summary of the same thing. Incomplete and
  sometimes wrong; prefer the JSON.

## The analysis

The goal is to partition several hundred sorted units into RGC types. The
pipeline, roughly:

1. `DataManager.get_cell_physics` assembles per-cell features from Vision data.
2. `analysis_core.build_feature_matrix` turns them into a weighted matrix:
   PCA blocks for the temporal STA, the autocorrelogram, the grating
   direction-tuning curve shape, and the chirp PSTH shape, plus RF long/short
   diameter as scalars. Weights are user-set sliders (`constants.py` holds the
   defaults and the reasoning behind them).
3. UMAP embeds it; Ward or K-Means clusters the embedding.
4. Clusters become groups in the tree view, which the user curates by hand and
   exports as a classification file.

**The validation criterion is the mosaic.** A real RGC type tiles the retina —
its receptive fields cover the array without overlapping, like tiles. A
candidate cluster whose RFs sit on top of each other is contaminated (two types
merged, or duplicate units). The `RFMapWidget` beside the UMAP and Feature
Extraction panels exists for exactly this check: lasso a group, see whether its
RFs tile.

Note that only ~10–15% of the RGCs under the array get sorted, so most types
will look sparse. Nearest-neighbour regularity and coverage factor both degrade
badly under that kind of subsampling; **RF overlap does not** — removing cells
can never make two remaining RFs overlap. Overlap is therefore the trustworthy
mosaic statistic here.

## Traps

**Vision ID = Kilosort `cluster_id` + 1**, except in vision-only mode where they
are equal. Always go through `DataManager.get_vision_id_for_cluster`. Off-by-one
here is silent and produces plausible-looking nonsense.

**Cells with no STA collapse into a fake cluster.** `get_raw_feature_blocks`
gives a cell with no RF fit an all-zero temporal row.
`build_feature_matrix` guards the case where *every* cell lacks an STA but not
the mixed case, so all such cells project to one identical point in the temporal
PCA block — and since temporal defaults to weight 10, that dominates the
embedding. It looks like a beautifully tight cell type. It is "cells with no
receptive field". The same applies to the grating and chirp zero-sentinels.
Not yet fixed.

**Feature weights are not what they look like.** A block's contribution to
Euclidean distance scales with `n_columns × weight²`. At defaults, temporal STA
(4 PCs × 10) contributes ~400 against ACG's (4 PCs × 1) ~4. The default
embedding is essentially "cluster on STA timecourse alone".

**Stixel size varies between preparations** and RF fits are only as good as it
allows. A 200 µm stixel against a ~100–300 µm mouse RF leaves the Gaussian fit
essentially unconstrained, so RF diameters from such a run are not comparable
with those from a 40–50 µm run.

**Check the animal before assuming wild-type.** The JSON records strain and
injection. An `rd10` retina with an optogenetic rescue (e.g. `Grm6-waChR` in ON
bipolar cells) has no photoreceptor-driven vision, so the ON/OFF axis the usual
taxonomy rests on partly collapses, and rd10 retina shows strong ~5–15 Hz
network oscillation that dominates the ACG as a *network* property rather than a
cell-type signature.

**Array IDs in the 500–1500 range are the Litke 512**, resolved by
`electrode_map.determine_array_type`. The recorded `arrayPitch` of 60 µm is
correct for it.

## Code layout

- `src/analysis/data_manager.py` — loading, caching, the per-cell feature SSOT
- `src/analysis/analysis_core.py` — STA metrics, EI computation, feature matrix
- `src/analysis/cross_run_matcher.py`, `reference_bridge.py` — bridging runs
- `src/gui/main_window.py` — the shell, tree and table views, context menus
- `src/gui/callbacks.py` — tree manipulation, loading, saving
- `src/gui/panels/` — one module per analysis tab
- `src/gui/recent_paths.py` — file-dialog location memory (QSettings)
