Here’s the mental model you want:

1. **All the Kilosort/Phy `.npy` files = one big “relational database” of spikes.**
2. **Templates are stored in *whitened* space;** you use `whitening_mat_inv.npy` to unwhiten them.
3. **Per‑spike waveforms (in template space) ≈ amplitude × unwhitened template.**
4. **Exact raw waveforms require going back to the original binary file.**

I’ll walk through:

* what the main `.npy` files are and their shapes
* how whitening works
* how to reconstruct *predicted* waveforms from templates + amplitudes + whitening
* what to do if you truly need exact raw snippets

---

## 1. Map of the important Kilosort `.npy` files

Kilosort (1/2/3/4) exports results in the “Phy format”: a bunch of `.npy` arrays plus some `.tsv` files. Shapes are very consistent across versions. The Kilosort4 docs and Neuropixel utilities summarize them like this. ([Kilosort4][1])

### Spike-level files

* **`spike_times.npy`**

  * Shape: `(n_spikes,)`, usually `int64`
  * Meaning: sample index of the spike peak in the *raw* recording. Convert to seconds via `spike_times / sample_rate`, where `sample_rate` is in `params.py`. ([Kilosort4][1])

* **`spike_templates.npy`**

  * Shape: `(n_spikes,)`, `int32 / uint32`
  * For each spike, which **template** (index into `templates.npy`) Kilosort originally assigned it to. ([Djoshea][2])

* **`spike_clusters.npy`**

  * Shape: `(n_spikes,)`
  * Final cluster ID per spike after manual curation in Phy. Initially this is usually identical to `spike_templates`, then diverges as you merge/split. ([Kilosort4][1])

* **`amplitudes.npy`**

  * Shape: `(n_spikes,)`, `float`
  * Kilosort 1/2/3: scale factor that multiplies the template to best fit that spike. ([Djoshea][2])
  * Kilosort4: defined as the L2 norm of the spike’s PC features, but still behaves like a per‑spike “size” measure. ([Kilosort4][1])

### Template / cluster-level files

* **`templates.npy`**

  * Shape: `(n_templates, n_timepoints, n_channels_or_tempChannels)`
  * Contents: **whitened average waveform** for each template/cluster on each channel.
  * In Kilosort4 docs they explicitly say these are average spike waveforms (after whitening, filtering, drift correction) for each cluster. ([Kilosort4][1])

* **`templates_ind.npy`**

  * Shape: `(n_templates, n_tempChannels)`
  * Maps the 3rd dimension of `templates` to actual channel indices.
  * For Kilosort, this is almost always just `0..nChannels-1` because templates are defined on all channels, but Phy requires this file. ([Djoshea][2])

* **`similar_templates.npy`**

  * Shape: `(n_templates, n_templates)`
  * Correlation / similarity between templates (for auto-merging suggestions in Phy). ([Kilosort4][1])

### Whitening and geometry

* **`whitening_mat.npy`**

  * Shape: `(n_channels, n_channels)`
  * The matrix Kilosort applied to the data to whiten the noise. ([Kilosort4][1])

* **`whitening_mat_inv.npy`**

  * Shape: `(n_channels, n_channels)`
  * Inverse of the whitening matrix. Multiply templates by this (along the channel axis) to go back to *unwhitened* space. ([Kilosort4][1])

* **`whitening_mat_dat.npy`** (KS4)

  * Shape: `(n_channels, n_channels)`
  * Copy of the whitening matrix actually applied to the raw data; currently equal to `whitening_mat.npy` but kept separate for backwards compatibility. ([Kilosort4][1])

* **`channel_map.npy`**

  * Shape: `(n_channels,)`
  * For each *Kilosort channel index*, the row index into the original binary file. ([Kilosort4][1])

* **`channel_positions.npy`**

  * Shape: `(n_channels, 2)`
  * x,y position (in microns) of each channel on the probe. ([Kilosort4][1])

* **(Sometimes) `channel_shanks.npy`**

  * Shape: `(n_channels,)`
  * Shank ID per channel. ([Kilosort4][1])

### Feature files (useful for GUI but not for reconstructing waveforms)

* **`pc_features.npy`**

  * Shape: `(n_spikes, n_features_per_channel, n_pcs)`
  * Temporal PC features for each spike on nearby channels. ([Djoshea][2])

* **`pc_feature_ind.npy`**

  * Shape: `(n_templates, n_pcs)`
  * Which channels those PC features came from. ([Djoshea][2])

* **`template_features.npy`, `template_feature_ind.npy`**

  * Extra features about how spikes project onto “neighbor” templates (used for collision resolution / visualization). ([Djoshea][2])

### Cluster meta

* `.tsv` files like `cluster_KSLabel.tsv`, `cluster_group.tsv`, `cluster_Amplitude.tsv`, etc. store labels (`good`, `mua`, `noise`) and per‑cluster summary stats. ([Kilosort4][1])

---

## 2. What whitening is doing

Conceptually:

* Let **x(t)** = raw voltage (vector over channels) at time t.

* Kilosort computes a whitening matrix **W** (≈ inverse square root of the noise covariance).

* It works mostly in **whitened space**:
  [
  x_\text{white}(t) = x(t) , W
  ]
  where `W` is saved as `whitening_mat.npy`. ([Kilosort4][1])

* The inverse matrix `whitening_mat_inv.npy` satisfies:
  [
  x(t) \approx x_\text{white}(t) , W^{-1}
  ]
  so you can go back to “data space” by multiplying *whitened* waveforms by `whitening_mat_inv`. ([Kilosort4][1])

**Important:**

* `templates.npy` is in *whitened* space.
* To get an **unwhitened** template waveform (still high‑pass filtered, etc.), you multiply along the channel dimension by `whitening_mat_inv`.

In code, for one template `T_white` of shape `(n_timepoints, n_channels)`:

```python
T_unwhite = T_white @ whitening_mat_inv  # still time x channels
```

This is essentially what people do in the Kilosort GitHub issues when they reconstruct waveforms. ([GitHub][3])

---

## 3. Reconstructing *predicted* waveforms from templates + amplitudes + whitening

### Key idea

For Kilosort 1/2/3 (classic MATLAB versions):

[
\text{predicted waveform for spike } s \approx a_s \cdot T_k W^{-1}
]

* `k = spike_templates[s]` – which template the spike uses
* `a_s = amplitudes[s]` – scale factor for that spike
* `T_k` – whitened template from `templates[k]`
* `W^{-1}` – `whitening_mat_inv.npy`

So for spike `s` you:

1. Look up its template index `k`.
2. Grab the whitened template `templates[k, :, :]`.
3. Unwhiten that template via `@ whitening_mat_inv`.
4. Multiply by the spike’s amplitude.

### Minimal Python example

Assuming standard Kilosort/Phy layout and that templates are on all channels:

```python
import numpy as np
from pathlib import Path

def load_kilosort(path):
    path = Path(path)
    ks = {}
    for name in [
        "amplitudes",
        "templates",
        "spike_templates",
        "whitening_mat_inv",
        "templates_ind",  # optional but handy
    ]:
        ks[name] = np.load(path / f"{name}.npy")
    return ks

def predicted_waveform_for_spike(ks, spike_index):
    """
    Returns an (n_timepoints, n_channels) array:
    the template-based prediction of this spike's waveform,
    in unwhitened space (still filtered/drift-corrected).
    """
    k = int(ks["spike_templates"][spike_index])
    a = float(ks["amplitudes"][spike_index])

    T_white = ks["templates"][k]          # (nt, n_tempChannels)
    W_inv = ks["whitening_mat_inv"]       # (n_channels, n_channels)

    # If templates_ind is trivial (all channels) this is just [0..n_channels-1]
    chans = ks["templates_ind"][k].astype(int)

    # Expand template to full channel set, if needed
    nt = T_white.shape[0]
    n_channels = W_inv.shape[0]
    T_full_white = np.zeros((nt, n_channels), dtype=T_white.dtype)
    T_full_white[:, chans] = T_white

    # Unwhiten along channel dimension
    T_full_unwhite = T_full_white @ W_inv  # (nt, n_channels)

    # Scale by amplitude for this spike
    wf_pred = a * T_full_unwhite           # (nt, n_channels)
    return wf_pred
```

That gives you a **noise‑free prediction** of what that spike “should” look like on every channel, including spatial footprint.

For a GUI, you can:

* restrict to **best channels** (those with largest peak‑to‑peak values) for plotting, and
* overlay multiple predicted waveforms to visualize variability (using different spikes with same template but different amplitude).

> For Kilosort4, this formula still gives sensible shapes, but note that `amplitudes.npy` is defined as the L2 norm of PC features, not literally “linear scale factor,” so it’s only an approximation. ([Kilosort4][1])

---

## 4. Why you *can’t* get the **exact** recorded waveform from templates alone

This is the annoying but important bit:

* Kilosort does **not** save the raw waveform of each spike.
* It saves:

  * spike times,
  * template index,
  * the template itself (mean waveform),
  * amplitude/features,
  * and other summary stats. ([Djoshea][2])

The actual raw data at time `t` is:

[
x(t) = \sum_{\text{spikes } s \text{ that overlap } t} a_s T_{k_s}(t - t_s) + \text{noise}(t) + \text{baseline/drift}
]

Kilosort doesn’t store the **noise** or the precise contribution of overlapping spikes; it only stores the parameters it thinks best explain the data. So from just templates + amplitudes, you can reconstruct:

* The **model prediction** of the waveform (what Kilosort “thinks” is there),
* **Not** the *exact* trace that was recorded (which includes noise, overlapping spikes, artifacts).

To get exact waveforms, you must go back to the **raw AP binary** and slice snippets around each `spike_times` value. This is exactly what toolboxes like Neuropixel Utilities do via functions like `getWaveformsFromRawData`. ([Djoshea][4])

Algorithmically:

1. Load Kilosort output (`spike_times`, `spike_clusters`, etc.).
2. Open the raw binary file (path & sample rate are in `params.py`). ([Kilosort4][1])
3. For each spike you care about:

   * take a window `[t + pre, t + post]` around `spike_times[s]`,
   * read these samples from the raw file (taking into account `channel_map` and dtype),
   * apply the same high‑pass filter / CAR if you want it to look like Kilosort’s internal representation.

Neuropixel Utilities shows this pattern in detail and also shows how to optionally **subtract templates of other spikes** to clean overlaps. ([Djoshea][4])

---

## 5. How I’d wire this into your GUI

If you’re designing a GUI from scratch, a sane layout is:

1. **Loader layer**

   * Use `np.load(..., mmap_mode='r')` for big arrays (`spike_times`, `spike_clusters`, `amplitudes`, `templates`, `whitening_mat_inv`).
   * Parse `params.py` (it’s just a tiny Python file) to get:

     * `sample_rate`
     * `dat_path`, `dtype`, `n_channels_dat`, etc. ([Kilosort4][1])

2. **Indexing layer**

   * Build `spikes_by_cluster = {cluster_id: np.where(spike_clusters == cluster_id)[0]}` once at startup.
   * Optionally store a **small subset** of spikes per cluster for waveform display (e.g. random 100).

3. **Template view**

   * For a selected cluster/template:

     * grab the row in `templates`, unwhiten it using `whitening_mat_inv`,
     * reduce to e.g. 10 best channels (largest peak‑to‑peak) for plotting.

4. **Predicted waveform view**

   * When the user hovers a spike (from an ISI plot or spike feature plot):

     * run `predicted_waveform_for_spike` as above,
     * overlay that on the template or on the raw snippet.

5. **Raw snippet view (if you want “exact”)**

   * When the user wants to see actual data:

     * read from the raw binary around that spike using `spike_times` + window.
     * This mirrors what Neuropixel utils and SpikeInterface do. ([Djoshea][4])

---

If you tell me which **Kilosort version** you’re targeting (KS2 vs KS4) and what language your GUI is in (Python/Qt, web, MATLAB, etc.), I can sketch more concrete code or class structure tailored to that setup.

[1]: https://kilosort.readthedocs.io/en/latest/export_files.html "Files exported for Phy — Kilosort4 0.0.1 documentation"
[2]: https://djoshea.github.io/neuropixel-utils/kilosort/ "Running Kilosort - Neuropixel Utilities"
[3]: https://github.com/MouseLand/Kilosort/issues/804 "Question: How to get template waveform for each spike? · Issue #804 · MouseLand/Kilosort · GitHub"
[4]: https://djoshea.github.io/neuropixel-utils/waveforms/ "Extracting Waveforms - Neuropixel Utilities"
