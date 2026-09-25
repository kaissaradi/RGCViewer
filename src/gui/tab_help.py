"""What each analysis tab shows, in plain words (PLAN.md Q49).

Shift+F1, or the "?" button in the header, shows the entry for the current
tab. One short paragraph on what the plots are, then what to look for.
Keep these true to the code; the references are in docs/design/rgc_types.md.
"""

TAB_HELP = {
    "Standard": (
        "The basic health of one cell's spike train.",
        ["Spatial template: the cell's mean spike on its main electrodes (Kilosort).",
         "Autocorrelation: how often the cell fires again at each lag after a spike. "
         "A dip near 0 ms is the refractory period; a peak at a few ms means bursts.",
         "Inter-spike interval: the red dashed line is the refractory limit. Many intervals "
         "below it mean the unit mixes spikes of more than one cell.",
         "Firing rate over the recording, with the amplitude (right axis) and the stimulus "
         "blocks shaded. The title warns when the cell went quiet during a block."]),
    "Chirp": (
        "Responses to the chirp: light steps, then a frequency sweep, then a contrast sweep.",
        ["Raster: one row per repeat. PSTH: the mean firing rate.",
         "ON/OFF bars: the response to light on and light off. Frequency and contrast "
         "tuning: how the response follows the sweeps.",
         "QI (quality index): how alike the repeats are, 0 to 1. Low values mean noise."]),
    "Contrast": (
        "Response against stimulus contrast, per light level.",
        ["A concatenated sort that spans runs at different light levels shows how the "
         "contrast response changes as the light dims.",
         "'No response' is often the real result at low light."]),
    "Grating": (
        "Drifting gratings: which direction and orientation the cell prefers.",
        ["Polar plot: firing rate for each motion direction, ± 1 SD across trials.",
         "DSI / OSI: direction and orientation selectivity (0 = none, 1 = only one "
         "direction / orientation), with a shuffle test across all conditions.",
         "Rasters around the polar plot: the spikes for each direction.",
         "Angles are the protocol's label θ. The lab's grating protocol moves the bars toward "
         "θ + 180°, in the same frame as the STA (PLAN.md, open defects).",
         "Compare DS runs… shows every DS grating run of the prep, or of every prep, in one "
         "frame: on the screen, or from the direction to the optic disc."]),
    "EI": (
        "The electrical image: the cell's mean voltage on every electrode around its spike.",
        ["The biggest signal marks the soma. A small signal that moves away over time is "
         "the axon (Array ▸ Find the Optic Disc uses these).",
         "The array is drawn turned to match the screen when Array ▸ Align Array Views is on."]),
    "STA": (
        "The spike-triggered average of the white-noise movie: the cell's receptive field.",
        ["The image is what the screen looked like, on average, just before a spike. "
         "Gray is zero; bright is ON, dark is OFF.",
         "The time course (right) shows when, in the frames before the spike, the light mattered. "
         "One lobe = sustained, two opposite lobes = transient (more biphasic).",
         "Heatmap and space-time views show the same STA in other ways."]),
    "UMAP": (
        "A map of all cells: cells that respond alike sit close together.",
        ["Built from the STA time course, autocorrelation, RF size and ON/OFF polarity "
         "(tick features on or off above the map).",
         "Color by group to see whether a class forms one island. Select cells on the map to "
         "see them everywhere else."]),
    "Types": (
        "The whole run by class.",
        ["Barcode: one row per cell, one band per class. A red tick marks a cell that fits "
         "another type better than its own.",
         "Mosaics: a real type tiles the retina, so its RFs sit about one RF apart. Red "
         "outlines overlap too much; grey rings are look-alikes in a gap.",
         "Suggest classes and the Type atlas use the cells the lab has already classified."]),
    "Waveforms": (
        "Individual spikes from the raw file, to judge how well the unit is isolated.",
        ["The cloud of spikes around the mean trace, split by amplitude, and a PCA view.",
         "Needs the raw data file (File ▸ Load Raw Data File)."]),
    "Raw": (
        "The raw voltage with the cell's spikes marked, or spike rasters without a raw file.",
        ["With a raw file: step through the recording and check each spike by eye.",
         "Without one: one row per cell over the whole recording; zoom in to see every spike."]),
}


# Shown when "?" is pressed on the welcome screen, before a run is open.
START_HELP = (
    "<p><b>Getting started</b></p><ul style='margin-left:-16px'>"
    "<li style='margin-bottom:6px'>Open run (Ctrl+O): pick a Kilosort folder. Encore finds the "
    "Vision files and the stimulus files next to it.</li>"
    "<li style='margin-bottom:6px'>Pick a cell in the list on the left; every tab then shows "
    "that cell. Up / Down steps through the list.</li>"
    "<li style='margin-bottom:6px'>The Types tab shows the whole run by class. The ? button "
    "(Shift+F1) explains the plots on any tab.</li></ul>"
    "<p style='color:gray'>F1 lists the keyboard shortcuts.</p>")


def html_for(tab_name: str) -> str:
    entry = TAB_HELP.get(tab_name)
    if entry is None:
        return f"<p>No notes for the {tab_name} tab yet.</p>"
    lead, points = entry
    items = "".join(f"<li style='margin-bottom:6px'>{p}</li>" for p in points)
    return (f"<p><b>{tab_name}</b> — {lead}</p><ul style='margin-left:-16px'>{items}</ul>"
            "<p style='color:gray'>F1 lists the keyboard shortcuts.</p>")
