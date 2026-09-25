"""What changed, in the user's words, for the welcome screen (PLAN.md Q49).

Newest first. One line each: what you can now do, and where. Keep it short;
the details live in README.md and docs/PLAN.md.
"""

WHATS_NEW = [
    ("Fixed: timing", "Firing rates were 1.5x too high and ISI / ACG lags 2/3 too short: Kilosort times were read at 30 kHz, not 20 kHz. Caches rebuild once, by themselves."),
    ("Suggested classes", "Types tab ▸ Suggest classes learns the lab's 5 named types from every "
     "classified run and suggests one per cell. Ctrl+J next to review, Ctrl+Enter accept."),
    ("Types tab", "Every class at once: a barcode of responses (a red tick marks a cell that fits "
     "another type better) and a mosaic per type (overlapping RFs in red, look-alikes in gaps)."),
    ("Type atlas", "What each named type looks like across the lab, with this run's cells on top."),
    ("Compare cells", "Ctrl+P puts the population beside the cell; the selected cell is drawn over "
     "its group; Ctrl+K pins up to 4 more for comparison."),
    ("Keyboard", "F1 lists every shortcut. Ctrl+M moves cells to a group by typing its name; "
     "Ctrl+1…9 switch tabs; Delete trashes from anywhere."),
    ("Stability", "Standard tab: amplitude over the recording and the stimulus blocks, with a "
     "warning when a cell went quiet during one."),
    ("Rasters", "No raw file? The Raw tab shows spike rasters of the whole recording."),
    ("UMAP", "ON and OFF cells no longer mix, and time-to-peak is kept."),
]
