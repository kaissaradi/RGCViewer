# CLAUDE.md — experiment, files, analysis traps

Read after `README.md`, before `docs/AGENTS.md` and `docs/PLAN.md`. Each
fact here was checked on lab data; the PLAN item gives the numbers.

## The experiment

A retina lies on a multi-electrode array (512 or 519 electrodes). The
recording is spike-sorted with Kilosort (2.5 or 4). Vision (vision7, Java)
then makes, for the sorted cells:

- `.neurons` — spike times per cell,
- `.ei` — the electrical image (mean waveform on every electrode),
- `.sta` — the spike-triggered average of a white-noise movie,
- `.params` — one row per cell: RF fit (x0, y0, SigmaX, SigmaY, Theta),
  time course, and the classification (`classID`).

Other stimuli (chirp, drifting gratings, contrast) are analysed from the
spike times into `.npy` files next to the Vision files.

## Files

```
/mnt/lab/Array-data/sorted/<prep>/<sorter>/<run>/     Vision files, stimulus .npy
/mnt/lab/Array-data/sorted/<prep>/<sorter>/<run>/ksfiles/   Kilosort output
/mnt/lab/Array-data/raw/<prep>/<run>/                 raw Litke .bin
```

`/mnt/lab` is a CIFS share over 1 GbE. Reads are slow and parallel reads are
slower, not faster. Encore writes its caches (`*.pkl`) into `ksfiles/` and
`<run>.ei.ei_index.pkl` next to the `.ei`.

## Analysis traps

1. **IDs.** Vision ID = Kilosort cluster + 1 (AGENTS.md Law 1). Derived
   `.npy` stimulus files already use Vision IDs.
2. **Vision files from another sort.** The +1 pairing is only right when
   the `.sta`/`.params` were made from the sort in `ksfiles/`. Two runs are
   known to fail: 20251212A/kilosort40/data018 (the harness default) and
   20250918A/kilosort40/data025. `DataManager.vision_sort_check()` tests
   the pairing from RF centre vs array position (PLAN.md Q32). A partial
   ID overlap is normal and proves nothing. Also suspect (Q32, Q37):
   20260721A/kilosort25 data006 and data007 (the STAs are noise, peak/RMS
   ~4) and 20260715A/kilosort25/data003 (RFs follow the cells weakly).
3. **Unmoved RF fits.** SigmaX = SigmaY = 1 exactly is Vision's first start
   value, not a fit (PLAN.md Q26). Up to 60 % of cells in some runs.
4. **Fit frame.** Vision stores y0 with y up; the STA is drawn with row 0 at
   the top. `rf_geometry.image_ellipse` converts (PLAN.md Q20).
5. **Grating times** in `*GratingDSOS.npy` are milliseconds from trial start.
   Each trial has its own preTime / stimTime (PLAN.md Q9).
6. **DS/OS significance** is one shuffle test per cell across all its
   conditions (PLAN.md Q33).
7. **Sample rate.** Kilosort's `params.py` stores it as `sample_rate`
   (20000 on this rig). Encore read `fs` until 2026-09-25 and used 30000
   (PLAN.md Q48). Every time from Kilosort spike samples depends on it.
8. **`.params` is shared with Vision.** Vision opens it read-write and saves
   it in place. Encore never edits it in place (`params_classification.py`,
   PLAN.md Q30–Q31). Close a run in Vision before Ctrl+S in Encore.

## Ground truth for Vision formats

The vision7 source and `Vision.jar` are on the lab machine under
`~/Documents/Development/MEA-fieldlab/src/vision7_symphony/`. A headless
Java check against it is in `tools/gui_harness.py` (`vision_reads_classes`).
