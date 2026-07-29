# ISI refractory period in milliseconds
ISI_REFRACTORY_PERIOD_MS = 1.0


# EI correlation threshold for considering duplicates
EI_CORR_THRESHOLD = 0.8


# ---------------------------------------------------------------------------
# Feature matrix & clustering defaults (used by dynamic clustering pipeline)
# ---------------------------------------------------------------------------

# PCA component counts for array-valued features
TEMPORAL_PCA_COMPONENTS = 4  # was 5 — trimmed as part of the feature-set
# consolidation pass (time_to_peak scalar
# removed as redundant with this block, so
# this doesn't need to carry as much alone)
ACG_PCA_COMPONENTS = 4  # was 3 — raised to match temporal's PC count
# so neither block structurally dominates
# the embedding just by having more raw
# dimensions; relative importance is now
# controlled purely by the weight below
GRATING_PCA_COMPONENTS = 4  # PCA on the pooled direction-tuning curve
# shape (grating_calc.pooled_direction_
# tuning_curve) — replaces the DSI/OSI/
# peak_rate/angle scalar block, which
# couldn't distinguish differently-shaped
# curves sharing the same DSI/OSI value
CHIRP_PCA_COMPONENTS = 4  # PCA on the L2-normalized chirp PSTH SHAPE
# (get_chirp_data_for_cluster['psth_mean']).
# Like grating, encodes response shape, not a
# scalar quality index — two cells with the same
# QI can have very different chirp responses.
# Not peak-aligned: the chirp stimulus is
# time-locked across all cells, so temporal
# position within the PSTH is meaningful.

# Chirp embedding QI gate: cells with quality_index below this contribute a
# zero sentinel row (kept out of the PCA) instead of injecting noise. Only
# affects the UMAP embedding, never the chirp panel display. Conservative
# default of 0.0 gates out only NaN/silent cells; raise per data.
CHIRP_MIN_QI = 0.0

# Minimum stimulus repeats for the chirp quality index to mean anything.
#
# The QI is a ratio of two variances both estimated from the repeats, so with
# few repeats it is not merely noisy — it stops tracking anything real. A
# split-half check on this data (compute the QI on odd trials, again on even
# trials, correlate) gives Spearman +0.86 on a 10-repeat run and -0.14 on a
# 5-repeat run: at 5 repeats the measure does not predict itself, and half the
# cells clearing a 0.25 threshold do so on under 25 spikes per trial.
#
# Runs below this are not hidden — the QI column still shows its values, but
# flagged (see HighlightStatusPandasModel) so the number is never read as a
# clean criterion. Baden et al. (2016) used 0.45 on chirp with many repeats.
CHIRP_MIN_REPEATS_FOR_QI = 8


# STA peak-to-RMS above which a cell is taken to have a real receptive field.
#
# The measure is max|STA| / RMS(STA) over the whole volume. It is bimodal on
# real data: on 20260715A/data010 (670 cells) noise STAs pile up at ~4.5 and
# cells with a receptive field spread from ~25 to ~50, with an almost empty
# valley between 10 and 25. Anything in that valley is a defensible cut; 15
# sits in the middle of it.
#
# Validated against the one thing that must hold if an STA belongs to its
# cell: spike count predicts STA quality (Spearman +0.57 there), and no cell
# with under 100 spikes clears the threshold. Where that relationship is
# absent the .sta does not belong to the sort — see
# DataManager.check_sta_consistency.
STA_SNR_GOOD = 15.0

# Pre-filter thresholds (The Bouncer)
DEFAULT_MIN_STA_STD = 1e-5  # Below this std, STA is considered flat noise
DEFAULT_MAX_RF_AREA = 300.0  # Above this, RF is artifact-scale

# HDBSCAN
DEFAULT_HDBSCAN_MIN_CLUSTER_SIZE = 5

# Default feature weights (UI slider starting values; range 0-10 for all).
#
# Consolidated feature set: firing_rate, isi_violations, time_to_peak, and
# ellipticity were removed as standalone weighted scalars. firing_rate/
# isi_violations are unit-quality (QC) metrics, not functional-identity
# signals, and belong in the spike-sorting QC pipeline rather than a
# functional-clustering embedding. time_to_peak is a derived summary
# statistic of the temporal STA that's already captured (non-independently)
# by the temporal PCA block above — keeping both let PC1 (which usually
# tracks fast/slow kinetics) and time_to_peak double-count the same signal,
# silently inflating temporal's effective weight beyond what
# DEFAULT_WEIGHT_TEMPORAL alone would suggest. ellipticity is dropped as a
# single scalar weight in favor of feeding the two RF axis lengths
# (rf_long_diameter, rf_short_diameter — see get_cell_physics) as separate
# scalars, letting PCA/UMAP find whatever size/shape relationship matters
# rather than pre-committing to one derived ratio.
DEFAULT_WEIGHT_TEMPORAL = 10.0  # max — temporal STA shape is the
# primary functional-identity signal
DEFAULT_WEIGHT_ACG = 1.0  # low — starting point, tune per data
DEFAULT_WEIGHT_RF_DIAMETER = 6.0  # high — RF size is a strong secondary
# signal; applies to both long and
# short axis diameters together
# (combined under one slider — see
# umap_panel.py's "RF Diameter" row)
DEFAULT_WEIGHT_GRATING_DSOS = 3.0  # starting point for the grating
# direction-tuning-shape PCA block
# (see GRATING_PCA_COMPONENTS) —
# moderate, since DS/OS tuning is
# real but only a subset of cells
# will have meaningful curves
DEFAULT_WEIGHT_CHIRP = 3.0  # parity with grating — chirp is the other
# "stimulus response shape" block; both are
# real but partial signals (only cells with a
# chirp response contribute a non-sentinel row)


# cell type labels
LS_CELL_TYPE_LABELS = [
    "OnP",
    "OffP",
    "OnM",
    "BlueOffM",
    "OffM",
    "OnS",
    "OffS",
    "SBC",
    "BT",
    "OnMystery",
    "OffMystery",
    "OffBoring",
    "OnWiggles",
    "OnLarge",
    "OffLarge",
    "Xmas",
    "RB",
    "Tufted",
    "BlueMystery",
    "A1",
    "Amacrine",
    "InterestingIfTrue",
    "BigMas",
    "Spotty",
    "Shadow",
    "Blobby",
    "BluePeaky",
]
