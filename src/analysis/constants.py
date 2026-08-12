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

# Bin width the reported chirp QI is computed at, after rebinning.
#
# The QI is a variance ratio over binned spike counts, so it is a property of
# (cell, bin width, trial count) rather than of the cell alone. On one run of
# 20260715A the number of cells clearing 0.45 runs 4 -> 54 -> 111 -> 167 -> 206
# as the bins widen 5 -> 10 -> 20 -> 40 -> 100 ms, from identical spikes. Chirp
# files on this drive ship at both 5 ms and 20 ms, so raw QIs from two files
# cannot be compared until they are put on a common footing.
#
# 20 ms is chosen because it is what the existing files mostly use, so the
# column keeps the values curators are already calibrated to: rebinning the
# 5 ms 20260715A file to 20 ms reproduces the older 20 ms file's result for the
# same run (111/887 cells over 0.45 vs 109/707). Finer than this and Poisson
# noise in the per-bin counts swamps the signal (~0.03 spikes/bin at 5 ms).
CHIRP_QI_BIN_MS = 20.0

# Spikes per trial below which the chirp QI is not reported at all.
#
# This is hygiene, NOT a fix for QI inflation. The intuition that a scale-free
# variance ratio lets a near-silent cell fake a high QI does not survive
# contact with the data: measured on 20260715A (887 cells, 20 ms bins), QI
# correlates +0.75 with spikes per trial, and the top-decile-QI cells fire a
# median 546 spikes/trial against 51 for everyone else. High QI goes with high
# rate, not low. Gating at 10 spikes/trial removes 25.7% of cells but costs
# only 3 of the 154 that clear QI 0.45; pushing it to 200 to catch more would
# start deleting genuinely active cells (38 lost).
#
# What it is for: 10 spikes over a 25 s chirp is 0.4 Hz. Below that there is
# not enough data to estimate either variance in the ratio, and the honest
# display is a blank cell meaning "not measured" rather than a number that
# invites ranking. The rate-dependence of the QI itself is intrinsic and no
# threshold fixes it — the cross-validated stimulus-filter R^2 is the measure
# that is nearly rate-independent (+0.11), at the cost of being noisier.
CHIRP_MIN_SPIKES_PER_TRIAL = 10.0


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

# Default feature weights and which blocks start checked.
#
# 2026-08-12 product call: Temporal STA, ACG, and RF diameter start on
# and at 10/10. Grating and chirp stay available but start off — they
# are real type signals for a subset of cells, not the default embedding.
# A block's Euclidean share is still n_columns × weight², so three blocks
# at the same slider value are not equal if their column counts differ
# (two RF diameter scalars vs four STA PCs).
#
# firing_rate / isi_violations are not embedding features. Mean rate
# alone is a QC metric and was dropped for that reason. Re-adding rate
# as a *functional* feature is parked: it has to separate high-baseline
# RGCs and bursty vs tonic firing without letting spike count dominate.
# See PLAN.md parked work.
#
# time_to_peak is a summary of the temporal STA already in the PCA block.
# ellipticity is dropped in favour of the two RF axis lengths as separate
# scalars.
DEFAULT_USE_TEMPORAL = True
DEFAULT_USE_ACG = True
DEFAULT_USE_RF_DIAMETER = True
DEFAULT_USE_GRATING_DSOS = False
DEFAULT_USE_CHIRP = False

DEFAULT_WEIGHT_TEMPORAL = 10.0
DEFAULT_WEIGHT_ACG = 10.0
DEFAULT_WEIGHT_RF_DIAMETER = 10.0
# Slider rest positions if the user turns these on. Not used at startup.
DEFAULT_WEIGHT_GRATING_DSOS = 10.0
DEFAULT_WEIGHT_CHIRP = 10.0


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
