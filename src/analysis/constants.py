# ISI refractory period in milliseconds
ISI_REFRACTORY_PERIOD_MS = 1.0


# EI correlation threshold for considering duplicates
EI_CORR_THRESHOLD = 0.8


# ---------------------------------------------------------------------------
# Feature matrix & clustering defaults (used by dynamic clustering pipeline)
# ---------------------------------------------------------------------------

# PCA component counts for array-valued features
TEMPORAL_PCA_COMPONENTS = 5
ACG_PCA_COMPONENTS = 3

# Pre-filter thresholds (The Bouncer)
DEFAULT_MIN_STA_STD = 1e-5       # Below this std, STA is considered flat noise
DEFAULT_MAX_RF_AREA = 300.0      # Above this, RF is artifact-scale

# HDBSCAN
DEFAULT_HDBSCAN_MIN_CLUSTER_SIZE = 5

# Default feature weights (UI spinbox starting values)
DEFAULT_WEIGHT_TEMPORAL = 3.0
DEFAULT_WEIGHT_ACG = 2.0
DEFAULT_WEIGHT_FIRING_RATE = 1.5
DEFAULT_WEIGHT_ISI_VIOLATIONS = 1.0
DEFAULT_WEIGHT_TIME_TO_PEAK = 1.0
DEFAULT_WEIGHT_RF_AREA = 1.0
DEFAULT_WEIGHT_ELLIPTICITY = 1.0


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
    "BluePeaky"
]
