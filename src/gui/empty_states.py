"""What an empty tab says, so a new user knows why and what would fill it (PLAN.md Q49)."""

_FILES = {
    # DataManager._ANALYSIS_GLOBS / find_analysis_files (names and folders).
    "chirp": ("chirp", "a .npy file with 'Chirp' in its name"),
    "grating": ("drifting-grating", "a .npy file with 'Grating' or 'DSOS' in its name"),
    "contrast": ("contrast-response", "a .npy file with 'contrast' or 'Contrast' in its name"),
}


def no_stimulus(kind: str) -> str:
    """The run has no analysis of this stimulus at all."""
    what, pattern = _FILES[kind]
    return (f"This run has no {what} analysis.\n\n"
            f"Encore looks for {pattern} in the run folder, its ksfiles/ folder, or the "
            f"folder above.\n"
            f"If the stimulus ran in another run of this retina, File ▸ Map Reference Run "
            f"can borrow its responses.")


def no_cell_response(kind: str, cluster_id) -> str:
    """The run has the stimulus, but not this cell."""
    what, _pattern = _FILES[kind]
    return (f"Cell {cluster_id} has no {what} response.\n\n"
            f"It is not in the {what} file: it may have been sorted after that analysis "
            f"was made, or fired too little during the stimulus.")
