"""Shared constants for the Dash app: default plot preferences, dropdown
options, cache paths and session_store key names. See the migration plan
(refactored-moseying-twilight.md) for how these map onto the original
PyQt5 dialogs.
"""
import os

EPD_CACHE_DIR = os.environ.get(
    "SOLOLAB_EPD_CACHE_DIR",
    os.path.join(os.path.dirname(__file__), ".epd_cache"),
)

INSTRUMENT_ORDER = ["stix", "hfr", "tnr", "epd"]

STIX_POLL_OPTIONS = ["mean", "median", "min", "max", "P_25", "P_75"]
STIX_POLL_DEFAULT = "mean"

RPW_POLL_OPTIONS = ["max", "mean", "median", "min", "P_25", "P_75"]
RPW_POLL_DEFAULT = "max"

EPD_RESAMPLE_OPTIONS = ["30sec", "1min", "2min", "5min", "10min"]
EPD_RESAMPLE_DEFAULT = "1min"
EPD_PARTICLE_OPTIONS = ["Electron", "Proton"]

PLOT_TYPE_OPTIONS = ["spectrogram", "time profiles", "overlay"]
RPW_OVERLAP_OPTIONS = ["Only TNR", "Only HFR", "Both"]

# Fixed preview channels/frequencies used by the import-dialog "Plot Preview"
# buttons (independent of the Plot Preferences selections) - matches the
# hardcoded values in the original PyQt5 dialogs.
EPD_PREVIEW_CHANNELS = [2, 6, 14, 18, 26]
RPW_HFR_PREVIEW_FREQS = [500, 3500, 13000]
RPW_TNR_PREVIEW_FREQS = [50, 100, 400]

DEFAULT_PLOT_PREFS = {
    "stix": {
        "type": "spectrogram",
        "logy": False,
        "logy_countrate": False,
        "logy_energy_overlay": False,
        "logy_countrate_overlay": False,
        "logz": True,
        "energy_ranges": [[4, 12], [16, 28]],
        "energy_range_enabled": False,
        "energy_range": [4, 28],
        "smoothing_points": 1,
    },
    "rpw": {
        "type": "spectrogram",
        "logy": False,
        "logy_intensity": False,
        "logy_frequency_overlay": False,
        "logy_intensity_overlay": False,
        "logz": True,
        "invert_y": True,
        "overlay": "Both",
        "freq_range_enabled": False,
        "freq_min": 100.0,
        "freq_max": 8000.0,
        "smoothing_points": 1,
        "selected_frequencies": [],
    },
    "epd": {
        "logy": False,
        "selected_channels": [2, 6, 14, 18, 26],
    },
}

DEFAULT_INSTRUMENT_STATUS = {
    "stix": {"loaded": False},
    "rpw_hfr": {"loaded": False},
    "rpw_tnr": {"loaded": False},
    "epd": {"loaded": False},
}

# session_store key name helpers -------------------------------------------------


def rpw_key(data_type, suffix):
    """data_type in {'hfr','tnr'} -> e.g. rpw_key('hfr','psd_final') == 'rpw_hfr_psd_final'."""
    return f"rpw_{data_type}_{suffix}"
