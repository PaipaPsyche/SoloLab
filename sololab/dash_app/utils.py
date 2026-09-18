"""Small shared helpers used across pages: uploaded-file handling, instrument
status badge text, and a generic editable-list modal (reused for STIX energy
ranges, RPW frequencies and EPD channels in pages/plot_prefs.py).
"""
import base64
import contextlib
import os
import shutil
import tempfile
from datetime import datetime

import dash_bootstrap_components as dbc
import pandas as pd
from dash import Input, Output, State, callback, dash_table, html
from dash.exceptions import PreventUpdate

from sololab.values import std_date_fmt

# All datetime text inputs in the Dash app (bkg start/end, combined-plot
# date range) use this plain, unambiguous format.
DT_INPUT_FMT = "%Y-%m-%d %H:%M:%S"


def parse_dt_input(value):
    """DT_INPUT_FMT string -> datetime, or None if value is falsy."""
    if not value:
        return None
    return datetime.strptime(value.strip(), DT_INPUT_FMT)


def _to_pydatetime(value):
    """datetime | numpy.datetime64 | pandas.Timestamp -> datetime.datetime.
    rpw_read.py's PSD dicts store `time` as numpy.datetime64 (via
    _coerce_time_data), while stix_read.py's counts dicts store plain
    datetime.datetime objects - callers here deal with both."""
    return pd.Timestamp(value).to_pydatetime()


def format_dt_input(value):
    """datetime | numpy.datetime64 -> DT_INPUT_FMT string, or "" if None."""
    if value is None:
        return ""
    return _to_pydatetime(value).strftime(DT_INPUT_FMT)


def to_stix_date_str(value):
    """datetime -> sololab.values.std_date_fmt string. stix_read.py parses
    date/bkg-range strings with a strict `datetime.strptime(x, std_date_fmt)`
    (unlike rpw_read.py's more permissive _parse_date), so STIX call sites
    must format explicitly to this exact format."""
    return _to_pydatetime(value).strftime(std_date_fmt)

INSTRUMENT_LABELS = {
    "stix": "STIX",
    "rpw_hfr": "RPW-HFR",
    "rpw_tnr": "RPW-TNR",
    "epd": "EPD",
}


# Upload handling -----------------------------------------------------------------


def decode_upload(contents):
    """dcc.Upload's `contents` data-URI string -> raw bytes."""
    _, content_string = contents.split(",", 1)
    return base64.b64decode(content_string)


def bytes_to_tempfile(data, filename):
    """Write bytes to a temp file preserving the ORIGINAL filename (not just
    its extension): stix_read.stix_create_counts and rpw_read.rpw_get_data
    both parse metadata (level, instrument, spectrogram-vs-imaging) out of
    os.path.basename(pathfile), e.g. "solo_L1_stix-sci-xray-spec_...fits" or
    "solo_L2_rpw-hfr-surv_...cdf" - a randomized tempfile.mkstemp() name
    would break that parsing, so each upload gets its own temp directory
    instead and keeps its real basename."""
    tmp_dir = tempfile.mkdtemp()
    path = os.path.join(tmp_dir, filename or "upload.dat")
    with open(path, "wb") as f:
        f.write(data)
    return path


def upload_to_tempfile(contents, filename):
    return bytes_to_tempfile(decode_upload(contents), filename)


@contextlib.contextmanager
def tempfile_from_bytes(data, filename):
    """Same as bytes_to_tempfile, but as a context manager that removes the
    temp directory on exit (success or exception). bytes_to_tempfile on its
    own leaked one temp dir per Preview/Preview-w-Bkg/Load click with no
    cleanup anywhere in the app - unbounded disk growth on a long-running
    server (Backend Fixes item: Dash disk leak)."""
    path = bytes_to_tempfile(data, filename)
    try:
        yield path
    finally:
        shutil.rmtree(os.path.dirname(path), ignore_errors=True)


# Status badges ---------------------------------------------------------------------


def status_badge_content(key, info):
    """(text, dbc.Badge color) for one instrument, given its
    instrument-status-store entry."""
    label = INSTRUMENT_LABELS[key]
    if not info or not info.get("loaded"):
        return f"{label}: No data loaded", "danger"
    if key == "epd":
        text = (
            f"{label}: Loaded | {info.get('date')} | "
            f"{info.get('particle')} | {info.get('resample')}"
        )
    else:
        text = f"{label}: Loaded | {info.get('min_time')} to {info.get('max_time')}"
    return text, "success"


# Generic editable-list modal -------------------------------------------------------


def make_list_editor_modal(id_prefix, title, columns):
    """Editable-list modal reused for STIX energy ranges ([int,int] pairs),
    RPW frequencies ([float]) and EPD channels ([int]).

    columns: list of {"name": str, "id": str, "type": "numeric"}.
    The caller is responsible for: (1) opening the modal (is_open=True) in
    response to its own "Select..." button, (2) populating the table's
    initial `data` from the relevant plot-prefs list when opened, and
    (3) validating + writing the table's `data` back into plot-prefs-store
    when the modal closes (ranges/values differ per list, so that logic
    isn't generic).
    """
    return dbc.Modal(
        [
            dbc.ModalHeader(dbc.ModalTitle(title)),
            dbc.ModalBody(
                [
                    dash_table.DataTable(
                        id=f"{id_prefix}-table",
                        columns=[
                            {**c, "deletable": False, "renamable": False}
                            for c in columns
                        ],
                        data=[],
                        editable=True,
                        row_deletable=True,
                        style_cell={"textAlign": "center"},
                        style_table={"marginBottom": "0.5rem"},
                    ),
                    dbc.Button(
                        "Add row",
                        id=f"{id_prefix}-add-row-btn",
                        size="sm",
                        color="secondary",
                        outline=True,
                    ),
                    html.Div(id=f"{id_prefix}-error", className="text-danger small mt-2"),
                ]
            ),
            dbc.ModalFooter(dbc.Button("Done", id=f"{id_prefix}-done-btn", color="primary")),
        ],
        id=f"{id_prefix}-modal",
        is_open=False,
        size="md",
    )


def register_list_editor_add_row_callback(id_prefix, empty_row):
    """Wires the generic "Add row" button. Persisting validated rows back to
    plot-prefs-store is done separately by each caller (see plot_prefs.py)."""

    @callback(
        Output(f"{id_prefix}-table", "data", allow_duplicate=True),
        Input(f"{id_prefix}-add-row-btn", "n_clicks"),
        State(f"{id_prefix}-table", "data"),
        prevent_initial_call=True,
    )
    def _add_row(n_clicks, rows):
        if not n_clicks:
            raise PreventUpdate
        rows = rows or []
        return rows + [dict(empty_row)]
