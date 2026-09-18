"""Import STIX data - port of ImportStixDialog in sololab_app.py.

Reference dialog for the whole app: two independent (non-exclusive)
background options (bkg file, bkg time range), a "Plot Background" modal,
and the Upload -> bytes -> session_store -> tempfile -> stix_read.* ->
plotting.* -> dcc.Graph pattern reused (via rpw_import_factory.py) by the
RPW-HFR/TNR pages.
"""
import contextlib
import logging

import dash
import dash_bootstrap_components as dbc
from dash import Input, Output, State, callback, dcc, html
from dash.exceptions import PreventUpdate

from sololab import stix_read
from sololab.dash_app import plotting
from sololab.dash_app.constants import STIX_POLL_DEFAULT, STIX_POLL_OPTIONS
from sololab.dash_app.session_store import session_store
from sololab.dash_app.utils import (
    decode_upload,
    format_dt_input,
    parse_dt_input,
    tempfile_from_bytes,
    to_stix_date_str,
)

logger = logging.getLogger(__name__)

dash.register_page(__name__, path="/import/stix", name="Import STIX")

layout = dbc.Container(
    [
        html.H3("Import STIX data", className="mt-3"),
        dbc.Alert(id="stix-alert", is_open=False, dismissable=True, className="mt-2"),
        dbc.Row(
            [
                dbc.Col(
                    [
                        dbc.Card(
                            dbc.CardBody(
                                [
                                    html.H5("Data file"),
                                    dcc.Upload(
                                        id="stix-upload",
                                        children=html.Div(
                                            ["Drag and drop or ", html.A("select a STIX FITS file")]
                                        ),
                                        className="upload-box",
                                    ),
                                    html.Div(id="stix-filename", className="text-muted small mt-1"),
                                    html.Hr(),
                                    html.H5("Background"),
                                    dbc.Checkbox(
                                        id="stix-bkg-file-enabled",
                                        label="Subtract background from file",
                                        value=False,
                                    ),
                                    html.Div(
                                        [
                                            dcc.Upload(
                                                id="stix-bkg-upload",
                                                children=html.Div(
                                                    ["Drag and drop or ", html.A("select a background FITS file")]
                                                ),
                                                className="upload-box",
                                            ),
                                            html.Div(id="stix-bkg-filename", className="text-muted small mt-1"),
                                        ],
                                        id="stix-bkg-file-row",
                                        style={"display": "none"},
                                        className="mb-2",
                                    ),
                                    dbc.Checkbox(
                                        id="stix-bkg-time-enabled",
                                        label="Subtract background from a time range",
                                        value=False,
                                    ),
                                    html.Div(
                                        [
                                            dbc.Row(
                                                [
                                                    dbc.Col(
                                                        dbc.Input(
                                                            id="stix-bkg-start",
                                                            type="text",
                                                            placeholder="YYYY-MM-DD HH:MM:SS",
                                                        )
                                                    ),
                                                    dbc.Col(
                                                        dbc.Input(
                                                            id="stix-bkg-end",
                                                            type="text",
                                                            placeholder="YYYY-MM-DD HH:MM:SS",
                                                        )
                                                    ),
                                                ]
                                            )
                                        ],
                                        id="stix-bkg-time-row",
                                        style={"display": "none"},
                                        className="mb-2 mt-1",
                                    ),
                                    html.Label("Background polling function"),
                                    dcc.Dropdown(
                                        id="stix-bkg-poll",
                                        options=STIX_POLL_OPTIONS,
                                        value=STIX_POLL_DEFAULT,
                                        clearable=False,
                                    ),
                                    html.Hr(),
                                    dbc.ButtonGroup(
                                        [
                                            dbc.Button(
                                                "Preview Data", id="stix-preview-btn", disabled=True, color="secondary"
                                            ),
                                            dbc.Button(
                                                "Preview with Background Subtraction",
                                                id="stix-preview-bkg-btn",
                                                disabled=True,
                                                color="secondary",
                                            ),
                                            dbc.Button(
                                                "Plot Background", id="stix-plot-bkg-btn", disabled=True, color="secondary"
                                            ),
                                        ],
                                        vertical=True,
                                        className="w-100 mb-2",
                                    ),
                                    dbc.Button("LOAD", id="stix-load-btn", disabled=True, color="success", className="w-100"),
                                ]
                            )
                        )
                    ],
                    md=4,
                ),
                dbc.Col(dcc.Graph(id="stix-preview-graph"), md=8),
            ]
        ),
        dbc.Modal(
            [
                dbc.ModalHeader(dbc.ModalTitle("STIX Background")),
                dbc.ModalBody(dcc.Graph(id="stix-bkg-graph")),
            ],
            id="stix-bkg-modal",
            is_open=False,
            size="lg",
        ),
    ],
    className="pb-5",
)


# --- upload handling ---------------------------------------------------------------


@callback(
    Output("stix-filename", "children"),
    Output("stix-preview-btn", "disabled"),
    Output("stix-preview-bkg-btn", "disabled"),
    Output("stix-load-btn", "disabled"),
    Input("stix-upload", "contents"),
    State("stix-upload", "filename"),
    State("session-id", "data"),
    prevent_initial_call=True,
)
def on_stix_upload(contents, filename, sid):
    if not contents:
        return "", True, True, True
    session_store.set(sid, "stix_file_bytes", (filename, decode_upload(contents)))
    return f"Selected: {filename}", False, False, False


@callback(
    Output("stix-bkg-filename", "children"),
    Input("stix-bkg-upload", "contents"),
    State("stix-bkg-upload", "filename"),
    State("session-id", "data"),
    prevent_initial_call=True,
)
def on_stix_bkg_upload(contents, filename, sid):
    if not contents:
        return ""
    session_store.set(sid, "stix_bkg_file_bytes", (filename, decode_upload(contents)))
    return f"Selected: {filename}"


@callback(
    Output("stix-bkg-file-row", "style"),
    Input("stix-bkg-file-enabled", "value"),
)
def toggle_bkg_file_row(enabled):
    return {"display": "block"} if enabled else {"display": "none"}


@callback(
    Output("stix-bkg-time-row", "style"),
    Input("stix-bkg-time-enabled", "value"),
)
def toggle_bkg_time_row(enabled):
    return {"display": "block"} if enabled else {"display": "none"}


# --- data processing -----------------------------------------------------------------


def _compute_stix_counts(sid, bkg_file_enabled, bkg_time_enabled, bkg_start, bkg_end, poll):
    """Shared logic between "Preview with Background Subtraction" and
    "LOAD" - port of ImportStixDialog._apply_background_subtraction /
    _load_stix_data. Always rereads the main file from cached bytes,
    matching the original's no-memoization behaviour."""
    filename, data = session_store.get(sid, "stix_file_bytes")
    with contextlib.ExitStack() as stack:
        path = stack.enter_context(tempfile_from_bytes(data, filename))

        if not bkg_file_enabled and not bkg_time_enabled:
            return stix_read.stix_create_counts(path)

        kwargs = {"energy_shift": 0, "bkg_poll_function": poll}
        if bkg_file_enabled:
            bkg_filename, bkg_data = session_store.get(sid, "stix_bkg_file_bytes")
            kwargs["pathbkg"] = stack.enter_context(tempfile_from_bytes(bkg_data, bkg_filename))
        if bkg_time_enabled:
            start = parse_dt_input(bkg_start)
            end = parse_dt_input(bkg_end)
            kwargs["stix_bkg_range"] = (to_stix_date_str(start), to_stix_date_str(end))
        return stix_read.stix_remove_bkg_counts(path, **kwargs)


@callback(
    Output("stix-preview-graph", "figure"),
    Output("stix-bkg-start", "value"),
    Output("stix-bkg-end", "value"),
    Output("stix-alert", "children"),
    Output("stix-alert", "color"),
    Output("stix-alert", "is_open"),
    Input("stix-preview-btn", "n_clicks"),
    State("session-id", "data"),
    prevent_initial_call=True,
)
def stix_preview(n_clicks, sid):
    if not n_clicks:
        raise PreventUpdate
    try:
        filename, data = session_store.get(sid, "stix_file_bytes")
        with tempfile_from_bytes(data, filename) as path:
            counts = stix_read.stix_create_counts(path)
        session_store.set(sid, "stix_counts_final", counts)
        fig = plotting.stix_spectrogram_figure(counts)
        start_str = format_dt_input(min(counts["time"]))
        end_str = format_dt_input(max(counts["time"]))
        return fig, start_str, end_str, "", "success", False
    except Exception as exc:  # noqa: BLE001
        logger.exception("Unhandled error in callback")
        return dash.no_update, dash.no_update, dash.no_update, str(exc), "danger", True


@callback(
    Output("stix-preview-graph", "figure", allow_duplicate=True),
    Output("stix-plot-bkg-btn", "disabled"),
    Output("stix-alert", "children", allow_duplicate=True),
    Output("stix-alert", "color", allow_duplicate=True),
    Output("stix-alert", "is_open", allow_duplicate=True),
    Input("stix-preview-bkg-btn", "n_clicks"),
    State("stix-bkg-file-enabled", "value"),
    State("stix-bkg-time-enabled", "value"),
    State("stix-bkg-start", "value"),
    State("stix-bkg-end", "value"),
    State("stix-bkg-poll", "value"),
    State("session-id", "data"),
    prevent_initial_call=True,
)
def stix_preview_with_bkg(n_clicks, bkg_file_enabled, bkg_time_enabled, bkg_start, bkg_end, poll, sid):
    if not n_clicks:
        raise PreventUpdate
    try:
        counts = _compute_stix_counts(sid, bkg_file_enabled, bkg_time_enabled, bkg_start, bkg_end, poll)
        session_store.set(sid, "stix_counts_final", counts)
        date_range = None
        if bkg_time_enabled:
            date_range = (parse_dt_input(bkg_start), parse_dt_input(bkg_end))
        fig = plotting.stix_spectrogram_figure(counts)
        if date_range:
            fig.add_vline(x=date_range[0], line_color="red")
            fig.add_vline(x=date_range[1], line_color="red")
        has_bkg = "background" in counts
        return fig, not has_bkg, "", "success", False
    except Exception as exc:  # noqa: BLE001
        logger.exception("Unhandled error in callback")
        return dash.no_update, dash.no_update, str(exc), "danger", True


@callback(
    Output("stix-bkg-graph", "figure"),
    Output("stix-bkg-modal", "is_open"),
    Input("stix-plot-bkg-btn", "n_clicks"),
    State("session-id", "data"),
    prevent_initial_call=True,
)
def stix_plot_background(n_clicks, sid):
    if not n_clicks:
        raise PreventUpdate
    counts = session_store.get(sid, "stix_counts_final")
    return plotting.stix_bkg_figure(counts), True


@callback(
    Output("instrument-status-store", "data", allow_duplicate=True),
    Output("stix-alert", "children", allow_duplicate=True),
    Output("stix-alert", "color", allow_duplicate=True),
    Output("stix-alert", "is_open", allow_duplicate=True),
    Input("stix-load-btn", "n_clicks"),
    State("stix-bkg-file-enabled", "value"),
    State("stix-bkg-time-enabled", "value"),
    State("stix-bkg-start", "value"),
    State("stix-bkg-end", "value"),
    State("stix-bkg-poll", "value"),
    State("session-id", "data"),
    State("instrument-status-store", "data"),
    prevent_initial_call=True,
)
def stix_load(n_clicks, bkg_file_enabled, bkg_time_enabled, bkg_start, bkg_end, poll, sid, status):
    if not n_clicks:
        raise PreventUpdate
    try:
        counts = _compute_stix_counts(sid, bkg_file_enabled, bkg_time_enabled, bkg_start, bkg_end, poll)
        session_store.set(sid, "stix_counts_final", counts)
        session_store.set(
            sid,
            "stix_meta",
            {
                "bkg_file_enabled": bool(bkg_file_enabled),
                "bkg_time_enabled": bool(bkg_time_enabled),
                "bkg_poll_function": poll,
            },
        )
        status = dict(status or {})
        status["stix"] = {
            "loaded": True,
            "min_time": format_dt_input(min(counts["time"])),
            "max_time": format_dt_input(max(counts["time"])),
            "bkg_enabled": bool(bkg_file_enabled) or bool(bkg_time_enabled),
        }
        return status, "STIX data loaded.", "success", True
    except Exception as exc:  # noqa: BLE001
        logger.exception("Unhandled error in callback")
        return dash.no_update, str(exc), "danger", True
