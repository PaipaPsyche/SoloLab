"""STIX Imaging - dedicated page for STIX pixel-data image
reconstruction (backprojection/CLEAN/MEM_GE via stixpy/xrayvision).

Originally built as a 4th "plot type" inside Plot Preferences' STIX tab;
moved to its own page + sidebar section because it's a fundamentally
different workflow (a pixel-data-specific pipeline with its own multi-step
parameter set: time/energy interval -> algorithm -> run -> view a
reconstructed image) rather than a plot-style toggle over already-final
counts data like the other three STIX plot types.

Reuses whatever pixel-data file is already loaded via /import/stix
(session_store's "stix_file_bytes"/"stix_bkg_file_bytes"/"stix_counts_final")
rather than duplicating the upload/download UI - see
instrument-status-store["stix"]["pixel_data"], set by import_stix.py's
stix_load, which gates the "Flare location and visibility" button here.

Two-stage flow (see sololab.stix_imaging's module docstring for the "why"):
1. "Flare location and visibility" - runs stix_imaging.
   estimate_flare_location_and_ancillary once per loaded file (a real
   coarse-imaging pass, the expensive part), caches the result in
   session_store as "stix_imaging_flare_estimate", displays ancillary
   observation info, and auto-fills the time-range pickers to the file's
   own start/end.
2. "Run STIX Imaging"/"Run Sequence" - reuses the cached flare location for
   every image (single interval or each Sequence tile), only requires the
   estimate to exist, not to be re-run.

EUI background overlay is a stated placeholder only (disabled checkbox) -
not implemented in this pass. See sololab.stix_imaging's module docstring
for the reconstruction pipeline itself, including confirmed upstream
stixpy 0.3.0 bugs that currently block reconstruction from completing.
"""
import contextlib
import logging

import astropy.units as u
import dash
import dash_bootstrap_components as dbc
from astropy.time import Time
from dash import Input, Output, State, callback, dcc, html
from dash.exceptions import PreventUpdate

from sololab import stix_imaging as stix_imaging_core
from sololab.dash_app import plotting
from sololab.dash_app.constants import STIX_IMAGING_ALGORITHM_DEFAULT, STIX_IMAGING_ALGORITHMS
from sololab.dash_app.session_store import session_store
from sololab.dash_app.utils import (
    format_dt_input,
    make_list_editor_modal,
    parse_dt_input,
    register_list_editor_add_row_callback,
    tempfile_from_bytes,
)

logger = logging.getLogger(__name__)

dash.register_page(__name__, path="/imaging/stix", name="STIX Imaging")


def _energy_ranges_modal():
    return make_list_editor_modal(
        "stix-imaging-energy-ranges",
        "STIX Imaging Energy Ranges (keV)",
        [{"name": "Min", "id": "min", "type": "numeric"}, {"name": "Max", "id": "max", "type": "numeric"}],
    )


register_list_editor_add_row_callback("stix-imaging-energy-ranges", {"min": 4, "max": 10})


def _contour_levels_modal():
    return make_list_editor_modal(
        "stix-imaging-contour-levels",
        "STIX Imaging Contour Levels (% of peak)",
        [{"name": "Percent", "id": "pct", "type": "numeric"}],
    )


register_list_editor_add_row_callback("stix-imaging-contour-levels", {"pct": 50})


# Default/reference energy ranges - both the initial value of the
# user-editable imaging energy-ranges list and the fixed bands the
# always-visible lightcurve panel plots (kept identical per the user's
# request that the lightcurve reflect the same ranges imaging uses by
# default).
DEFAULT_ENERGY_RANGES = [(6, 10), (12, 25), (32, 60)]

# Default contour levels, as percent of each image's own peak value.
DEFAULT_CONTOUR_LEVELS_PCT = [50, 75, 90]


def layout(**kwargs):
    return dbc.Container(
        [
            html.H3("STIX Imaging", className="mt-3"),
            _energy_ranges_modal(),
            _contour_levels_modal(),
            dbc.Modal(
                [
                    dbc.ModalHeader(dbc.ModalTitle("Flare location")),
                    dbc.ModalBody(dcc.Graph(id="stix-imaging-location-graph")),
                ],
                id="stix-imaging-location-modal",
                is_open=False,
                size="lg",
            ),
            dcc.Store(id="stix-imaging-energy-ranges-store", data=[list(r) for r in DEFAULT_ENERGY_RANGES]),
            dcc.Store(id="stix-imaging-contour-levels-store", data=list(DEFAULT_CONTOUR_LEVELS_PCT)),
            dcc.Store(id="stix-imaging-mosaic-figure-store"),
            dcc.Store(id="stix-imaging-download-dummy"),
            dcc.Store(id="stix-imaging-download-all-dummy"),
            dbc.Alert(id="stix-imaging-alert", is_open=False, dismissable=True, className="mt-2"),
            dbc.Row(
                [
                    dbc.Col(_controls_card(), md=4),
                    dbc.Col(
                        [
                            dbc.Spinner(dcc.Graph(id="stix-imaging-graph", style={"height": "500px"}), color="success"),
                            html.Div(
                                dbc.Progress(id="stix-imaging-progress", value=0, striped=True, animated=True),
                                id="stix-imaging-progress-row",
                                className="mb-2",
                                style={"display": "none"},
                            ),
                            dbc.Row(
                                [
                                    dbc.Col(dbc.Button("◀", id="stix-imaging-prev-btn", color="secondary", outline=True, size="sm"), width="auto"),
                                    dbc.Col(dcc.Slider(id="stix-imaging-tile-slider", min=0, max=0, step=1, value=0, marks={})),
                                    dbc.Col(dbc.Button("▶", id="stix-imaging-next-btn", color="secondary", outline=True, size="sm"), width="auto"),
                                ],
                                id="stix-imaging-tile-slider-row",
                                className="mb-2 align-items-center g-2",
                                style={"display": "none"},
                            ),
                            dbc.Row(
                                [
                                    dbc.Col(dbc.Button("▶ Play", id="stix-imaging-play-btn", color="secondary", outline=True, size="sm"), width="auto"),
                                    dbc.Col(dbc.Input(id="stix-imaging-play-seconds", type="number", min=0.1, step=0.1, value=0.5), width=3),
                                    dbc.Col(html.Span("sec/frame", className="text-muted small"), width="auto"),
                                ],
                                id="stix-imaging-play-row",
                                className="mb-2 align-items-center g-2",
                                style={"display": "none"},
                            ),
                            dcc.Interval(id="stix-imaging-animate-interval", interval=500, disabled=True, n_intervals=0),
                            html.Div(
                                dbc.RadioItems(
                                    id="stix-imaging-sequence-view",
                                    options=[{"label": "Single", "value": "single"}, {"label": "Mosaic", "value": "mosaic"}],
                                    value="single",
                                    inline=True,
                                ),
                                id="stix-imaging-sequence-view-row",
                                className="mb-2",
                                style={"display": "none"},
                            ),
                            dbc.Row(
                                [
                                    dbc.Col(dbc.Button("Download Image", id="stix-imaging-download-btn", color="secondary", outline=True, size="sm"), width="auto"),
                                    dbc.Col(dbc.Button("Download All", id="stix-imaging-download-all-btn", color="secondary", outline=True, size="sm"), width="auto"),
                                ],
                                id="stix-imaging-download-row",
                                className="mb-2 g-2",
                                style={"display": "none"},
                            ),
                            html.Div("STIX lightcurve (imaging interval highlighted)", className="text-muted small mt-2 mb-1"),
                            dcc.Graph(id="stix-imaging-context-graph", style={"height": "138px"}),
                        ],
                        md=8,
                    ),
                ]
            ),
        ],
        className="pb-5",
    )


def _time_picker_row(prefix, label):
    """One row: a date picker + a single "HH:MM:SS" clock text field. STIX
    observations are frequently under a minute (the bundled sample file is
    20 seconds), so sub-minute resolution matters, hence seconds are part
    of the clock field rather than an hour-only picker. Combined into the
    single `YYYY-MM-DD HH:MM:SS` string the rest of the pipeline already
    expects by a small callback below, written into a hidden
    stix-imaging-time-{start,end} input so no downstream code
    (run_stix_imaging, update_stix_imaging_highlight) needs to change."""
    return dbc.Row(
        [
            dbc.Col(html.Label(label), width=12, className="small text-muted"),
            dbc.Col(dcc.DatePickerSingle(id=f"stix-imaging-time-{prefix}-date", display_format="YYYY-MM-DD"), width=6),
            dbc.Col(dbc.Input(id=f"stix-imaging-time-{prefix}-clock", type="text", placeholder="HH:MM:SS", maxLength=8), width=6),
        ],
        className="mb-1 g-1",
    )


def _controls_card():
    return dbc.Card(
        dbc.CardBody(
            [
                html.Div(id="stix-imaging-filename", className="text-muted small mb-2"),
                dbc.Spinner(
                    dbc.Button(
                        "Flare location and visibility", id="stix-imaging-estimate-btn",
                        color="secondary", disabled=True, className="w-100 mb-2",
                    ),
                    color="secondary",
                ),
                html.Div(id="stix-imaging-ancillary", className="small mb-2", style={"display": "none"}),
                dbc.Button(
                    "Plot flare location", id="stix-imaging-plot-location-btn",
                    color="secondary", outline=True, size="sm", disabled=True, className="w-100 mb-2",
                ),
                html.Hr(),
                _time_picker_row("start", "Start"),
                _time_picker_row("end", "End"),
                # Hidden combined "YYYY-MM-DD HH:MM:SS" strings - every
                # downstream consumer (run_stix_imaging,
                # update_stix_imaging_highlight) reads these exactly as
                # before; only how they get filled in has changed.
                dbc.Input(id="stix-imaging-time-start", type="hidden", value=""),
                dbc.Input(id="stix-imaging-time-end", type="hidden", value=""),
                html.Div(
                    [
                        html.Label("Energy ranges (keV)"),
                        html.Div(
                            dbc.Button(
                                "Set Energy Ranges", id="stix-imaging-set-energy-ranges-btn",
                                color="secondary", outline=True, size="sm",
                            )
                        ),
                        html.Div(id="stix-imaging-energy-ranges-summary", className="text-muted small mt-1"),
                    ],
                    className="mb-2",
                ),
                dbc.RadioItems(
                    id="stix-imaging-type",
                    options=[{"label": "Single interval", "value": "single"}, {"label": "Sequence", "value": "sequence"}],
                    value="single",
                    inline=True,
                    className="mb-2",
                ),
                html.Div(
                    [
                        html.Label("Tile duration (s)"),
                        dbc.Input(id="stix-imaging-tile-duration", type="number", min=1, value=30),
                        html.Div(id="stix-imaging-tile-count", className="text-muted small mt-1"),
                    ],
                    id="stix-imaging-tile-duration-row",
                    className="mb-2",
                    style={"display": "none"},
                ),
                html.Label("Algorithm"),
                dcc.Dropdown(
                    id="stix-imaging-algorithm",
                    options=STIX_IMAGING_ALGORITHMS,
                    value=STIX_IMAGING_ALGORITHM_DEFAULT,
                    clearable=False,
                    className="mb-2",
                ),
                dbc.Row(
                    [
                        dbc.Col([html.Label("Image size (px)"), dbc.Input(id="stix-imaging-npix", type="number", min=8, value=128)]),
                        dbc.Col([html.Label("Pixel size (arcsec)"), dbc.Input(id="stix-imaging-pixel-size", type="number", min=0.1, value=2.0)]),
                    ],
                    className="mb-2",
                ),
                html.Div(
                    dcc.Dropdown(
                        id="stix-imaging-weighting",
                        options=["natural", "uniform"],
                        value="natural",
                        clearable=False,
                    ),
                    id="stix-imaging-backprojection-row",
                    className="mb-2",
                ),
                html.Div(
                    dbc.Row(
                        [
                            dbc.Col([html.Label("Gain"), dbc.Input(id="stix-imaging-gain", type="number", min=0.01, max=1, step=0.01, value=0.1)]),
                            dbc.Col([html.Label("Iterations"), dbc.Input(id="stix-imaging-clean-niter", type="number", min=1, value=200)]),
                            dbc.Col([html.Label("Beam (arcsec)"), dbc.Input(id="stix-imaging-clean-beam", type="number", min=0, value=20)]),
                        ]
                    ),
                    id="stix-imaging-clean-row",
                    className="mb-2",
                    style={"display": "none"},
                ),
                html.Div(
                    [
                        html.Label(
                            [
                                "% lambda (blank = auto) ",
                                html.Span("ⓘ", id="stix-imaging-lambda-info", style={"cursor": "help", "color": "#6c757d"}),
                            ]
                        ),
                        dbc.Tooltip(
                            "Blank = auto-estimated from data SNR (2/(snr²+90), typically ~0.001-0.05, "
                            "matching stixpy's own example). Valid range: 0.0001-0.2. Lower = smoother/less "
                            "noisy but less detail; higher = sharper but noisier.",
                            target="stix-imaging-lambda-info",
                            placement="right",
                        ),
                        dbc.Input(id="stix-imaging-percent-lambda", type="number", min=0.0001, max=0.2, step=0.0001),
                    ],
                    id="stix-imaging-mem-row",
                    className="mb-2",
                    style={"display": "none"},
                ),
                dbc.Row(
                    [
                        dbc.Col([html.Label("Flare Tx override (arcsec)"), dbc.Input(id="stix-imaging-flare-x", type="number", placeholder="auto")]),
                        dbc.Col([html.Label("Flare Ty override (arcsec)"), dbc.Input(id="stix-imaging-flare-y", type="number", placeholder="auto")]),
                    ],
                    className="mb-2",
                ),
                dbc.RadioItems(
                    id="stix-imaging-display-mode",
                    # Default 3 energy ranges (see DEFAULT_ENERGY_RANGES) means
                    # Heatmap starts disabled - matches
                    # toggle_stix_imaging_heatmap_option's reactive logic, just
                    # set once here for the initial page load (that callback
                    # is prevent_initial_call=True, so it doesn't run then).
                    options=[{"label": "Heatmap", "value": "heatmap", "disabled": True}, {"label": "Contour", "value": "contour"}],
                    value="contour",
                    inline=True,
                    className="mb-2",
                ),
                html.Div(
                    [
                        html.Div(
                            dbc.Button(
                                "Set Contour Levels", id="stix-imaging-set-contour-levels-btn",
                                color="secondary", outline=True, size="sm",
                            )
                        ),
                        html.Div(id="stix-imaging-contour-levels-summary", className="text-muted small mt-1"),
                    ],
                    className="mb-2",
                ),
                dbc.Button("Run STIX Imaging", id="stix-run-imaging-btn", color="success", disabled=True, className="w-100 mb-3"),
                html.Hr(),
                dbc.Checkbox(
                    id="stix-imaging-eui-overlay-enabled",
                    label="Enable EUI overlay (coming soon)",
                    value=False,
                    disabled=True,
                ),
            ]
        ),
        className="mt-2",
    )


def _combine_date_clock(date_str, clock_str):
    if not date_str:
        return ""
    parts = (clock_str or "").split(":")
    nums = [int(p) if p.strip().isdigit() else 0 for p in parts[:3]]
    nums += [0] * (3 - len(nums))
    h, m, s = (max(0, min(limit, n)) for n, limit in zip(nums, (23, 59, 59)))
    return f"{date_str} {h:02d}:{m:02d}:{s:02d}"


@callback(
    Output("stix-imaging-time-start", "value"),
    Input("stix-imaging-time-start-date", "date"),
    Input("stix-imaging-time-start-clock", "value"),
)
def combine_stix_imaging_start_time(date_str, clock_str):
    return _combine_date_clock(date_str, clock_str)


@callback(
    Output("stix-imaging-time-end", "value"),
    Input("stix-imaging-time-end-date", "date"),
    Input("stix-imaging-time-end-clock", "value"),
)
def combine_stix_imaging_end_time(date_str, clock_str):
    return _combine_date_clock(date_str, clock_str)


@callback(
    Output("stix-imaging-filename", "children"),
    Output("stix-imaging-estimate-btn", "disabled"),
    Output("stix-imaging-estimate-btn", "color", allow_duplicate=True),
    Output("stix-run-imaging-btn", "disabled"),
    Output("stix-imaging-plot-location-btn", "disabled", allow_duplicate=True),
    Input("instrument-status-store", "data"),
    State("session-id", "data"),
    prevent_initial_call="initial_duplicate",
)
def show_stix_imaging_filename(status, sid):
    """Also re-gates the Run button: a cached flare estimate is only valid
    for the exact file it was computed from (see
    estimate_stix_imaging_flare_location's `_source_filename` tag) - if a
    *different* file has since been loaded, Run must go back to disabled
    until the user re-estimates for the new file, even though this
    callback fires on every instrument-status-store change (RPW/EPD
    included), not just STIX ones. Deliberately the *primary* (non-
    duplicate) writer of stix-run-imaging-btn.disabled. `prevent_initial_call
    ="initial_duplicate"` (rather than plain False) is required because the
    estimate button's `color` Output is also written by
    estimate_stix_imaging_flare_location - but this callback must still fire
    on initial page load, so navigating straight to this page after already
    loading a file elsewhere in the session shows the correct state
    immediately rather than the layout's static disabled=True default.
    Also highlights the estimate button (primary/blue) whenever a pixel file
    is loaded but hasn't been estimated for yet - a visual nudge for the
    required first step."""
    if not sid or not (status or {}).get("stix", {}).get("pixel_data"):
        return "Load a STIX pixel-data file via Import STIX first.", True, "secondary", True, True
    file_entry = session_store.get(sid, "stix_file_bytes")
    filename = file_entry[0] if file_entry else "(unknown file)"
    estimate = session_store.get(sid, "stix_imaging_flare_estimate")
    needs_estimate = not (estimate and estimate.get("_source_filename") == filename)
    color = "primary" if needs_estimate else "secondary"
    return f"Loaded: {filename}", False, color, needs_estimate, needs_estimate


@callback(
    Output("stix-imaging-alert", "children"),
    Output("stix-imaging-alert", "color"),
    Output("stix-imaging-alert", "is_open"),
    Output("stix-imaging-ancillary", "children"),
    Output("stix-imaging-ancillary", "style"),
    Output("stix-run-imaging-btn", "disabled", allow_duplicate=True),
    Output("stix-imaging-estimate-btn", "color", allow_duplicate=True),
    Output("stix-imaging-plot-location-btn", "disabled", allow_duplicate=True),
    Output("stix-imaging-time-start-date", "date"),
    Output("stix-imaging-time-start-clock", "value"),
    Output("stix-imaging-time-end-date", "date"),
    Output("stix-imaging-time-end-clock", "value"),
    Input("stix-imaging-estimate-btn", "n_clicks"),
    State("session-id", "data"),
    prevent_initial_call=True,
)
def estimate_stix_imaging_flare_location(n_clicks, sid):
    """Runs the expensive coarse-imaging flare-location pass once and
    caches it - see stix_imaging.estimate_flare_location_and_ancillary.
    Auto-fills the time pickers to the file's own observation window so
    the user isn't left typing timestamps by hand."""
    if not n_clicks:
        raise PreventUpdate
    no_fill = (dash.no_update,) * 4
    try:
        file_entry = session_store.get(sid, "stix_file_bytes")
        if file_entry is None:
            raise ValueError("No STIX pixel-data file loaded. Import STIX data first.")
        filename, data = file_entry
        with tempfile_from_bytes(data, filename) as pixel_path:
            estimate = stix_imaging_core.estimate_flare_location_and_ancillary(pixel_path)
        estimate["_source_filename"] = filename  # ties the cache to this exact file, see show_stix_imaging_filename
        session_store.set(sid, "stix_imaging_flare_estimate", estimate)

        obs_start, obs_end = estimate["obs_start"], estimate["obs_end"]
        ancillary = html.Ul(
            [
                html.Li(f"SolO-Sun distance: {estimate['sun_distance_au']:.3f} AU"),
                html.Li(f"SolO-Earth distance: {estimate['earth_distance_au']:.3f} AU"),
                html.Li(f"Flare location: Tx={estimate['flare_tx_arcsec']:.1f}\", Ty={estimate['flare_ty_arcsec']:.1f}\""),
                html.Li(f"Observation: {format_dt_input(obs_start.datetime)} to {format_dt_input(obs_end.datetime)}"),
            ],
            className="mb-0",
        )
        return (
            "Flare location and visibility estimated.", "success", True,
            ancillary, {"display": "block"},
            False,
            "secondary",
            False,
            obs_start.datetime.date().isoformat(),
            f"{obs_start.datetime.hour:02d}:{obs_start.datetime.minute:02d}:{obs_start.datetime.second:02d}",
            obs_end.datetime.date().isoformat(),
            f"{obs_end.datetime.hour:02d}:{obs_end.datetime.minute:02d}:{obs_end.datetime.second:02d}",
        )
    except Exception as exc:  # noqa: BLE001
        logger.exception("Unhandled error in callback")
        return str(exc), "danger", True, dash.no_update, dash.no_update, True, dash.no_update, dash.no_update, *no_fill


@callback(
    Output("stix-imaging-location-graph", "figure"),
    Output("stix-imaging-location-modal", "is_open"),
    Output("stix-imaging-alert", "children", allow_duplicate=True),
    Output("stix-imaging-alert", "color", allow_duplicate=True),
    Output("stix-imaging-alert", "is_open", allow_duplicate=True),
    Input("stix-imaging-plot-location-btn", "n_clicks"),
    State("stix-imaging-flare-x", "value"),
    State("stix-imaging-flare-y", "value"),
    State("session-id", "data"),
    prevent_initial_call=True,
)
def plot_stix_imaging_flare_location(n_clicks, flare_x, flare_y, sid):
    """Flare position (the cached estimate, or the Tx/Ty override if set) on
    the solar disk as seen from Solar Orbiter, in a popup like the
    background plots."""
    if not n_clicks:
        raise PreventUpdate
    try:
        estimate = session_store.get(sid, "stix_imaging_flare_estimate")
        if estimate is None:
            raise ValueError('Click "Flare location and visibility" first.')
        flare_location = estimate["flare_location"]
        if flare_x or flare_y:
            frame = flare_location.frame
            flare_location = stix_imaging_core.build_flare_location(
                (frame.obstime, frame.obstime_end), flare_x or 0, flare_y or 0
            )
        info = stix_imaging_core.flare_location_on_disk(flare_location)
        return plotting.stix_flare_location_figure(info), True, dash.no_update, dash.no_update, dash.no_update
    except Exception as exc:  # noqa: BLE001
        logger.exception("Unhandled error in callback")
        return dash.no_update, dash.no_update, str(exc), "danger", True


@callback(
    Output("stix-imaging-tile-duration-row", "style"),
    Input("stix-imaging-type", "value"),
)
def toggle_stix_imaging_type_row(imaging_type):
    show, hide = {"display": "block"}, {"display": "none"}
    return show if imaging_type == "sequence" else hide


@callback(
    Output("stix-imaging-backprojection-row", "style"),
    Output("stix-imaging-clean-row", "style"),
    Output("stix-imaging-mem-row", "style"),
    Input("stix-imaging-algorithm", "value"),
)
def toggle_stix_imaging_algorithm_params(algorithm):
    show, hide = {"display": "block"}, {"display": "none"}
    return (
        show if algorithm == "backprojection" else hide,
        show if algorithm == "clean" else hide,
        show if algorithm == "mem_ge" else hide,
    )


@callback(
    Output("stix-imaging-energy-ranges-summary", "children"),
    Input("stix-imaging-energy-ranges-store", "data"),
)
def show_stix_imaging_energy_ranges_summary(ranges):
    if not ranges:
        return "No energy ranges set."
    return ", ".join(f"{lo:.0f}-{hi:.0f}" for lo, hi in ranges) + " keV"


@callback(
    Output("stix-imaging-energy-ranges-modal", "is_open", allow_duplicate=True),
    Output("stix-imaging-energy-ranges-table", "data", allow_duplicate=True),
    Input("stix-imaging-set-energy-ranges-btn", "n_clicks"),
    State("stix-imaging-energy-ranges-store", "data"),
    prevent_initial_call=True,
)
def open_stix_imaging_energy_ranges(n_clicks, ranges):
    if not n_clicks:
        raise PreventUpdate
    return True, [{"min": lo, "max": hi} for lo, hi in (ranges or [])]


@callback(
    Output("stix-imaging-energy-ranges-modal", "is_open", allow_duplicate=True),
    Output("stix-imaging-energy-ranges-store", "data", allow_duplicate=True),
    Output("stix-imaging-energy-ranges-error", "children"),
    Input("stix-imaging-energy-ranges-done-btn", "n_clicks"),
    State("stix-imaging-energy-ranges-table", "data"),
    prevent_initial_call=True,
)
def close_stix_imaging_energy_ranges(n_clicks, rows):
    if not n_clicks:
        raise PreventUpdate
    ranges = []
    for row in rows or []:
        try:
            lo, hi = float(row["min"]), float(row["max"])
        except (KeyError, TypeError, ValueError):
            continue
        if 4 <= lo < hi <= 150:
            ranges.append([lo, hi])
    if not ranges:
        return dash.no_update, dash.no_update, "At least one valid range (4 <= min < max <= 150) is required."
    return False, ranges, ""


@callback(
    Output("stix-imaging-contour-levels-summary", "children"),
    Input("stix-imaging-contour-levels-store", "data"),
)
def show_stix_imaging_contour_levels_summary(levels):
    if not levels:
        return "No contour levels set."
    return ", ".join(f"{p:.0f}" for p in levels) + " %"


@callback(
    Output("stix-imaging-contour-levels-modal", "is_open", allow_duplicate=True),
    Output("stix-imaging-contour-levels-table", "data", allow_duplicate=True),
    Input("stix-imaging-set-contour-levels-btn", "n_clicks"),
    State("stix-imaging-contour-levels-store", "data"),
    prevent_initial_call=True,
)
def open_stix_imaging_contour_levels(n_clicks, levels):
    if not n_clicks:
        raise PreventUpdate
    return True, [{"pct": p} for p in (levels or [])]


@callback(
    Output("stix-imaging-contour-levels-modal", "is_open", allow_duplicate=True),
    Output("stix-imaging-contour-levels-store", "data", allow_duplicate=True),
    Output("stix-imaging-contour-levels-error", "children"),
    Input("stix-imaging-contour-levels-done-btn", "n_clicks"),
    State("stix-imaging-contour-levels-table", "data"),
    prevent_initial_call=True,
)
def close_stix_imaging_contour_levels(n_clicks, rows):
    if not n_clicks:
        raise PreventUpdate
    levels = sorted({float(row["pct"]) for row in (rows or []) if _valid_pct(row.get("pct"))})
    if not levels:
        return dash.no_update, dash.no_update, "At least one valid level (0 < percent <= 100) is required."
    return False, levels, ""


def _valid_pct(pct):
    try:
        return 0 < float(pct) <= 100
    except (TypeError, ValueError):
        return False


@callback(
    Output("stix-imaging-display-mode", "options"),
    Output("stix-imaging-display-mode", "value", allow_duplicate=True),
    Input("stix-imaging-energy-ranges-store", "data"),
    State("stix-imaging-display-mode", "value"),
    prevent_initial_call=True,
)
def toggle_stix_imaging_heatmap_option(ranges, current_mode):
    """Heatmap only makes sense for a single energy range - contour is the
    only mode that can overlay several at once."""
    multi = len(ranges or []) > 1
    options = [
        {"label": "Heatmap", "value": "heatmap", "disabled": multi},
        {"label": "Contour", "value": "contour"},
    ]
    value = "contour" if multi and current_mode == "heatmap" else dash.no_update
    return options, value


def _compute_tiles(time_start, time_end, imaging_type, tile_duration):
    """Shared by the highlight redraw and the tile-count preview - both
    need the exact same tile list, computed the same way."""
    if not (time_start and time_end):
        return None
    start, end = parse_dt_input(time_start), parse_dt_input(time_end)
    if not (start and end and end > start):
        return None
    if imaging_type != "sequence" or not tile_duration:
        return [(start, end)]
    tiles = stix_imaging_core.tile_time_range(Time(start), Time(end), float(tile_duration) * u.s)
    return [(t0.datetime, t1.datetime) for t0, t1 in tiles]


# Fixed reference bands for the always-visible lightcurve panel - not the
# user's imaging energy range(s), just a general-purpose "what does the
# data look like" view (same role stix_counts_figure already serves
# elsewhere in the app - see Plot Preferences' "time profiles"). Kept in
# sync with DEFAULT_ENERGY_RANGES per the user's request.
LIGHTCURVE_ENERGY_BANDS = DEFAULT_ENERGY_RANGES


@callback(
    Output("stix-imaging-context-graph", "figure"),
    Input("instrument-status-store", "data"),
    State("session-id", "data"),
)
def render_stix_imaging_lightcurve(status, sid):
    """Plain lightcurve only - no spectrogram heatmap, no interval
    highlighting. Simplified deliberately: the earlier highlighted-overlay
    version wasn't rendering reliably for the user, so this isolates the
    one thing that has to work (the base lightcurve) from the
    interval-highlighting logic, which is dropped here rather than
    re-debugged blind. Fires once per file load, not on every keystroke -
    it no longer depends on the time-range/sequence controls at all."""
    if not sid or not (status or {}).get("stix", {}).get("pixel_data"):
        raise PreventUpdate
    counts = session_store.get(sid, "stix_counts_final")
    if counts is None:
        raise PreventUpdate
    return plotting.stix_counts_figure(counts, integrate_bins=LIGHTCURVE_ENERGY_BANDS, height=138)


@callback(
    Output("stix-imaging-tile-count", "children"),
    Input("stix-imaging-time-start", "value"),
    Input("stix-imaging-time-end", "value"),
    Input("stix-imaging-type", "value"),
    Input("stix-imaging-tile-duration", "value"),
    Input("stix-imaging-energy-ranges-store", "data"),
)
def update_stix_imaging_tile_count(time_start, time_end, imaging_type, tile_duration, energy_ranges):
    """Sequence-mode tile-count preview - split out from the (now simpler)
    lightcurve render above, since this one still needs to react to every
    time/duration keystroke."""
    tiles = _compute_tiles(time_start, time_end, imaging_type, tile_duration)
    n_ranges = len(energy_ranges or []) or 1
    if imaging_type == "sequence" and tiles:
        n_tiles = len(tiles)
        return f"{n_tiles} tile(s) x {n_ranges} energy range(s) = {n_tiles * n_ranges} image(s)"
    return ""


@callback(
    Output("stix-imaging-tile-slider", "value", allow_duplicate=True),
    Input("stix-imaging-prev-btn", "n_clicks"),
    State("stix-imaging-tile-slider", "value"),
    prevent_initial_call=True,
)
def stix_imaging_prev_tile(n_clicks, value):
    if not n_clicks:
        raise PreventUpdate
    return max(0, (value or 0) - 1)


@callback(
    Output("stix-imaging-tile-slider", "value", allow_duplicate=True),
    Input("stix-imaging-next-btn", "n_clicks"),
    State("stix-imaging-tile-slider", "value"),
    State("stix-imaging-tile-slider", "max"),
    prevent_initial_call=True,
)
def stix_imaging_next_tile(n_clicks, value, max_value):
    if not n_clicks:
        raise PreventUpdate
    return min(max_value or 0, (value or 0) + 1)


@callback(
    Output("stix-imaging-animate-interval", "disabled", allow_duplicate=True),
    Output("stix-imaging-play-btn", "children", allow_duplicate=True),
    Input("stix-imaging-play-btn", "n_clicks"),
    State("stix-imaging-animate-interval", "disabled"),
    prevent_initial_call=True,
)
def toggle_stix_imaging_play(n_clicks, disabled):
    if not n_clicks:
        raise PreventUpdate
    now_playing = disabled  # was disabled -> about to enable -> now playing
    return (not disabled), ("⏸ Pause" if now_playing else "▶ Play")


@callback(
    Output("stix-imaging-animate-interval", "disabled", allow_duplicate=True),
    Output("stix-imaging-play-btn", "children", allow_duplicate=True),
    Input("stix-imaging-sequence-view", "value"),
    prevent_initial_call=True,
)
def pause_stix_imaging_play_on_mosaic(view):
    """Switching to Mosaic shows every tile at once - a per-tile animation
    no longer makes sense, so stop it rather than leaving it silently
    running behind the mosaic view."""
    if view != "mosaic":
        raise PreventUpdate
    return True, "▶ Play"


@callback(
    Output("stix-imaging-animate-interval", "interval"),
    Input("stix-imaging-play-seconds", "value"),
)
def set_stix_imaging_play_interval(seconds):
    seconds = seconds or 0.5
    return max(100, int(float(seconds) * 1000))


@callback(
    Output("stix-imaging-tile-slider", "value", allow_duplicate=True),
    Input("stix-imaging-animate-interval", "n_intervals"),
    State("stix-imaging-tile-slider", "value"),
    State("stix-imaging-tile-slider", "max"),
    prevent_initial_call=True,
)
def advance_stix_imaging_animation(n_intervals, value, max_value):
    """Loops back to the first tile after the last one - a looping viewer,
    not one-shot playback."""
    if not max_value:
        raise PreventUpdate
    return ((value or 0) + 1) % (max_value + 1)


def _contour_level_fractions(levels_pct):
    return [p / 100.0 for p in (levels_pct or DEFAULT_CONTOUR_LEVELS_PCT)]


@callback(
    Output("stix-imaging-graph", "figure", allow_duplicate=True),
    Output("stix-imaging-alert", "children", allow_duplicate=True),
    Output("stix-imaging-alert", "color", allow_duplicate=True),
    Output("stix-imaging-alert", "is_open", allow_duplicate=True),
    Output("stix-imaging-tile-slider", "max"),
    Output("stix-imaging-tile-slider", "marks"),
    Output("stix-imaging-tile-slider", "value"),
    Output("stix-imaging-tile-slider-row", "style"),
    Output("stix-imaging-play-row", "style", allow_duplicate=True),
    Output("stix-imaging-sequence-view-row", "style"),
    Output("stix-imaging-sequence-view", "value"),
    Output("stix-imaging-download-row", "style"),
    Output("stix-imaging-download-all-btn", "style", allow_duplicate=True),
    Input("stix-run-imaging-btn", "n_clicks"),
    State("stix-imaging-time-start", "value"),
    State("stix-imaging-time-end", "value"),
    State("stix-imaging-type", "value"),
    State("stix-imaging-tile-duration", "value"),
    State("stix-imaging-algorithm", "value"),
    State("stix-imaging-npix", "value"),
    State("stix-imaging-pixel-size", "value"),
    State("stix-imaging-weighting", "value"),
    State("stix-imaging-gain", "value"),
    State("stix-imaging-clean-niter", "value"),
    State("stix-imaging-clean-beam", "value"),
    State("stix-imaging-percent-lambda", "value"),
    State("stix-imaging-flare-x", "value"),
    State("stix-imaging-flare-y", "value"),
    State("stix-imaging-display-mode", "value"),
    State("stix-imaging-energy-ranges-store", "data"),
    State("stix-imaging-contour-levels-store", "data"),
    State("session-id", "data"),
    background=True,
    progress=[
        Output("stix-imaging-progress", "value"),
        Output("stix-imaging-progress", "max"),
        Output("stix-imaging-progress", "label"),
    ],
    running=[(Output("stix-imaging-progress-row", "style"), {"display": "block"}, {"display": "none"})],
    prevent_initial_call=True,
)
def run_stix_imaging(
    set_progress,
    n_clicks, time_start, time_end, imaging_type, tile_duration, algorithm,
    npix, pixel_size, weighting, gain, clean_niter, clean_beam, percent_lambda,
    flare_x, flare_y, display_mode, energy_ranges, contour_levels_pct, sid,
):
    if not n_clicks:
        raise PreventUpdate
    hide = {"display": "none"}
    err = (dash.no_update,) * 9
    levels = _contour_level_fractions(contour_levels_pct)
    try:
        start, end = parse_dt_input(time_start), parse_dt_input(time_end)
        if not start or not end or end <= start:
            raise ValueError("Set a valid imaging time range (start before end) first.")
        energy_ranges = [(float(lo), float(hi)) for lo, hi in (energy_ranges or [])]
        if not energy_ranges:
            raise ValueError("Set at least one energy range first.")
        if display_mode == "heatmap" and len(energy_ranges) > 1:
            raise ValueError("Heatmap only supports a single energy range - switch to Contour or remove ranges.")

        estimate = session_store.get(sid, "stix_imaging_flare_estimate")
        if estimate is None:
            raise ValueError('Click "Flare location and visibility" first.')
        flare_location = estimate["flare_location"]
        if flare_x or flare_y:
            # Manual override - build a fresh STIXImaging SkyCoord at the
            # given interval's own time range rather than the estimate's.
            flare_location = stix_imaging_core.build_flare_location((start, end), flare_x or 0, flare_y or 0)

        algo_params = {"npix": npix or 128, "pixel_size": pixel_size or 2.0}
        if algorithm == "backprojection":
            algo_params["weighting"] = weighting or "natural"
        elif algorithm == "clean":
            algo_params["gain"] = gain or 0.1
            algo_params["niter"] = clean_niter or 200
            algo_params["clean_beam_width"] = clean_beam if clean_beam is not None else 20.0
        elif algorithm == "mem_ge":
            algo_params["percent_lambda"] = percent_lambda  # None -> SNR-based auto default

        file_entry = session_store.get(sid, "stix_file_bytes")
        if file_entry is None:
            raise ValueError("No STIX pixel-data file loaded. Import STIX data first.")
        filename, data = file_entry
        bkg_entry = session_store.get(sid, "stix_bkg_file_bytes")

        with contextlib.ExitStack() as stack:
            pixel_path = stack.enter_context(tempfile_from_bytes(data, filename))
            bkg_path = None
            if bkg_entry:
                bkg_filename, bkg_data = bkg_entry
                bkg_path = stack.enter_context(tempfile_from_bytes(bkg_data, bkg_filename))

            if imaging_type == "sequence":
                tiles = stix_imaging_core.tile_time_range(Time(start), Time(end), float(tile_duration or 30) * u.s)
                n_tiles = len(tiles)
                n_ranges = len(energy_ranges)
                total = n_tiles * n_ranges
                results_per_tile = []
                count = 0
                set_progress((0, total, f"0/{total}"))
                for t0, t1 in tiles:
                    tile_results = []
                    for e_lo, e_hi in energy_ranges:
                        tile_results.append(
                            stix_imaging_core.reconstruct_stix_image(
                                pixel_path, bkg_path, (t0, t1), (e_lo, e_hi), algorithm, algo_params, flare_location,
                            )
                        )
                        count += 1
                        set_progress((count, total, f"{count}/{total}"))
                    results_per_tile.append(tile_results)
                session_store.set(sid, "stix_imaging_result", results_per_tile)
                marks = {
                    i: format_dt_input(tile_results[0]["time_range"][0].datetime)[-8:]
                    for i, tile_results in enumerate(results_per_tile)
                }
                fig = plotting.stix_imaging_figure(results_per_tile[0], mode=display_mode or "heatmap", levels=levels)
                return (
                    fig, "", "success", False,
                    len(results_per_tile) - 1, marks, 0,
                    {}, {}, {}, "single",
                    {}, {},
                )

            n_ranges = len(energy_ranges)
            set_progress((0, n_ranges, f"0/{n_ranges}"))
            results = []
            for i, (e_lo, e_hi) in enumerate(energy_ranges):
                results.append(
                    stix_imaging_core.reconstruct_stix_image(
                        pixel_path, bkg_path, (start, end), (e_lo, e_hi), algorithm, algo_params, flare_location,
                    )
                )
                set_progress((i + 1, n_ranges, f"{i + 1}/{n_ranges}"))
            session_store.set(sid, "stix_imaging_result", results)
            fig = plotting.stix_imaging_figure(results, mode=display_mode or "heatmap", levels=levels)
            return fig, "", "success", False, 0, {}, 0, hide, hide, hide, "single", {}, hide
    except Exception as exc:  # noqa: BLE001
        logger.exception("Unhandled error in callback")
        return dash.no_update, str(exc), "danger", True, *err


@callback(
    Output("stix-imaging-graph", "figure", allow_duplicate=True),
    Output("stix-imaging-tile-slider-row", "style", allow_duplicate=True),
    Output("stix-imaging-play-row", "style", allow_duplicate=True),
    Output("stix-imaging-download-btn", "style", allow_duplicate=True),
    Input("stix-imaging-sequence-view", "value"),
    Input("stix-imaging-display-mode", "value"),
    Input("stix-imaging-tile-slider", "value"),
    State("stix-imaging-contour-levels-store", "data"),
    State("session-id", "data"),
    prevent_initial_call=True,
)
def toggle_stix_imaging_display(view, display_mode, tile_index, contour_levels_pct, sid):
    """Cheap redisplay of the cached stix_imaging_result - no recompute.
    result is a list[dict] (one per energy range) for a single interval, or
    a list[list[dict]] (one inner list per Sequence tile) for a sequence -
    distinguished by whether its first element is itself a list. Mosaic
    shows every tile at once, so there's no single "current" image to
    download there - stix-imaging-download-btn hides along with the
    slider/play controls in that view; stix-imaging-download-all-btn is
    untouched here (its visibility only depends on sequence vs single, set
    once by run_stix_imaging)."""
    result = session_store.get(sid, "stix_imaging_result")
    if not result:
        raise PreventUpdate
    show, hide = {}, {"display": "none"}
    levels = _contour_level_fractions(contour_levels_pct)
    is_sequence = isinstance(result[0], list)
    if is_sequence and view == "mosaic":
        fig = plotting.stix_imaging_mosaic_figure(result, mode=display_mode or "contour", levels=levels)
        return fig, hide, hide, hide
    if is_sequence:
        if tile_index is None or tile_index >= len(result):
            raise PreventUpdate
        fig = plotting.stix_imaging_figure(result[tile_index], mode=display_mode or "heatmap", levels=levels)
        return fig, show, show, show
    fig = plotting.stix_imaging_figure(result, mode=display_mode or "heatmap", levels=levels)
    return fig, hide, hide, show


@callback(
    Output("stix-imaging-mosaic-figure-store", "data"),
    Input("stix-imaging-tile-slider", "max"),
    Input("stix-imaging-display-mode", "value"),
    Input("stix-imaging-contour-levels-store", "data"),
    State("session-id", "data"),
    prevent_initial_call=True,
)
def refresh_stix_imaging_mosaic_store(_slider_max, display_mode, contour_levels_pct, sid):
    """Keeps a ready-to-export mosaic figure (every Sequence tile, as one
    grid) in a hidden Store for the client-side "Download All" button below
    - it needs the mosaic figure available even when the user is looking at
    a single tile, not just when the Mosaic view is actually selected.
    `stix-imaging-tile-slider.max` is used as the "a run just finished"
    trigger (it's one of run_stix_imaging's own Outputs, updated only after
    the background callback completes - unlike its own n_clicks Input,
    which would fire before session_store has the new result)."""
    result = session_store.get(sid, "stix_imaging_result")
    if not result or not isinstance(result[0], list):
        raise PreventUpdate
    levels = _contour_level_fractions(contour_levels_pct)
    fig = plotting.stix_imaging_mosaic_figure(result, mode=display_mode or "contour", levels=levels)
    return fig.to_plotly_json()


# Both download buttons render entirely client-side via Plotly.js (already
# loaded for the graphs) - no server-side image rendering. A prior attempt
# used kaleido (Plotly's usual server-side PNG exporter), but its headless-
# Chromium dependency failed to launch in this environment (missing VC++
# runtime) and that risk would follow into deployment too - the user chose
# the dependency-free client-side approach instead. "Download Image" saves
# exactly what's on screen (the same PNG export the graph's own modebar
# camera icon already offers, just via a labeled button); "Download All"
# saves one combined mosaic PNG (every tile in a grid, from
# stix-imaging-mosaic-figure-store) rather than a zip of separate files,
# since a real zip would need either that same broken server dependency or
# a new client-side zip-writing library.
dash.clientside_callback(
    """
    function(n_clicks) {
        if (!n_clicks) { return window.dash_clientside.no_update; }
        var gd = document.getElementById('stix-imaging-graph');
        if (gd && window.Plotly) {
            window.Plotly.downloadImage(gd, {format: 'png', filename: 'stix_image'});
        }
        return n_clicks;
    }
    """,
    Output("stix-imaging-download-dummy", "data"),
    Input("stix-imaging-download-btn", "n_clicks"),
    prevent_initial_call=True,
)

dash.clientside_callback(
    """
    function(n_clicks, figureData) {
        if (!n_clicks || !figureData || !window.Plotly) { return window.dash_clientside.no_update; }
        window.Plotly.toImage(figureData, {format: 'png'}).then(function(url) {
            var a = document.createElement('a');
            a.href = url;
            a.download = 'stix_images_mosaic.png';
            document.body.appendChild(a);
            a.click();
            document.body.removeChild(a);
        });
        return n_clicks;
    }
    """,
    Output("stix-imaging-download-all-dummy", "data"),
    Input("stix-imaging-download-all-btn", "n_clicks"),
    State("stix-imaging-mosaic-figure-store", "data"),
    prevent_initial_call=True,
)
