"""Plot Preferences - port of PlotPrefsDialog in sololab_app.py.

Three tabs (STIX / RPW / EPD), each with a plot-type dropdown that shows
different controls depending on the choice (spectrogram / time profiles /
overlay), live-synced into plot-prefs-store (no explicit Apply/Cancel step,
unlike the desktop dialog - see the migration plan's decision 8). Three
editable-list modals (STIX energy-integration bins, RPW frequencies shared
between HFR/TNR, EPD channels) reuse utils.make_list_editor_modal.
"""
import dash
import dash_bootstrap_components as dbc
from dash import Input, Output, State, callback, dcc, html
from dash.exceptions import PreventUpdate

from sololab.dash_app import plotting
from sololab.dash_app.constants import (
    EPD_PARTICLE_OPTIONS,
    PLOT_TYPE_OPTIONS,
    RPW_HFR_PREVIEW_FREQS,
    RPW_OVERLAP_OPTIONS,
    RPW_TNR_PREVIEW_FREQS,
    rpw_key,
)
from sololab.dash_app.session_store import session_store
from sololab.dash_app.utils import make_list_editor_modal, register_list_editor_add_row_callback

dash.register_page(__name__, path="/plot-prefs", name="Plot Preferences")


def layout(**kwargs):
    return dbc.Container(
        [
            html.H3("Plot Preferences", className="mt-3"),
            dbc.Alert(id="plotprefs-alert", is_open=False, dismissable=True, className="mt-2"),
            dbc.Tabs(
                [
                    dbc.Tab(_stix_tab(), label="STIX data", tab_id="stix"),
                    dbc.Tab(_rpw_tab(), label="RPW data", tab_id="rpw"),
                    dbc.Tab(_epd_tab(), label="EPD data", tab_id="epd"),
                ],
                id="plotprefs-tabs",
                active_tab="stix",
                className="mt-3",
            ),
            dcc.Graph(id="plotprefs-graph", className="mt-3"),
            html.Div(dbc.Button("Combined Plot", href="/combined-plot", color="primary"), className="mt-3"),
            _stix_energy_ranges_modal(),
            _rpw_frequencies_modal(),
            _epd_channels_modal(),
        ],
        className="pb-5",
    )


# --- STIX tab ------------------------------------------------------------------------


def _stix_tab():
    return dbc.Card(
        dbc.CardBody(
            [
                html.Label("Plot type"),
                dcc.Dropdown(id="stix-plot-type", options=PLOT_TYPE_OPTIONS, value="spectrogram", clearable=False, className="mb-2"),
                html.Div(dbc.Checkbox(id="stix-logy-energy", label="Log Y (energy)", value=False), id="stix-logy-energy-row"),
                html.Div(dbc.Checkbox(id="stix-logy-countrate", label="Log Y (count rate)", value=False), id="stix-logy-countrate-row"),
                html.Div(
                    [
                        dbc.Checkbox(id="stix-logy-energy-overlay", label="Log Y (energy)", value=False),
                        dbc.Checkbox(id="stix-logy-countrate-overlay", label="Log Y (count rate)", value=False),
                    ],
                    id="stix-logy-overlay-row",
                ),
                html.Div(dbc.Checkbox(id="stix-logz", label="Log Z", value=True), id="stix-logz-row"),
                html.Div(
                    [html.Label("Smoothing points"), dbc.Input(id="stix-smoothing", type="number", min=1, max=100, value=1)],
                    id="stix-smoothing-row",
                ),
                html.Div(
                    [
                        dbc.Checkbox(id="stix-energy-range-enabled", label="Limit energy range (keV)", value=False),
                        dbc.Row(
                            [
                                dbc.Col(dbc.Input(id="stix-energy-min", type="number", min=4, max=150, value=4)),
                                dbc.Col(dbc.Input(id="stix-energy-max", type="number", min=4, max=150, value=28)),
                            ]
                        ),
                    ],
                    id="stix-energy-range-row",
                ),
                html.Div(
                    dbc.Button("Set Energy Ranges", id="stix-set-energy-ranges-btn", color="secondary", outline=True),
                    id="stix-energy-ranges-row",
                    className="mb-2",
                ),
                html.Hr(),
                dbc.ButtonGroup(
                    [
                        dbc.Button("Plot STIX Preview", id="stix-prefs-preview-btn", color="secondary"),
                        dbc.Button("Plot STIX Background", id="stix-prefs-bkg-btn", color="secondary", disabled=True),
                    ]
                ),
            ]
        ),
        className="mt-2",
    )


# --- RPW tab -------------------------------------------------------------------------


def _rpw_tab():
    return dbc.Card(
        dbc.CardBody(
            [
                html.Label("Plot type"),
                dcc.Dropdown(id="rpw-plot-type", options=PLOT_TYPE_OPTIONS, value="spectrogram", clearable=False, className="mb-2"),
                html.Div(dbc.Checkbox(id="rpw-logy-frequency", label="Log Y (frequency)", value=False), id="rpw-logy-frequency-row"),
                html.Div(dbc.Checkbox(id="rpw-logy-intensity", label="Log Y (intensity)", value=False), id="rpw-logy-intensity-row"),
                html.Div(
                    [
                        dbc.Checkbox(id="rpw-logy-frequency-overlay", label="Log Y (frequency)", value=False),
                        dbc.Checkbox(id="rpw-logy-intensity-overlay", label="Log Y (intensity)", value=False),
                    ],
                    id="rpw-logy-overlay-row",
                ),
                html.Div(dbc.Checkbox(id="rpw-logz", label="Log Z", value=True), id="rpw-logz-row"),
                html.Div(dbc.Checkbox(id="rpw-invert-y", label="Invert Y axis", value=True), id="rpw-invert-y-row"),
                html.Div(
                    [
                        html.Label("HFR/TNR overlap handling"),
                        dcc.Dropdown(id="rpw-overlay-choice", options=RPW_OVERLAP_OPTIONS, value="Both", clearable=False),
                    ],
                    id="rpw-overlay-choice-row",
                ),
                html.Div(
                    [html.Label("Smoothing points"), dbc.Input(id="rpw-smoothing", type="number", min=1, max=100, value=1)],
                    id="rpw-smoothing-row",
                ),
                dbc.Checkbox(id="rpw-freq-range-enabled", label="Limit frequency range (kHz)", value=False),
                dbc.Row(
                    [
                        dbc.Col(dbc.Input(id="rpw-freq-min", type="text", value="100")),
                        dbc.Col(dbc.Input(id="rpw-freq-max", type="text", value="8000")),
                    ],
                    className="mb-2",
                ),
                html.Div(
                    dbc.Button("Select Frequencies", id="rpw-select-frequencies-btn", color="secondary", outline=True),
                    id="rpw-select-frequencies-row",
                    className="mb-2",
                ),
                html.Hr(),
                dbc.ButtonGroup(
                    [
                        dbc.Button("Plot RPW-HFR Preview", id="rpw-prefs-preview-hfr-btn", color="secondary"),
                        dbc.Button("Plot RPW-TNR Preview", id="rpw-prefs-preview-tnr-btn", color="secondary"),
                        dbc.Button("Plot RPW-HFR Background", id="rpw-prefs-bkg-hfr-btn", color="secondary", disabled=True),
                        dbc.Button("Plot RPW-TNR Background", id="rpw-prefs-bkg-tnr-btn", color="secondary", disabled=True),
                    ],
                    vertical=True,
                ),
            ]
        ),
        className="mt-2",
    )


# --- EPD tab -------------------------------------------------------------------------


def _epd_tab():
    return dbc.Card(
        dbc.CardBody(
            [
                dbc.Checkbox(id="epd-prefs-logy", label="Log Y", value=False, className="mb-2"),
                dbc.Button(
                    "Select Energy Channels", id="epd-select-channels-btn", color="secondary", outline=True, disabled=True, className="mb-2"
                ),
                html.Hr(),
                dbc.Button("Plot EPD Preview", id="epd-prefs-preview-btn", color="secondary"),
            ]
        ),
        className="mt-2",
    )


def _stix_energy_ranges_modal():
    return make_list_editor_modal(
        "stix-energy-ranges",
        "STIX Energy Integration Ranges (keV)",
        [{"name": "Min", "id": "min", "type": "numeric"}, {"name": "Max", "id": "max", "type": "numeric"}],
    )


def _rpw_frequencies_modal():
    return make_list_editor_modal(
        "rpw-frequencies", "RPW Frequencies (kHz)", [{"name": "Frequency (kHz)", "id": "freq", "type": "numeric"}]
    )


def _epd_channels_modal():
    return make_list_editor_modal("epd-channels", "EPD Energy Channels", [{"name": "Channel", "id": "channel", "type": "numeric"}])


register_list_editor_add_row_callback("stix-energy-ranges", {"min": 4, "max": 12})
register_list_editor_add_row_callback("rpw-frequencies", {"freq": 500})
register_list_editor_add_row_callback("epd-channels", {"channel": 0})


# --- visibility toggling ---------------------------------------------------------------


@callback(
    Output("stix-logy-energy-row", "style"),
    Output("stix-logy-countrate-row", "style"),
    Output("stix-logy-overlay-row", "style"),
    Output("stix-logz-row", "style"),
    Output("stix-smoothing-row", "style"),
    Output("stix-energy-range-row", "style"),
    Output("stix-energy-ranges-row", "style"),
    Input("stix-plot-type", "value"),
)
def toggle_stix_controls(plot_type):
    show, hide = {"display": "block"}, {"display": "none"}
    spec, curve, overlay = plot_type == "spectrogram", plot_type == "time profiles", plot_type == "overlay"
    return (
        show if spec else hide,
        show if curve else hide,
        show if overlay else hide,
        show if (spec or overlay) else hide,
        show if (curve or overlay) else hide,
        show if (spec or overlay) else hide,
        show if (curve or overlay) else hide,
    )


@callback(
    Output("rpw-logy-frequency-row", "style"),
    Output("rpw-logy-intensity-row", "style"),
    Output("rpw-logy-overlay-row", "style"),
    Output("rpw-logz-row", "style"),
    Output("rpw-invert-y-row", "style"),
    Output("rpw-smoothing-row", "style"),
    Output("rpw-select-frequencies-row", "style"),
    Input("rpw-plot-type", "value"),
)
def toggle_rpw_controls(plot_type):
    show, hide = {"display": "block"}, {"display": "none"}
    spec, curve, overlay = plot_type == "spectrogram", plot_type == "time profiles", plot_type == "overlay"
    return (
        show if spec else hide,
        show if curve else hide,
        show if overlay else hide,
        show if (spec or overlay) else hide,
        show if (spec or overlay) else hide,
        show if (curve or overlay) else hide,
        show if (curve or overlay) else hide,
    )


@callback(Output("epd-select-channels-btn", "disabled"), Input("instrument-status-store", "data"))
def toggle_epd_channels_btn(status):
    status = status or {}
    return not status.get("epd", {}).get("loaded")


@callback(
    Output("stix-prefs-bkg-btn", "disabled"),
    Input("instrument-status-store", "data"),
)
def toggle_stix_bkg_btn(status):
    status = status or {}
    return not status.get("stix", {}).get("bkg_enabled")


@callback(
    Output("rpw-prefs-bkg-hfr-btn", "disabled"),
    Output("rpw-prefs-bkg-tnr-btn", "disabled"),
    Input("instrument-status-store", "data"),
)
def toggle_rpw_bkg_btns(status):
    status = status or {}
    return (
        not status.get("rpw_hfr", {}).get("bkg_enabled"),
        not status.get("rpw_tnr", {}).get("bkg_enabled"),
    )


# --- live sync to plot-prefs-store ----------------------------------------------------


@callback(
    Output("plot-prefs-store", "data", allow_duplicate=True),
    Input("stix-plot-type", "value"),
    Input("stix-logy-energy", "value"),
    Input("stix-logy-countrate", "value"),
    Input("stix-logy-energy-overlay", "value"),
    Input("stix-logy-countrate-overlay", "value"),
    Input("stix-logz", "value"),
    Input("stix-smoothing", "value"),
    Input("stix-energy-range-enabled", "value"),
    Input("stix-energy-min", "value"),
    Input("stix-energy-max", "value"),
    State("plot-prefs-store", "data"),
    prevent_initial_call=True,
)
def sync_stix_prefs(plot_type, logy_e, logy_c, logy_eo, logy_co, logz, smoothing, range_enabled, e_min, e_max, prefs):
    prefs = dict(prefs or {})
    prefs["stix"] = {
        **prefs.get("stix", {}),
        "type": plot_type,
        "logy": bool(logy_e),
        "logy_countrate": bool(logy_c),
        "logy_energy_overlay": bool(logy_eo),
        "logy_countrate_overlay": bool(logy_co),
        "logz": bool(logz),
        "smoothing_points": smoothing or 1,
        "energy_range_enabled": bool(range_enabled),
        "energy_range": [e_min or 4, e_max or 28],
    }
    return prefs


@callback(
    Output("plot-prefs-store", "data", allow_duplicate=True),
    Input("rpw-plot-type", "value"),
    Input("rpw-logy-frequency", "value"),
    Input("rpw-logy-intensity", "value"),
    Input("rpw-logy-frequency-overlay", "value"),
    Input("rpw-logy-intensity-overlay", "value"),
    Input("rpw-logz", "value"),
    Input("rpw-invert-y", "value"),
    Input("rpw-overlay-choice", "value"),
    Input("rpw-smoothing", "value"),
    Input("rpw-freq-range-enabled", "value"),
    Input("rpw-freq-min", "value"),
    Input("rpw-freq-max", "value"),
    State("plot-prefs-store", "data"),
    prevent_initial_call=True,
)
def sync_rpw_prefs(plot_type, logy_f, logy_i, logy_fo, logy_io, logz, invert_y, overlay_choice, smoothing, range_enabled, f_min, f_max, prefs):
    prefs = dict(prefs or {})
    freq_range_enabled = False
    try:
        fmin, fmax = float(f_min), float(f_max)
        freq_range_enabled = bool(range_enabled) and 1 <= fmin < fmax <= 16400
    except (TypeError, ValueError):
        fmin, fmax = None, None
    prefs["rpw"] = {
        **prefs.get("rpw", {}),
        "type": plot_type,
        "logy": bool(logy_f),
        "logy_intensity": bool(logy_i),
        "logy_frequency_overlay": bool(logy_fo),
        "logy_intensity_overlay": bool(logy_io),
        "logz": bool(logz),
        "invert_y": bool(invert_y),
        "overlay": overlay_choice,
        "smoothing_points": smoothing or 1,
        "freq_range_enabled": freq_range_enabled,
        "freq_min": fmin,
        "freq_max": fmax,
    }
    return prefs


@callback(
    Output("plot-prefs-store", "data", allow_duplicate=True),
    Input("epd-prefs-logy", "value"),
    State("plot-prefs-store", "data"),
    prevent_initial_call=True,
)
def sync_epd_prefs(logy, prefs):
    prefs = dict(prefs or {})
    prefs["epd"] = {**prefs.get("epd", {}), "logy": bool(logy)}
    return prefs


# --- editable-list modals: open (populate) / done (persist) --------------------------


@callback(
    Output("stix-energy-ranges-modal", "is_open", allow_duplicate=True),
    Output("stix-energy-ranges-table", "data", allow_duplicate=True),
    Input("stix-set-energy-ranges-btn", "n_clicks"),
    State("plot-prefs-store", "data"),
    prevent_initial_call=True,
)
def open_stix_energy_ranges(n_clicks, prefs):
    if not n_clicks:
        raise PreventUpdate
    ranges = (prefs or {}).get("stix", {}).get("energy_ranges", [])
    return True, [{"min": lo, "max": hi} for lo, hi in ranges]


@callback(
    Output("stix-energy-ranges-modal", "is_open", allow_duplicate=True),
    Output("plot-prefs-store", "data", allow_duplicate=True),
    Output("stix-energy-ranges-error", "children"),
    Input("stix-energy-ranges-done-btn", "n_clicks"),
    State("stix-energy-ranges-table", "data"),
    State("plot-prefs-store", "data"),
    prevent_initial_call=True,
)
def close_stix_energy_ranges(n_clicks, rows, prefs):
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
    prefs = dict(prefs or {})
    prefs["stix"] = {**prefs.get("stix", {}), "energy_ranges": ranges}
    return False, prefs, ""


@callback(
    Output("rpw-frequencies-modal", "is_open", allow_duplicate=True),
    Output("rpw-frequencies-table", "data", allow_duplicate=True),
    Input("rpw-select-frequencies-btn", "n_clicks"),
    State("plot-prefs-store", "data"),
    prevent_initial_call=True,
)
def open_rpw_frequencies(n_clicks, prefs):
    if not n_clicks:
        raise PreventUpdate
    freqs = (prefs or {}).get("rpw", {}).get("selected_frequencies", [])
    return True, [{"freq": f} for f in freqs]


@callback(
    Output("rpw-frequencies-modal", "is_open", allow_duplicate=True),
    Output("plot-prefs-store", "data", allow_duplicate=True),
    Input("rpw-frequencies-done-btn", "n_clicks"),
    State("rpw-frequencies-table", "data"),
    State("plot-prefs-store", "data"),
    prevent_initial_call=True,
)
def close_rpw_frequencies(n_clicks, rows, prefs):
    if not n_clicks:
        raise PreventUpdate
    freqs = []
    for row in rows or []:
        try:
            freqs.append(float(row["freq"]))
        except (KeyError, TypeError, ValueError):
            continue
    prefs = dict(prefs or {})
    prefs["rpw"] = {**prefs.get("rpw", {}), "selected_frequencies": freqs}
    return False, prefs


@callback(
    Output("epd-channels-modal", "is_open", allow_duplicate=True),
    Output("epd-channels-table", "data", allow_duplicate=True),
    Input("epd-select-channels-btn", "n_clicks"),
    State("plot-prefs-store", "data"),
    prevent_initial_call=True,
)
def open_epd_channels(n_clicks, prefs):
    if not n_clicks:
        raise PreventUpdate
    channels = (prefs or {}).get("epd", {}).get("selected_channels", [])
    return True, [{"channel": c} for c in channels]


@callback(
    Output("epd-channels-modal", "is_open", allow_duplicate=True),
    Output("plot-prefs-store", "data", allow_duplicate=True),
    Input("epd-channels-done-btn", "n_clicks"),
    State("epd-channels-table", "data"),
    State("plot-prefs-store", "data"),
    prevent_initial_call=True,
)
def close_epd_channels(n_clicks, rows, prefs):
    if not n_clicks:
        raise PreventUpdate
    channels = []
    for row in rows or []:
        try:
            channels.append(int(row["channel"]))
        except (KeyError, TypeError, ValueError):
            continue
    prefs = dict(prefs or {})
    prefs["epd"] = {**prefs.get("epd", {}), "selected_channels": channels}
    return False, prefs


# --- previews --------------------------------------------------------------------------


@callback(
    Output("plotprefs-graph", "figure"),
    Output("plotprefs-alert", "children"),
    Output("plotprefs-alert", "color"),
    Output("plotprefs-alert", "is_open"),
    Input("stix-prefs-preview-btn", "n_clicks"),
    State("plot-prefs-store", "data"),
    State("session-id", "data"),
    prevent_initial_call=True,
)
def stix_prefs_preview(n_clicks, prefs, sid):
    if not n_clicks:
        raise PreventUpdate
    try:
        counts = session_store.get(sid, "stix_counts_final")
        if counts is None:
            raise ValueError("No STIX data loaded. Import STIX data first.")
        p = (prefs or {}).get("stix", {})
        plot_type = p.get("type", "spectrogram")
        energy_range = p.get("energy_range") if p.get("energy_range_enabled") else None
        if plot_type == "spectrogram":
            fig = plotting.stix_spectrogram_figure(counts, energy_range=energy_range, logscale=p.get("logz", True), ylogscale=p.get("logy", False))
        elif plot_type == "time profiles":
            fig = plotting.stix_counts_figure(
                counts,
                integrate_bins=p.get("energy_ranges") or None,
                smoothing_pts=p.get("smoothing_points", 1),
                ylogscale=p.get("logy_countrate", False),
            )
        else:
            fig = plotting.stix_overlay_figure(
                counts,
                energy_range=energy_range,
                stix_energy_bins=p.get("energy_ranges") or None,
                stix_smoothing_points=p.get("smoothing_points", 1),
                stix_spec_zlogscale=p.get("logz", True),
                stix_spec_ylogscale=p.get("logy_energy_overlay", False),
                stix_curves_ylogscale=p.get("logy_countrate_overlay", True),
            )
        return fig, "", "success", False
    except Exception as exc:  # noqa: BLE001
        return dash.no_update, str(exc), "danger", True


@callback(
    Output("plotprefs-graph", "figure", allow_duplicate=True),
    Output("plotprefs-alert", "children", allow_duplicate=True),
    Output("plotprefs-alert", "color", allow_duplicate=True),
    Output("plotprefs-alert", "is_open", allow_duplicate=True),
    Input("stix-prefs-bkg-btn", "n_clicks"),
    State("session-id", "data"),
    prevent_initial_call=True,
)
def stix_prefs_bkg_preview(n_clicks, sid):
    if not n_clicks:
        raise PreventUpdate
    counts = session_store.get(sid, "stix_counts_final")
    return plotting.stix_bkg_figure(counts), "", "success", False


def _rpw_prefs_preview(data_type, prefs):
    p = (prefs or {}).get("rpw", {})
    plot_type = p.get("type", "spectrogram")
    freq_range = [p["freq_min"], p["freq_max"]] if p.get("freq_range_enabled") else None
    default_freqs = RPW_HFR_PREVIEW_FREQS if data_type == "hfr" else RPW_TNR_PREVIEW_FREQS
    freqs = p.get("selected_frequencies") or default_freqs

    def _load(sid):
        psd = session_store.get(sid, rpw_key(data_type, "psd_final"))
        if psd is None:
            raise ValueError(f"No RPW-{data_type.upper()} data loaded. Import it first.")
        return psd

    return plot_type, freq_range, freqs, _load


@callback(
    Output("plotprefs-graph", "figure", allow_duplicate=True),
    Output("plotprefs-alert", "children", allow_duplicate=True),
    Output("plotprefs-alert", "color", allow_duplicate=True),
    Output("plotprefs-alert", "is_open", allow_duplicate=True),
    Input("rpw-prefs-preview-hfr-btn", "n_clicks"),
    Input("rpw-prefs-preview-tnr-btn", "n_clicks"),
    State("plot-prefs-store", "data"),
    State("session-id", "data"),
    prevent_initial_call=True,
)
def rpw_prefs_preview(n_hfr, n_tnr, prefs, sid):
    triggered = dash.callback_context.triggered_id
    if triggered == "rpw-prefs-preview-hfr-btn" and not n_hfr:
        raise PreventUpdate
    if triggered == "rpw-prefs-preview-tnr-btn" and not n_tnr:
        raise PreventUpdate
    data_type = "hfr" if triggered == "rpw-prefs-preview-hfr-btn" else "tnr"
    try:
        plot_type, freq_range, freqs, load = _rpw_prefs_preview(data_type, prefs)
        psd = load(sid)
        p = (prefs or {}).get("rpw", {})
        if plot_type == "spectrogram":
            fig = plotting.rpw_psd_figure(psd, frequency_range=freq_range)
        elif plot_type == "time profiles":
            fig = plotting.rpw_curve_figure(
                psd, freqs, smoothing_pts=p.get("smoothing_points", 1), ylogscale=p.get("logy_intensity", False)
            )
        else:
            fig = plotting.rpw_overlay_figure(
                psd, freqs, frequency_range=freq_range, invert_y=p.get("invert_y", True), smoothing_pts=p.get("smoothing_points", 1)
            )
        return fig, "", "success", False
    except Exception as exc:  # noqa: BLE001
        return dash.no_update, str(exc), "danger", True


@callback(
    Output("plotprefs-graph", "figure", allow_duplicate=True),
    Output("plotprefs-alert", "children", allow_duplicate=True),
    Output("plotprefs-alert", "color", allow_duplicate=True),
    Output("plotprefs-alert", "is_open", allow_duplicate=True),
    Input("rpw-prefs-bkg-hfr-btn", "n_clicks"),
    Input("rpw-prefs-bkg-tnr-btn", "n_clicks"),
    State("session-id", "data"),
    prevent_initial_call=True,
)
def rpw_prefs_bkg_preview(n_hfr, n_tnr, sid):
    triggered = dash.callback_context.triggered_id
    if triggered == "rpw-prefs-bkg-hfr-btn" and not n_hfr:
        raise PreventUpdate
    if triggered == "rpw-prefs-bkg-tnr-btn" and not n_tnr:
        raise PreventUpdate
    data_type = "hfr" if triggered == "rpw-prefs-bkg-hfr-btn" else "tnr"
    psd = session_store.get(sid, rpw_key(data_type, "psd_final"))
    return plotting.rpw_bkg_figure(psd), "", "success", False


@callback(
    Output("plotprefs-graph", "figure", allow_duplicate=True),
    Output("plotprefs-alert", "children", allow_duplicate=True),
    Output("plotprefs-alert", "color", allow_duplicate=True),
    Output("plotprefs-alert", "is_open", allow_duplicate=True),
    Input("epd-prefs-preview-btn", "n_clicks"),
    State("plot-prefs-store", "data"),
    State("session-id", "data"),
    prevent_initial_call=True,
)
def epd_prefs_preview(n_clicks, prefs, sid):
    if not n_clicks:
        raise PreventUpdate
    try:
        meta = session_store.get(sid, "epd_meta")
        if not meta:
            raise ValueError("No EPD data loaded. Import EPD data first.")
        particle = meta["particle"]
        epd_data = session_store.get(sid, "epd_electrons_df" if particle == "Electron" else "epd_protons_df")
        energies = session_store.get(sid, "epd_energies")
        channels = (prefs or {}).get("epd", {}).get("selected_channels") or [2, 6, 14, 18, 26]

        from datetime import datetime as _dt

        day = _dt.strptime(meta["date"], "%Y-%m-%d")
        date_range = (day.replace(hour=0, minute=0, second=0), day.replace(hour=23, minute=59, second=59))
        fig = plotting.epd_flux_figure(
            epd_data, energies, particle=particle, channels=channels, date_range=date_range, resample=meta.get("resample")
        )
        return fig, "", "success", False
    except Exception as exc:  # noqa: BLE001
        return dash.no_update, str(exc), "danger", True
