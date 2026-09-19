"""Combined Plot - port of CombinedPlotDialog (+ InstrumentSelectionDialog)
in sololab_app.py. Stacks the loaded instruments into one quicklook figure
via plotting.quicklook_plot_plotly.

Panel order is user-adjustable (up/down arrows) rather than the desktop's
fixed Add/Remove insertion order, and persists in plot-prefs-store (like
every other setting on this page - selected instruments, line width, font
size, date-range toggle) so it survives navigating away and back, per
explicit user request. Default top-to-bottom order is EPD, TNR, HFR, STIX -
also per explicit request (a combined figure is easier to read roughly
square, so settings live in a side column here rather than stacked above
the plot like the single-instrument pages).
"""
import logging
from datetime import datetime

import dash
import dash_bootstrap_components as dbc
from dash import ALL, Input, Output, State, callback, dcc, html
from dash.exceptions import PreventUpdate

from sololab.dash_app import plotting
from sololab.dash_app.constants import DEFAULT_PLOT_PREFS, INSTRUMENT_ORDER, rpw_key
from sololab.dash_app.session_store import session_store
from sololab.dash_app.utils import format_dt_input, parse_dt_input

logger = logging.getLogger(__name__)

dash.register_page(__name__, path="/combined-plot", name="Combined Plot")

INSTRUMENT_LABELS = {"stix": "STIX", "hfr": "RPW-HFR", "tnr": "RPW-TNR", "epd": "EPD"}
STATUS_KEYS = {"stix": "stix", "hfr": "rpw_hfr", "tnr": "rpw_tnr", "epd": "epd"}


def _order_rows(display):
    """One row per currently-selected instrument, with up/down buttons to
    move it in the panel stack. Top of this list = top panel of the figure."""
    rows = []
    for i, code in enumerate(display):
        rows.append(
            dbc.Row(
                [
                    dbc.Col(INSTRUMENT_LABELS[code], width=6),
                    dbc.Col(
                        dbc.ButtonGroup(
                            [
                                dbc.Button("^", id={"type": "cp-order-move", "code": code, "dir": "up"}, size="sm", outline=True, color="secondary", disabled=i == 0),
                                dbc.Button("v", id={"type": "cp-order-move", "code": code, "dir": "down"}, size="sm", outline=True, color="secondary", disabled=i == len(display) - 1),
                            ]
                        ),
                        width=6,
                    ),
                ],
                className="mb-1 align-items-center",
            )
        )
    return rows


def layout(**kwargs):
    return dbc.Container(
        [
            html.H3("Combined Plot", className="mt-3"),
            dbc.Alert(id="cp-alert", is_open=False, dismissable=True, className="mt-2"),
            dbc.Row(
                [
                    dbc.Col(
                        dbc.Card(
                            dbc.CardBody(
                                [
                                    dbc.Button("Select instruments to plot", id="cp-select-instruments-btn", color="secondary", outline=True, className="w-100"),
                                    html.Div(id="cp-selected-instruments-label", className="text-muted small mt-1"),
                                    html.Hr(),
                                    html.Div("Panel order (top to bottom)", className="small text-muted mb-1"),
                                    html.Div(id="cp-order-rows"),
                                    html.Hr(),
                                    dbc.Checkbox(id="cp-date-range-enabled", label="Restrict date range", value=False),
                                    dbc.Row(
                                        [
                                            dbc.Col(dbc.Input(id="cp-date-start", type="text", placeholder="YYYY-MM-DD HH:MM:SS")),
                                            dbc.Col(dbc.Input(id="cp-date-end", type="text", placeholder="YYYY-MM-DD HH:MM:SS")),
                                        ],
                                        className="mb-2",
                                    ),
                                    dbc.Row(
                                        [
                                            dbc.Col([html.Label("Line width"), dbc.Input(id="cp-linewidth", type="number", min=0.1, max=5.0, step=0.1, value=1.5)]),
                                            dbc.Col([html.Label("Font size"), dbc.Input(id="cp-fontsize", type="number", min=3, max=48, value=6)]),
                                        ],
                                        className="mb-2",
                                    ),
                                    dbc.Button("PLOT", id="cp-plot-btn", color="success", className="w-100"),
                                ]
                            )
                        ),
                        md=4,
                    ),
                    dbc.Col(dcc.Graph(id="cp-combined-graph"), md=8),
                ]
            ),
            dcc.Store(id="cp-display-instruments", data=[]),
            dbc.Modal(
                [
                    dbc.ModalHeader(dbc.ModalTitle("Select instruments to plot")),
                    dbc.ModalBody(dbc.Checklist(id="cp-instrument-checklist", options=[], value=[])),
                    dbc.ModalFooter(dbc.Button("Done", id="cp-instruments-done-btn", color="primary")),
                ],
                id="cp-instrument-modal",
                is_open=False,
            ),
        ],
        className="pb-5",
    )


# --- restore persisted settings on page mount -----------------------------------------


@callback(
    Output("cp-display-instruments", "data", allow_duplicate=True),
    Output("cp-date-range-enabled", "value"),
    Output("cp-linewidth", "value"),
    Output("cp-fontsize", "value"),
    Input("url", "pathname"),
    State("plot-prefs-store", "data"),
    prevent_initial_call=True,
)
def restore_combined_prefs(pathname, prefs):
    if pathname != "/combined-plot":
        raise PreventUpdate
    cp = (prefs or {}).get("combined", DEFAULT_PLOT_PREFS["combined"])
    return (
        cp.get("display_instruments", []),
        cp.get("date_range_enabled", False),
        cp.get("linewidth", 1.5),
        cp.get("fontsize", 6),
    )


# --- instrument selection + ordering --------------------------------------------------


@callback(
    Output("cp-instrument-modal", "is_open"),
    Output("cp-instrument-checklist", "options"),
    Output("cp-instrument-checklist", "value"),
    Input("cp-select-instruments-btn", "n_clicks"),
    State("instrument-status-store", "data"),
    State("cp-display-instruments", "data"),
    prevent_initial_call=True,
)
def open_instrument_modal(n_clicks, status, current):
    if not n_clicks:
        raise PreventUpdate
    status = status or {}
    options = [
        {"label": INSTRUMENT_LABELS[code], "value": code, "disabled": not status.get(STATUS_KEYS[code], {}).get("loaded")}
        for code in INSTRUMENT_ORDER
    ]
    return True, options, current or []


@callback(
    Output("cp-instrument-modal", "is_open", allow_duplicate=True),
    Output("cp-display-instruments", "data", allow_duplicate=True),
    Output("plot-prefs-store", "data", allow_duplicate=True),
    Input("cp-instruments-done-btn", "n_clicks"),
    State("cp-instrument-checklist", "value"),
    State("plot-prefs-store", "data"),
    prevent_initial_call=True,
)
def close_instrument_modal(n_clicks, selected, prefs):
    if not n_clicks:
        raise PreventUpdate
    selected = set(selected or [])
    prefs = prefs or {}
    cp = dict(DEFAULT_PLOT_PREFS["combined"], **prefs.get("combined", {}))
    prior_order = cp.get("panel_order", DEFAULT_PLOT_PREFS["combined"]["panel_order"])
    # keep prior relative order for instruments still selected, append any
    # newly-selected one not seen before at the end
    ordered = [c for c in prior_order if c in selected] + [c for c in selected if c not in prior_order]
    cp["panel_order"] = ordered
    cp["display_instruments"] = ordered
    prefs["combined"] = cp
    return False, ordered, prefs


@callback(
    Output("cp-order-rows", "children"),
    Input("cp-display-instruments", "data"),
)
def render_order_rows(display):
    return _order_rows(display or [])


@callback(
    Output("cp-display-instruments", "data", allow_duplicate=True),
    Output("plot-prefs-store", "data", allow_duplicate=True),
    Input({"type": "cp-order-move", "code": ALL, "dir": ALL}, "n_clicks"),
    State("cp-display-instruments", "data"),
    State("plot-prefs-store", "data"),
    prevent_initial_call=True,
)
def move_panel(n_clicks_list, display, prefs):
    if not any(n_clicks_list):
        raise PreventUpdate
    ctx = dash.callback_context
    if not ctx.triggered_id:
        raise PreventUpdate
    code, direction = ctx.triggered_id["code"], ctx.triggered_id["dir"]
    display = list(display or [])
    i = display.index(code)
    j = i - 1 if direction == "up" else i + 1
    if not (0 <= j < len(display)):
        raise PreventUpdate
    display[i], display[j] = display[j], display[i]

    prefs = prefs or {}
    cp = dict(DEFAULT_PLOT_PREFS["combined"], **prefs.get("combined", {}))
    cp["panel_order"] = display
    cp["display_instruments"] = display
    prefs["combined"] = cp
    return display, prefs


@callback(
    Output("cp-selected-instruments-label", "children"),
    Input("cp-display-instruments", "data"),
)
def update_selected_label(display):
    display = display or []
    return "Selected: " + ", ".join(INSTRUMENT_LABELS[c] for c in display) if display else "No instruments selected."


# --- misc field sync (persisted, mirrors plot_prefs.py's live-sync pattern) ------------


@callback(
    Output("plot-prefs-store", "data", allow_duplicate=True),
    Input("cp-date-range-enabled", "value"),
    Input("cp-linewidth", "value"),
    Input("cp-fontsize", "value"),
    State("plot-prefs-store", "data"),
    prevent_initial_call=True,
)
def sync_combined_fields(date_range_enabled, linewidth, fontsize, prefs):
    prefs = prefs or {}
    cp = dict(DEFAULT_PLOT_PREFS["combined"], **prefs.get("combined", {}))
    cp["date_range_enabled"] = bool(date_range_enabled)
    cp["linewidth"] = linewidth
    cp["fontsize"] = fontsize
    prefs["combined"] = cp
    return prefs


@callback(
    Output("cp-date-start", "value"),
    Output("cp-date-end", "value"),
    Input("cp-display-instruments", "data"),
    State("instrument-status-store", "data"),
    prevent_initial_call=True,
)
def prefill_date_range(display, status):
    if not display:
        raise PreventUpdate
    status = status or {}
    starts, ends = [], []
    for code in display:
        info = status.get(STATUS_KEYS[code], {})
        if not info.get("loaded"):
            continue
        if code == "epd":
            day = info.get("date")
            if day:
                starts.append(f"{day} 00:00:00")
                ends.append(f"{day} 23:59:59")
        else:
            if info.get("min_time"):
                starts.append(info["min_time"])
            if info.get("max_time"):
                ends.append(info["max_time"])
    if not starts or not ends:
        raise PreventUpdate
    start = max(datetime.strptime(s, "%Y-%m-%d %H:%M:%S") for s in starts)
    end = min(datetime.strptime(e, "%Y-%m-%d %H:%M:%S") for e in ends)
    return format_dt_input(start), format_dt_input(end)


@callback(
    Output("cp-combined-graph", "figure"),
    Output("cp-alert", "children"),
    Output("cp-alert", "color"),
    Output("cp-alert", "is_open"),
    Input("cp-plot-btn", "n_clicks"),
    State("cp-display-instruments", "data"),
    State("cp-date-range-enabled", "value"),
    State("cp-date-start", "value"),
    State("cp-date-end", "value"),
    State("cp-linewidth", "value"),
    State("cp-fontsize", "value"),
    State("plot-prefs-store", "data"),
    State("session-id", "data"),
    prevent_initial_call=True,
)
def render_combined_plot(n_clicks, display, date_range_enabled, date_start, date_end, linewidth, fontsize, prefs, sid):
    if not n_clicks:
        raise PreventUpdate
    if not display:
        return dash.no_update, "Select at least one instrument first.", "danger", True
    try:
        prefs = prefs or {}
        stix_p = prefs.get("stix", {})
        rpw_p = prefs.get("rpw", {})
        epd_p = prefs.get("epd", {})

        type_map = {"spectrogram": "spec", "time profiles": "curve", "overlay": "overlay"}
        overlap_map = {"Only HFR": "hfr", "Only TNR": "tnr", "Both": "both"}

        stix_energy_range = stix_p.get("energy_range") if stix_p.get("energy_range_enabled") else [4, 28]
        stix_energy_bins = stix_p.get("energy_ranges") or [[4, 12], [16, 28]]
        rpw_freqs = rpw_p.get("selected_frequencies") or []
        rpw_freq_range = (
            [rpw_p["freq_min"], rpw_p["freq_max"]] if rpw_p.get("freq_range_enabled") else None
        )

        date_range = None
        if date_range_enabled and date_start and date_end:
            date_range = (parse_dt_input(date_start), parse_dt_input(date_end))

        stix_counts = session_store.get(sid, "stix_counts_final") if "stix" in display else None
        hfr_psd = session_store.get(sid, rpw_key("hfr", "psd_final")) if "hfr" in display else None
        tnr_psd = session_store.get(sid, rpw_key("tnr", "psd_final")) if "tnr" in display else None

        epd_data, epd_energies, epd_particle, epd_resample = None, None, "Electron", None
        if "epd" in display:
            epd_meta = session_store.get(sid, "epd_meta") or {}
            epd_particle = epd_meta.get("particle", "Electron")
            epd_resample = epd_meta.get("resample")
            epd_data = session_store.get(sid, "epd_electrons_final" if epd_particle == "Electron" else "epd_protons_final")
            epd_energies = session_store.get(sid, "epd_energies")

        fig = plotting.quicklook_plot_plotly(
            stix_counts=stix_counts,
            hfr_psd=hfr_psd,
            tnr_psd=tnr_psd,
            epd_data=epd_data,
            epd_energies=epd_energies,
            display=display,
            date_range=date_range,
            stix_energy_range=stix_energy_range,
            stix_energy_bins=stix_energy_bins,
            stix_mode=type_map.get(stix_p.get("type", "spectrogram"), "spec"),
            stix_smoothing_points=stix_p.get("smoothing_points", 1),
            stix_curves_ylogscale=stix_p.get("logy_countrate_overlay", True) if stix_p.get("type") == "overlay" else stix_p.get("logy_countrate", True),
            stix_spec_ylogscale=stix_p.get("logy_energy_overlay", False) if stix_p.get("type") == "overlay" else stix_p.get("logy", False),
            stix_spec_zlogscale=stix_p.get("logz", True),
            rpw_frequency_range=rpw_freq_range,
            hfr_frequencies=rpw_freqs,
            tnr_frequencies=rpw_freqs,
            rpw_mode=type_map.get(rpw_p.get("type", "spectrogram"), "spec"),
            rpw_overlap=overlap_map.get(rpw_p.get("overlay", "Both"), "both"),
            rpw_units="wmhz",
            rpw_invert_yaxis=rpw_p.get("invert_y", True),
            rpw_smoothing_points=rpw_p.get("smoothing_points", 1),
            epd_channels=epd_p.get("selected_channels") or [2, 6, 14, 18, 26],
            epd_particle=epd_particle,
            epd_round_label=True,
            epd_resample=epd_resample,
            fontsize=fontsize or 6,
            linewidth=linewidth or 1.5,
        )
        return fig, "", "success", False
    except Exception as exc:  # noqa: BLE001
        logger.exception("Unhandled error in callback")
        return dash.no_update, str(exc), "danger", True
