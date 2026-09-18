"""Combined Plot - port of CombinedPlotDialog (+ InstrumentSelectionDialog)
in sololab_app.py. Stacks the loaded instruments into one quicklook figure
via plotting.quicklook_plot_plotly.

Deviation from the original: instrument display order is the fixed
INSTRUMENT_ORDER (stix, hfr, tnr, epd) intersected with the selection,
rather than the desktop's free Add/Remove insertion order - a documented
simplification (migration plan, section 5). Date-range fields are
pre-filled reactively with the intersection of the selected instruments'
time ranges (max of the starts, min of the ends - the widest range a
combined plot can meaningfully cover), per explicit user request.
"""
import logging
from datetime import datetime

import dash
import dash_bootstrap_components as dbc
from dash import Input, Output, State, callback, dcc, html
from dash.exceptions import PreventUpdate

from sololab.dash_app import plotting
from sololab.dash_app.constants import INSTRUMENT_ORDER, rpw_key
from sololab.dash_app.session_store import session_store
from sololab.dash_app.utils import format_dt_input, parse_dt_input

logger = logging.getLogger(__name__)

dash.register_page(__name__, path="/combined-plot", name="Combined Plot")

INSTRUMENT_LABELS = {"stix": "STIX", "hfr": "RPW-HFR", "tnr": "RPW-TNR", "epd": "EPD"}
STATUS_KEYS = {"stix": "stix", "hfr": "rpw_hfr", "tnr": "rpw_tnr", "epd": "epd"}


def layout(**kwargs):
    return dbc.Container(
        [
            html.H3("Combined Plot", className="mt-3"),
            dbc.Alert(id="cp-alert", is_open=False, dismissable=True, className="mt-2"),
            dbc.Card(
                dbc.CardBody(
                    [
                        dbc.Button("Select instruments to plot", id="cp-select-instruments-btn", color="secondary", outline=True),
                        html.Div(id="cp-selected-instruments-label", className="text-muted small mt-1"),
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
                        dbc.Button("PLOT", id="cp-plot-btn", color="success"),
                    ]
                ),
                className="mb-3",
            ),
            dcc.Graph(id="cp-combined-graph"),
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
    Output("cp-display-instruments", "data"),
    Output("cp-selected-instruments-label", "children"),
    Input("cp-instruments-done-btn", "n_clicks"),
    State("cp-instrument-checklist", "value"),
    prevent_initial_call=True,
)
def close_instrument_modal(n_clicks, selected):
    if not n_clicks:
        raise PreventUpdate
    selected = selected or []
    ordered = [code for code in INSTRUMENT_ORDER if code in selected]
    label = "Selected: " + ", ".join(INSTRUMENT_LABELS[c] for c in ordered) if ordered else "No instruments selected."
    return False, ordered, label


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
            epd_data = session_store.get(sid, "epd_electrons_df" if epd_particle == "Electron" else "epd_protons_df")
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
