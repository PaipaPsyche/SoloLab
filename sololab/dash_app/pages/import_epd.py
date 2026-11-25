"""Import EPD (EPT) data - port of ImportEpdDialog in sololab_app.py.

Unlike STIX/RPW, this isn't a local-file upload: solo_epd_loader.epd_load
downloads (and caches) the requested day's CDF from the SOAR archive itself
when autodownload=True. The desktop dialog let the user pick a local
download folder; that has no web equivalent (the server, not the user's
machine, does the download), so this page drops that field and always uses
a fixed, shared server-side cache directory (EPD_CACHE_DIR) - the data is
public per-day science data, so sharing the cache across sessions/users is
fine and avoids re-downloading the same day repeatedly.
"""
import logging
from datetime import date, datetime

import dash
import dash_bootstrap_components as dbc
import numpy as np
from dash import Input, Output, State, callback, dcc, html
from dash.exceptions import PreventUpdate
from solo_epd_loader import epd_load

from sololab.dash_app import plotting
from sololab.dash_app.constants import (
    EPD_CACHE_DIR,
    EPD_PARTICLE_OPTIONS,
    EPD_POLL_DEFAULT,
    EPD_POLL_OPTIONS,
    EPD_PREVIEW_CHANNELS,
    EPD_RESAMPLE_DEFAULT,
    EPD_RESAMPLE_OPTIONS,
)
from sololab.dash_app.session_store import session_store
from sololab.dash_app.utils import format_dt_input, parse_dt_input
from sololab.values import get_poll_func

logger = logging.getLogger(__name__)

dash.register_page(__name__, path="/import/epd", name="Import EPD")


def _epd_bkg_subtract(df, particle, bkg_start, bkg_end, poll_function):
    """Time-range background subtraction for EPD flux, per energy channel -
    same logic as STIX/RPW: poll each channel's flux over [bkg_start,
    bkg_end], subtract it from every row, and keep the poll result + std
    per channel so the background can be plotted (mirrors
    stix_read.stix_remove_bkg_counts / rpw_read.rpw_create_PSD's bkg
    tracking). Floored at 0 like the other two instruments (flux can't be
    negative) - matters most with a poll function like "max", which can
    subtract a value bigger than nearly every other point in a spiky
    timeseries; plotting.epd_flux_figure's own 0.1 floor is separate, only
    needed because 0 itself can't be placed on the always-log Y-axis."""
    flux_key = f"{particle}_Flux"
    flux = df[flux_key]
    mask = (df.index >= bkg_start) & (df.index <= bkg_end)
    window = flux.loc[mask].values
    func = get_poll_func(poll_function)
    bkg = func(window, axis=0)
    bkg_std = np.std(window, axis=0)
    df_out = df.copy()
    df_out[flux_key] = np.clip(flux.values - bkg, 0, None)
    return df_out, bkg, bkg_std


def layout(**kwargs):
    # a function (not a static module-level value) so date.today() is
    # re-evaluated on every page visit instead of being frozen at server
    # start / first import.
    return _layout()


def _layout():
    return dbc.Container(
    [
        html.H3("Import EPD data", className="mt-3"),
        dbc.Alert(id="epd-alert", is_open=False, dismissable=True, className="mt-2"),
        dbc.Row(
            [
                dbc.Col(
                    dbc.Card(
                        dbc.CardBody(
                            [
                                html.Label("Observation date"),
                                dcc.DatePickerSingle(
                                    id="epd-date",
                                    date=date.today().isoformat(),
                                    display_format="YYYY-MM-DD",
                                    className="d-block mb-2",
                                ),
                                html.Label("Particle type"),
                                dcc.Dropdown(
                                    id="epd-particle",
                                    options=EPD_PARTICLE_OPTIONS,
                                    value="Electron",
                                    clearable=False,
                                    className="mb-2",
                                ),
                                html.Label("Resample"),
                                dcc.Dropdown(
                                    id="epd-resample",
                                    options=EPD_RESAMPLE_OPTIONS,
                                    value=EPD_RESAMPLE_DEFAULT,
                                    clearable=False,
                                    className="mb-2",
                                ),
                                html.Hr(),
                                dbc.Spinner(
                                    dbc.Button(
                                        "Download EPD Data", id="epd-download-btn", color="primary", className="w-100"
                                    ),
                                    color="primary",
                                ),
                                html.Hr(),
                                dbc.ButtonGroup(
                                    [dbc.Button("Plot Preview", id="epd-preview-btn", disabled=True, color="secondary")],
                                    className="w-100 mb-2",
                                ),
                                html.Div(
                                    [
                                        html.H5("Background"),
                                        dbc.Checkbox(
                                            id="epd-bkg-time-enabled",
                                            label="Subtract background from a time range",
                                            value=False,
                                        ),
                                        html.Div(
                                            [
                                                dbc.Row(
                                                    [
                                                        dbc.Col(dbc.Input(id="epd-bkg-start", type="text", placeholder="YYYY-MM-DD HH:MM:SS")),
                                                        dbc.Col(dbc.Input(id="epd-bkg-end", type="text", placeholder="YYYY-MM-DD HH:MM:SS")),
                                                    ]
                                                )
                                            ],
                                            id="epd-bkg-time-row",
                                            style={"display": "none"},
                                            className="mb-2 mt-1",
                                        ),
                                        html.Label(
                                            [
                                                "Background polling function ",
                                                html.Span("ⓘ", id="epd-bkg-poll-info", style={"cursor": "help", "color": "#6c757d"}),
                                            ]
                                        ),
                                        dbc.Tooltip(
                                            "\"max\"/\"min\" are unreliable for spiky count data - a single outlier "
                                            "in the background window becomes the whole subtracted background. "
                                            "\"mean\" or \"median\" are more robust.",
                                            target="epd-bkg-poll-info",
                                            placement="right",
                                        ),
                                        dcc.Dropdown(
                                            id="epd-bkg-poll",
                                            options=EPD_POLL_OPTIONS,
                                            value=EPD_POLL_DEFAULT,
                                            clearable=False,
                                        ),
                                    ],
                                    id="epd-bkg-section",
                                    style={"display": "none"},
                                ),
                                html.Hr(),
                                dbc.ButtonGroup(
                                    [
                                        dbc.Button(
                                            "Preview with Background Subtraction",
                                            id="epd-preview-bkg-btn",
                                            disabled=True,
                                            color="secondary",
                                        ),
                                        dbc.Button("Plot Background", id="epd-plot-bkg-btn", disabled=True, color="secondary"),
                                    ],
                                    vertical=True,
                                    className="w-100 mb-2",
                                ),
                                dbc.Button("LOAD", id="epd-load-btn", disabled=True, color="success", className="w-100"),
                            ]
                        )
                    ),
                    md=4,
                ),
                dbc.Col(dbc.Spinner(dcc.Graph(id="epd-preview-graph"), color="secondary"), md=8),
            ]
        ),
        dbc.Modal(
            [
                dbc.ModalHeader(dbc.ModalTitle("EPD Background")),
                dbc.ModalBody(dbc.Spinner(dcc.Graph(id="epd-bkg-graph"), color="secondary")),
            ],
            id="epd-bkg-modal",
            is_open=False,
            size="lg",
        ),
    ],
    className="pb-5",
)


@callback(
    Output("epd-preview-btn", "disabled"),
    Output("epd-preview-bkg-btn", "disabled"),
    Output("epd-load-btn", "disabled"),
    Output("epd-alert", "children"),
    Output("epd-alert", "color"),
    Output("epd-alert", "is_open"),
    Input("epd-download-btn", "n_clicks"),
    State("epd-date", "date"),
    State("session-id", "data"),
    prevent_initial_call=True,
)
def epd_download(n_clicks, date_str, sid):
    if not n_clicks:
        raise PreventUpdate
    try:
        date_int = int(date_str.replace("-", ""))
        df_protons, df_electrons, energies = epd_load(
            sensor="ept",
            level="l2",
            startdate=date_int,
            enddate=date_int,
            viewing="sun",
            path=EPD_CACHE_DIR,
            autodownload=True,
        )
        # epd_load returns ([], [], []) - empty lists, not DataFrames -
        # when no data file exists for the requested date/viewing, instead
        # of raising. Treat that as a real error rather than a silent
        # "success" that would crash later on df.index.
        if isinstance(df_electrons, list) or isinstance(df_protons, list):
            raise ValueError(f"No EPD data available for {date_str} (viewing='sun').")
        session_store.set(sid, "epd_protons_df", df_protons)
        session_store.set(sid, "epd_electrons_df", df_electrons)
        session_store.set(sid, "epd_energies", energies)
        return False, False, False, f"EPD data downloaded for {date_str}.", "success", True
    except Exception as exc:  # noqa: BLE001
        logger.exception("Unhandled error in callback")
        return True, True, True, str(exc), "danger", True


@callback(
    Output("epd-bkg-time-row", "style"),
    Input("epd-bkg-time-enabled", "value"),
)
def toggle_epd_bkg_time_row(enabled):
    return {"display": "block"} if enabled else {"display": "none"}


@callback(
    Output("epd-preview-graph", "figure"),
    Output("epd-bkg-start", "value"),
    Output("epd-bkg-end", "value"),
    Output("epd-bkg-section", "style"),
    Output("epd-alert", "children", allow_duplicate=True),
    Output("epd-alert", "color", allow_duplicate=True),
    Output("epd-alert", "is_open", allow_duplicate=True),
    Input("epd-preview-btn", "n_clicks"),
    State("epd-particle", "value"),
    State("epd-resample", "value"),
    State("epd-date", "date"),
    State("session-id", "data"),
    prevent_initial_call=True,
)
def epd_preview(n_clicks, particle, resample, date_str, sid):
    if not n_clicks:
        raise PreventUpdate
    try:
        df_protons = session_store.get(sid, "epd_protons_df")
        df_electrons = session_store.get(sid, "epd_electrons_df")
        energies = session_store.get(sid, "epd_energies")
        epd_data = df_electrons if particle == "Electron" else df_protons

        day = datetime.strptime(date_str, "%Y-%m-%d")
        date_range = (day.replace(hour=0, minute=0, second=0), day.replace(hour=23, minute=59, second=59))

        fig = plotting.epd_flux_figure(
            epd_data,
            energies,
            particle=particle,
            channels=EPD_PREVIEW_CHANNELS,
            date_range=date_range,
            resample=resample,
        )
        return fig, format_dt_input(date_range[0]), format_dt_input(date_range[1]), {"display": "block"}, "", "success", False
    except Exception as exc:  # noqa: BLE001
        logger.exception("Unhandled error in callback")
        return dash.no_update, dash.no_update, dash.no_update, dash.no_update, str(exc), "danger", True


def _epd_selected_df(sid, particle):
    df_protons = session_store.get(sid, "epd_protons_df")
    df_electrons = session_store.get(sid, "epd_electrons_df")
    return df_electrons if particle == "Electron" else df_protons


@callback(
    Output("epd-preview-graph", "figure", allow_duplicate=True),
    Output("epd-plot-bkg-btn", "disabled"),
    Output("epd-alert", "children", allow_duplicate=True),
    Output("epd-alert", "color", allow_duplicate=True),
    Output("epd-alert", "is_open", allow_duplicate=True),
    Input("epd-preview-bkg-btn", "n_clicks"),
    State("epd-particle", "value"),
    State("epd-resample", "value"),
    State("epd-bkg-start", "value"),
    State("epd-bkg-end", "value"),
    State("epd-bkg-poll", "value"),
    State("session-id", "data"),
    prevent_initial_call=True,
)
def epd_preview_with_bkg(n_clicks, particle, resample, bkg_start, bkg_end, poll, sid):
    if not n_clicks:
        raise PreventUpdate
    try:
        df = _epd_selected_df(sid, particle)
        energies = session_store.get(sid, "epd_energies")
        start, end = parse_dt_input(bkg_start), parse_dt_input(bkg_end)
        df_sub, bkg, bkg_std = _epd_bkg_subtract(df, particle, start, end, poll)
        session_store.set(sid, "epd_background", bkg)
        session_store.set(sid, "epd_background_std", bkg_std)

        fig = plotting.epd_flux_figure(
            df_sub, energies, particle=particle, channels=EPD_PREVIEW_CHANNELS,
            date_range=(df.index.min(), df.index.max()), resample=resample,
        )
        fig.add_vline(x=start, line_color="red")
        fig.add_vline(x=end, line_color="red")
        return fig, False, "", "success", False
    except Exception as exc:  # noqa: BLE001
        logger.exception("Unhandled error in callback")
        return dash.no_update, dash.no_update, str(exc), "danger", True


@callback(
    Output("epd-bkg-graph", "figure"),
    Output("epd-bkg-modal", "is_open"),
    Input("epd-plot-bkg-btn", "n_clicks"),
    State("epd-particle", "value"),
    State("session-id", "data"),
    prevent_initial_call=True,
)
def epd_plot_background(n_clicks, particle, sid):
    if not n_clicks:
        raise PreventUpdate
    bkg = session_store.get(sid, "epd_background")
    bkg_std = session_store.get(sid, "epd_background_std")
    energies = session_store.get(sid, "epd_energies")
    return plotting.epd_bkg_figure(bkg, bkg_std, energies, particle), True


@callback(
    Output("instrument-status-store", "data", allow_duplicate=True),
    Output("epd-alert", "children", allow_duplicate=True),
    Output("epd-alert", "color", allow_duplicate=True),
    Output("epd-alert", "is_open", allow_duplicate=True),
    Input("epd-load-btn", "n_clicks"),
    State("epd-date", "date"),
    State("epd-particle", "value"),
    State("epd-resample", "value"),
    State("epd-bkg-time-enabled", "value"),
    State("epd-bkg-start", "value"),
    State("epd-bkg-end", "value"),
    State("epd-bkg-poll", "value"),
    State("session-id", "data"),
    State("instrument-status-store", "data"),
    prevent_initial_call=True,
)
def epd_load_click(n_clicks, date_str, particle, resample, bkg_enabled, bkg_start, bkg_end, poll, sid, status):
    if not n_clicks:
        raise PreventUpdate
    if not session_store.has(sid, "epd_energies"):
        return dash.no_update, "Download EPD data first.", "danger", True

    df = _epd_selected_df(sid, particle)
    if bkg_enabled:
        start, end = parse_dt_input(bkg_start), parse_dt_input(bkg_end)
        df, bkg, bkg_std = _epd_bkg_subtract(df, particle, start, end, poll)
        session_store.set(sid, "epd_background", bkg)
        session_store.set(sid, "epd_background_std", bkg_std)

    final_key = "epd_electrons_final" if particle == "Electron" else "epd_protons_final"
    session_store.set(sid, final_key, df)
    session_store.set(
        sid, "epd_meta",
        {"date": date_str, "particle": particle, "resample": resample, "bkg_enabled": bool(bkg_enabled), "bkg_poll_function": poll},
    )
    status = dict(status or {})
    status["epd"] = {
        "loaded": True, "date": date_str, "particle": particle, "resample": resample, "bkg_enabled": bool(bkg_enabled),
    }
    return status, "EPD data loaded.", "success", True
