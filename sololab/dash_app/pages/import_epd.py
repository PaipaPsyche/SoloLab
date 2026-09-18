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
from dash import Input, Output, State, callback, dcc, html
from dash.exceptions import PreventUpdate
from solo_epd_loader import epd_load

from sololab.dash_app import plotting
from sololab.dash_app.constants import (
    EPD_CACHE_DIR,
    EPD_PARTICLE_OPTIONS,
    EPD_PREVIEW_CHANNELS,
    EPD_RESAMPLE_DEFAULT,
    EPD_RESAMPLE_OPTIONS,
)
from sololab.dash_app.session_store import session_store

logger = logging.getLogger(__name__)

dash.register_page(__name__, path="/import/epd", name="Import EPD")


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
                                    [
                                        dbc.Button("Plot Preview", id="epd-preview-btn", disabled=True, color="secondary"),
                                        dbc.Button("LOAD", id="epd-load-btn", disabled=True, color="success"),
                                    ],
                                    vertical=True,
                                    className="w-100",
                                ),
                            ]
                        )
                    ),
                    md=4,
                ),
                dbc.Col(dcc.Graph(id="epd-preview-graph"), md=8),
            ]
        ),
    ],
    className="pb-5",
)


@callback(
    Output("epd-preview-btn", "disabled"),
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
        return False, False, f"EPD data downloaded for {date_str}.", "success", True
    except Exception as exc:  # noqa: BLE001
        logger.exception("Unhandled error in callback")
        return True, True, str(exc), "danger", True


@callback(
    Output("epd-preview-graph", "figure"),
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
        return fig, "", "success", False
    except Exception as exc:  # noqa: BLE001
        logger.exception("Unhandled error in callback")
        return dash.no_update, str(exc), "danger", True


@callback(
    Output("instrument-status-store", "data", allow_duplicate=True),
    Output("epd-alert", "children", allow_duplicate=True),
    Output("epd-alert", "color", allow_duplicate=True),
    Output("epd-alert", "is_open", allow_duplicate=True),
    Input("epd-load-btn", "n_clicks"),
    State("epd-date", "date"),
    State("epd-particle", "value"),
    State("epd-resample", "value"),
    State("session-id", "data"),
    State("instrument-status-store", "data"),
    prevent_initial_call=True,
)
def epd_load_click(n_clicks, date_str, particle, resample, sid, status):
    if not n_clicks:
        raise PreventUpdate
    if not session_store.has(sid, "epd_energies"):
        return dash.no_update, "Download EPD data first.", "danger", True

    session_store.set(sid, "epd_meta", {"date": date_str, "particle": particle, "resample": resample})
    status = dict(status or {})
    status["epd"] = {"loaded": True, "date": date_str, "particle": particle, "resample": resample}
    return status, "EPD data loaded.", "success", True
