"""Import RPW-HFR / RPW-TNR data - port of ImportRpwHfrDialog and
ImportRpwTnrDialog in sololab_app.py.

The two original dialogs are structurally identical (same controls, same
polling options, same call pattern) aside from which sololab.rpw_read
sensor/data_type they target and which session_store keys they write to -
so this factory builds ids/layout/callbacks parametrized by
data_type in {"hfr", "tnr"} instead of duplicating ~180 near-identical
lines per instrument. Unlike the original PyQt5 app (where only HFR has a
"Plot Background" button), this port adds it to TNR too for consistency -
an intentional deviation confirmed with the user.
"""
import dash
import dash_bootstrap_components as dbc
from dash import Input, Output, State, callback, dcc, html
from dash.exceptions import PreventUpdate

from sololab import rpw_read
from sololab.dash_app import plotting
from sololab.dash_app.constants import RPW_POLL_DEFAULT, RPW_POLL_OPTIONS, rpw_key
from sololab.dash_app.session_store import session_store
from sololab.dash_app.utils import bytes_to_tempfile, decode_upload, format_dt_input, parse_dt_input

PREVIEW_FREQUENCY_RANGE = [0, 17000]


def make_rpw_import_layout(data_type):
    label = "RPW-HFR" if data_type == "hfr" else "RPW-TNR"
    p = f"rpw-{data_type}"

    return dbc.Container(
        [
            html.H3(f"Import {label} data", className="mt-3"),
            dbc.Alert(id=f"{p}-alert", is_open=False, dismissable=True, className="mt-2"),
            dbc.Row(
                [
                    dbc.Col(
                        dbc.Card(
                            dbc.CardBody(
                                [
                                    html.H5("Data file"),
                                    dcc.Upload(
                                        id=f"{p}-upload",
                                        children=html.Div(
                                            ["Drag and drop or ", html.A(f"select a {label} CDF file")]
                                        ),
                                        className="upload-box",
                                    ),
                                    html.Div(id=f"{p}-filename", className="text-muted small mt-1"),
                                    html.Hr(),
                                    html.H5("Background"),
                                    dbc.RadioItems(
                                        id=f"{p}-bkg-option",
                                        options=[
                                            {"label": "No background subtraction", "value": 0},
                                            {"label": "Subtract background from a time range", "value": 1},
                                        ],
                                        value=0,
                                    ),
                                    html.Div(
                                        [
                                            dbc.Row(
                                                [
                                                    dbc.Col(
                                                        dbc.Input(
                                                            id=f"{p}-bkg-start",
                                                            type="text",
                                                            placeholder="YYYY-MM-DD HH:MM:SS",
                                                        )
                                                    ),
                                                    dbc.Col(
                                                        dbc.Input(
                                                            id=f"{p}-bkg-end",
                                                            type="text",
                                                            placeholder="YYYY-MM-DD HH:MM:SS",
                                                        )
                                                    ),
                                                ]
                                            )
                                        ],
                                        id=f"{p}-bkg-time-row",
                                        style={"display": "none"},
                                        className="mb-2 mt-1",
                                    ),
                                    html.Label("Background polling function"),
                                    dcc.Dropdown(
                                        id=f"{p}-bkg-poll",
                                        options=RPW_POLL_OPTIONS,
                                        value=RPW_POLL_DEFAULT,
                                        clearable=False,
                                    ),
                                    html.Hr(),
                                    dbc.ButtonGroup(
                                        [
                                            dbc.Button(
                                                "Preview Data", id=f"{p}-preview-btn", disabled=True, color="secondary"
                                            ),
                                            dbc.Button(
                                                "Preview with Background Subtraction",
                                                id=f"{p}-preview-bkg-btn",
                                                disabled=True,
                                                color="secondary",
                                            ),
                                            dbc.Button(
                                                "Plot Background", id=f"{p}-plot-bkg-btn", disabled=True, color="secondary"
                                            ),
                                        ],
                                        vertical=True,
                                        className="w-100 mb-2",
                                    ),
                                    dbc.Button(
                                        "LOAD", id=f"{p}-load-btn", disabled=True, color="success", className="w-100"
                                    ),
                                ]
                            )
                        ),
                        md=4,
                    ),
                    dbc.Col(dcc.Graph(id=f"{p}-preview-graph"), md=8),
                ]
            ),
            dbc.Modal(
                [
                    dbc.ModalHeader(dbc.ModalTitle(f"{label} Background")),
                    dbc.ModalBody(dcc.Graph(id=f"{p}-bkg-graph")),
                ],
                id=f"{p}-bkg-modal",
                is_open=False,
                size="lg",
            ),
        ],
        className="pb-5",
    )


def register_rpw_import_callbacks(data_type):
    p = f"rpw-{data_type}"
    status_key = f"rpw_{data_type}"  # matches instrument-status-store's "rpw_hfr" / "rpw_tnr" keys

    def _load_psd(sid):
        filename, data = session_store.get(sid, rpw_key(data_type, "file_bytes"))
        path = bytes_to_tempfile(data, filename)
        rpw_data = rpw_read.rpw_get_data(path)
        return rpw_read.rpw_create_PSD(rpw_data, which_freqs="non_zero")

    def _load_psd_with_bkg(sid, bkg_start, bkg_end, poll):
        filename, data = session_store.get(sid, rpw_key(data_type, "file_bytes"))
        path = bytes_to_tempfile(data, filename)
        rpw_data = rpw_read.rpw_get_data(path)
        start = parse_dt_input(bkg_start)
        end = parse_dt_input(bkg_end)
        return rpw_read.rpw_create_PSD(
            rpw_data, which_freqs="non_zero", rpw_bkg_interval=(start, end), bkg_poll_function=poll
        )

    @callback(
        Output(f"{p}-filename", "children"),
        Output(f"{p}-preview-btn", "disabled"),
        Output(f"{p}-preview-bkg-btn", "disabled"),
        Output(f"{p}-load-btn", "disabled"),
        Input(f"{p}-upload", "contents"),
        State(f"{p}-upload", "filename"),
        State("session-id", "data"),
        prevent_initial_call=True,
    )
    def on_upload(contents, filename, sid):
        if not contents:
            return "", True, True, True
        session_store.set(sid, rpw_key(data_type, "file_bytes"), (filename, decode_upload(contents)))
        return f"Selected: {filename}", False, False, False

    @callback(
        Output(f"{p}-bkg-time-row", "style"),
        Input(f"{p}-bkg-option", "value"),
    )
    def toggle_bkg_row(bkg_option):
        return {"display": "block"} if bkg_option == 1 else {"display": "none"}

    @callback(
        Output(f"{p}-preview-graph", "figure"),
        Output(f"{p}-bkg-start", "value"),
        Output(f"{p}-bkg-end", "value"),
        Output(f"{p}-alert", "children"),
        Output(f"{p}-alert", "color"),
        Output(f"{p}-alert", "is_open"),
        Input(f"{p}-preview-btn", "n_clicks"),
        State("session-id", "data"),
        prevent_initial_call=True,
    )
    def preview(n_clicks, sid):
        if not n_clicks:
            raise PreventUpdate
        try:
            psd = _load_psd(sid)
            session_store.set(sid, rpw_key(data_type, "psd_final"), psd)
            fig = plotting.rpw_psd_figure(psd, frequency_range=PREVIEW_FREQUENCY_RANGE)
            start_str = format_dt_input(min(psd["time"]))
            end_str = format_dt_input(max(psd["time"]))
            return fig, start_str, end_str, "", "success", False
        except Exception as exc:  # noqa: BLE001
            return dash.no_update, dash.no_update, dash.no_update, str(exc), "danger", True

    @callback(
        Output(f"{p}-preview-graph", "figure", allow_duplicate=True),
        Output(f"{p}-plot-bkg-btn", "disabled"),
        Output(f"{p}-alert", "children", allow_duplicate=True),
        Output(f"{p}-alert", "color", allow_duplicate=True),
        Output(f"{p}-alert", "is_open", allow_duplicate=True),
        Input(f"{p}-preview-bkg-btn", "n_clicks"),
        State(f"{p}-bkg-start", "value"),
        State(f"{p}-bkg-end", "value"),
        State(f"{p}-bkg-poll", "value"),
        State("session-id", "data"),
        prevent_initial_call=True,
    )
    def preview_with_bkg(n_clicks, bkg_start, bkg_end, poll, sid):
        if not n_clicks:
            raise PreventUpdate
        try:
            psd = _load_psd_with_bkg(sid, bkg_start, bkg_end, poll)
            session_store.set(sid, rpw_key(data_type, "psd_final"), psd)
            fig = plotting.rpw_psd_figure(psd, frequency_range=PREVIEW_FREQUENCY_RANGE)
            start = parse_dt_input(bkg_start)
            end = parse_dt_input(bkg_end)
            fig.add_vline(x=start, line_color="red")
            fig.add_vline(x=end, line_color="red")
            return fig, False, "", "success", False
        except Exception as exc:  # noqa: BLE001
            return dash.no_update, dash.no_update, str(exc), "danger", True

    @callback(
        Output(f"{p}-bkg-graph", "figure"),
        Output(f"{p}-bkg-modal", "is_open"),
        Input(f"{p}-plot-bkg-btn", "n_clicks"),
        State("session-id", "data"),
        prevent_initial_call=True,
    )
    def plot_background(n_clicks, sid):
        if not n_clicks:
            raise PreventUpdate
        psd = session_store.get(sid, rpw_key(data_type, "psd_final"))
        return plotting.rpw_bkg_figure(psd), True

    @callback(
        Output("instrument-status-store", "data", allow_duplicate=True),
        Output(f"{p}-alert", "children", allow_duplicate=True),
        Output(f"{p}-alert", "color", allow_duplicate=True),
        Output(f"{p}-alert", "is_open", allow_duplicate=True),
        Input(f"{p}-load-btn", "n_clicks"),
        State(f"{p}-bkg-option", "value"),
        State(f"{p}-bkg-start", "value"),
        State(f"{p}-bkg-end", "value"),
        State(f"{p}-bkg-poll", "value"),
        State("session-id", "data"),
        State("instrument-status-store", "data"),
        prevent_initial_call=True,
    )
    def load(n_clicks, bkg_option, bkg_start, bkg_end, poll, sid, status):
        if not n_clicks:
            raise PreventUpdate
        try:
            if bkg_option == 1:
                psd = _load_psd_with_bkg(sid, bkg_start, bkg_end, poll)
            else:
                psd = _load_psd(sid)
            session_store.set(sid, rpw_key(data_type, "psd_final"), psd)
            session_store.set(sid, rpw_key(data_type, "meta"), {"bkg_option": bkg_option, "bkg_poll_function": poll})
            status = dict(status or {})
            status[status_key] = {
                "loaded": True,
                "min_time": format_dt_input(min(psd["time"])),
                "max_time": format_dt_input(max(psd["time"])),
                "bkg_enabled": bkg_option == 1,
            }
            return status, f"{data_type.upper()} data loaded.", "success", True
        except Exception as exc:  # noqa: BLE001
            return dash.no_update, str(exc), "danger", True
