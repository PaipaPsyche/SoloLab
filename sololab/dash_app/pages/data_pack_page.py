"""Save/load session data as a .pkl - UI for sololab/dash_app/data_pack.py
(port of MainWindow's "Save/Load Data Pack" buttons)."""
import dash
import dash_bootstrap_components as dbc
from dash import Input, Output, State, callback, dcc, html
from dash.exceptions import PreventUpdate

from sololab.dash_app import data_pack
from sololab.dash_app.utils import decode_upload

dash.register_page(__name__, path="/data-pack", name="Data Pack")


def layout(**kwargs):
    return dbc.Container(
        [
            html.H3("Data Pack", className="mt-3"),
            dbc.Alert(id="dp-alert", is_open=False, dismissable=True, className="mt-2"),
            dbc.Card(
                dbc.CardBody(
                    [
                        html.H5("Save"),
                        html.P(
                            "Save the instrument data currently loaded in this session "
                            "(STIX/RPW-HFR/RPW-TNR/EPD) as a downloadable .pkl file.",
                            className="text-muted",
                        ),
                        dbc.Button("Save Data Pack", id="dp-save-btn", color="primary", disabled=True),
                        dcc.Download(id="dp-download"),
                    ]
                ),
                className="mb-3",
            ),
            dbc.Card(
                dbc.CardBody(
                    [
                        html.H5("Load"),
                        html.P(
                            "Load a previously saved .pkl back into this session "
                            "(overwrites any currently loaded data for the instruments it contains).",
                            className="text-muted",
                        ),
                        dcc.Upload(
                            id="dp-upload",
                            children=html.Div(["Drag and drop or ", html.A("select a .pkl file")]),
                            className="upload-box",
                        ),
                    ]
                )
            ),
        ],
        className="pb-5",
    )


@callback(Output("dp-save-btn", "disabled"), Input("instrument-status-store", "data"))
def toggle_save_btn(status):
    status = status or {}
    return not any(v.get("loaded") for v in status.values())


@callback(
    Output("dp-download", "data"),
    Input("dp-save-btn", "n_clicks"),
    State("session-id", "data"),
    State("instrument-status-store", "data"),
    prevent_initial_call=True,
)
def save_data_pack(n_clicks, sid, status):
    if not n_clicks:
        raise PreventUpdate
    payload = data_pack.build_payload(sid)
    blob = data_pack.payload_to_bytes(payload)
    filename = data_pack.suggested_filename(status)
    return dcc.send_bytes(blob, filename)


@callback(
    Output("instrument-status-store", "data", allow_duplicate=True),
    Output("dp-alert", "children"),
    Output("dp-alert", "color"),
    Output("dp-alert", "is_open"),
    Input("dp-upload", "contents"),
    State("session-id", "data"),
    State("instrument-status-store", "data"),
    prevent_initial_call=True,
)
def load_data_pack(contents, sid, status):
    if not contents:
        raise PreventUpdate
    try:
        blob = decode_upload(contents)
        payload = data_pack.bytes_to_payload(blob)
        new_status = data_pack.apply_payload(sid, payload)
        status = dict(status or {})
        status.update(new_status)
        loaded = [k for k, v in new_status.items() if v.get("loaded")]
        msg = f"Loaded: {', '.join(loaded)}." if loaded else "No instrument data found in this file."
        return status, msg, "success", True
    except Exception as exc:  # noqa: BLE001
        return dash.no_update, str(exc), "danger", True
