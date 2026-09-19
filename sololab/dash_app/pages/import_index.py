"""Import Data landing page - the sidebar's "Import Data" button lands here
first so the user picks an instrument, then goes to that instrument's
existing import form (moved out of home.py, which no longer needs to repeat
this now that the sidebar covers navigation)."""
import dash
import dash_bootstrap_components as dbc
from dash import html

dash.register_page(__name__, path="/import", name="Import Data")


def _import_card(title, href, description):
    return dbc.Card(
        dbc.CardBody(
            [
                html.H5(title, className="card-title"),
                html.P(description, className="card-text text-muted"),
                dbc.Button("Open", href=href, color="primary", outline=True, size="sm"),
            ]
        ),
        className="h-100",
    )


layout = dbc.Container(
    [
        html.H3("Import Data", className="mt-3"),
        html.P("Choose which instrument to import.", className="text-muted"),
        dbc.Row(
            [
                dbc.Col(
                    _import_card(
                        "STIX",
                        "/import/stix",
                        "Upload a STIX FITS spectrogram, optionally subtract a background.",
                    ),
                    md=3,
                    className="mb-3",
                ),
                dbc.Col(
                    _import_card(
                        "RPW-HFR",
                        "/import/rpw-hfr",
                        "Upload an RPW-HFR CDF file (L2 or L3).",
                    ),
                    md=3,
                    className="mb-3",
                ),
                dbc.Col(
                    _import_card(
                        "RPW-TNR",
                        "/import/rpw-tnr",
                        "Upload an RPW-TNR CDF file (L2).",
                    ),
                    md=3,
                    className="mb-3",
                ),
                dbc.Col(
                    _import_card(
                        "EPD",
                        "/import/epd",
                        "Download EPT electron/proton flux for a given day.",
                    ),
                    md=3,
                    className="mb-3",
                ),
            ]
        ),
    ],
    className="pb-5",
)
