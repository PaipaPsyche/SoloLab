import dash
import dash_bootstrap_components as dbc
from dash import html

dash.register_page(__name__, path="/", name="Home")


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
        html.H2("SoloLab", className="mt-3"),
        html.P(
            "Import Solar Orbiter STIX / RPW / EPD data, set plot preferences, "
            "and build a combined quicklook plot.",
            className="text-muted",
        ),
        dbc.Row(
            [
                dbc.Col(
                    _import_card(
                        "Import STIX",
                        "/import/stix",
                        "Upload a STIX FITS spectrogram, optionally subtract a background.",
                    ),
                    md=3,
                    className="mb-3",
                ),
                dbc.Col(
                    _import_card(
                        "Import RPW-HFR",
                        "/import/rpw-hfr",
                        "Upload an RPW-HFR CDF file (L2 or L3).",
                    ),
                    md=3,
                    className="mb-3",
                ),
                dbc.Col(
                    _import_card(
                        "Import RPW-TNR",
                        "/import/rpw-tnr",
                        "Upload an RPW-TNR CDF file (L2).",
                    ),
                    md=3,
                    className="mb-3",
                ),
                dbc.Col(
                    _import_card(
                        "Import EPD",
                        "/import/epd",
                        "Download EPT electron/proton flux for a given day.",
                    ),
                    md=3,
                    className="mb-3",
                ),
            ]
        ),
        dbc.Row(
            [
                dbc.Col(
                    _import_card(
                        "Plot Preferences",
                        "/plot-prefs",
                        "Choose plot type, log scales, energy/frequency ranges per instrument.",
                    ),
                    md=4,
                    className="mb-3",
                ),
                dbc.Col(
                    _import_card(
                        "Combined Plot",
                        "/combined-plot",
                        "Stack the loaded instruments into one quicklook figure.",
                    ),
                    md=4,
                    className="mb-3",
                ),
                dbc.Col(
                    _import_card(
                        "Data Pack",
                        "/data-pack",
                        "Save or load the current session's loaded data as a .pkl file.",
                    ),
                    md=4,
                    className="mb-3",
                ),
            ]
        ),
    ],
    className="pb-5",
)
