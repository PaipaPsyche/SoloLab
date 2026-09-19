import dash
import dash_bootstrap_components as dbc
from dash import html

dash.register_page(__name__, path="/", name="Home")


def _instrument_card(title, body):
    return dbc.Card(dbc.CardBody([html.H5(title, className="card-title"), html.P(body, className="card-text")]), className="h-100")


layout = dbc.Container(
    [
        html.Img(src="/assets/sololab_banner.png", style={"maxWidth": "480px", "width": "100%"}, className="mt-3 mb-2"),
        html.P(
            "A tool for multi-instrument analysis of Solar Orbiter data - correlating X-ray, "
            "radio, and in-situ particle measurements from the same event.",
            className="text-muted",
        ),
        html.Hr(),
        html.H4("Solar Orbiter"),
        html.P(
            [
                html.A("Solar Orbiter", href="https://www.esa.int/Science_Exploration/Space_Science/Solar_Orbiter", target="_blank"),
                " is an ESA/NASA mission launched in February 2020 to study the Sun and inner "
                "heliosphere at closer range than any previous mission with a full instrument "
                "suite, combining remote-sensing telescopes with in-situ particle and field "
                "detectors. It carries 10 instruments in total; SoloLab focuses on three of them.",
            ]
        ),
        dbc.Row(
            [
                dbc.Col(
                    _instrument_card(
                        "STIX",
                        "Spectrometer Telescope for Imaging X-rays - images and measures the energy "
                        "spectrum of X-rays from solar flares, tracing accelerated electrons and "
                        "flare energy release.",
                    ),
                    md=4,
                    className="mb-3",
                ),
                dbc.Col(
                    _instrument_card(
                        "RPW",
                        "Radio and Plasma Waves instrument - measures electric and magnetic fields; "
                        "SoloLab uses its HFR and TNR receivers to track radio bursts (e.g. Type III) "
                        "and in-situ plasma waves.",
                    ),
                    md=4,
                    className="mb-3",
                ),
                dbc.Col(
                    _instrument_card(
                        "EPD",
                        "Energetic Particles Detector - measures in-situ fluxes of energetic "
                        "electrons, protons and ions (via its EPT sensor) across a range of "
                        "energies, tracing particle acceleration and transport from the Sun.",
                    ),
                    md=4,
                    className="mb-3",
                ),
            ]
        ),
        html.Hr(),
        html.H4("What this app does"),
        html.Ul(
            [
                html.Li(
                    [
                        html.B("Import"),
                        " - STIX (upload a FITS file, or search/download directly from the STIX "
                        "Data Center), RPW (upload a CDF, or download directly from CDAWeb), EPD "
                        "(auto-downloaded for a given day via solo-epd-loader).",
                    ]
                ),
                html.Li(
                    [
                        html.B("Background subtraction"),
                        " - pick a quiet time interval (or, for STIX, a dedicated background file) "
                        "and a polling statistic (mean/median/min/max/percentile). The app subtracts "
                        "a per-channel/per-frequency background and keeps track of its spread (std) "
                        "so it can be shown as error bars.",
                    ]
                ),
                html.Li(
                    [
                        html.B("Visualization"),
                        " - spectrograms (time vs. energy/frequency, colored by intensity) or "
                        "per-channel light curves for each instrument individually, in Plot Data.",
                    ]
                ),
                html.Li(
                    [
                        html.B("Combined Plot"),
                        " - stack any combination of loaded instruments into one multi-panel figure "
                        "sharing a single time axis, with a user-controlled panel order - the main "
                        "tool for correlating an event across X-ray, radio, and particle signatures.",
                    ]
                ),
                html.Li(
                    [
                        html.B("Data Pack"),
                        " - save everything currently loaded in a session to a file, and reload it "
                        "later without re-downloading or re-importing.",
                    ]
                ),
            ]
        ),
        dbc.Button("Import Data", href="/import", color="primary", className="mt-2"),
    ],
    className="pb-5",
)
