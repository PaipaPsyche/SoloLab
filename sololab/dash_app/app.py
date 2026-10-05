"""SoloLab Dash app entry point.

Run locally with:
    python -m sololab.dash_app.app

Deploy with:
    gunicorn sololab.dash_app.app:server
"""
import logging
import os
import threading
import uuid

import dash
import dash_bootstrap_components as dbc
import diskcache
from dash import Input, Output, State, callback, dcc, html, DiskcacheManager

from sololab.dash_app.constants import DEFAULT_INSTRUMENT_STATUS, DEFAULT_PLOT_PREFS
from sololab.dash_app.session_store import session_store
from sololab.dash_app.utils import status_badge_content

# Every page module's `logger.exception(...)` calls (in the broad
# `except Exception` blocks around user actions) rely on this being
# configured - previously nothing configured a handler, so unexpected
# errors were shown to the user but never recorded anywhere server-side.
# SOLOLAB_LOG_LEVEL defaults to INFO; set it to DEBUG/WARNING as needed.
logging.basicConfig(
    level=os.environ.get("SOLOLAB_LOG_LEVEL", "INFO").upper(),
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)

# Backs Dash's background callbacks (currently only STIX Imaging's
# run_stix_imaging, for Sequence-mode progress reporting) - a separate
# diskcache.Cache instance from session_store.py's, matching Dash's own
# documented DiskcacheManager pattern (results/progress cache, not
# per-session app data - different lifetime/TTL semantics, kept apart).
_background_callback_cache_dir = os.environ.get(
    "SOLOLAB_CALLBACK_CACHE_DIR",
    os.path.join(os.path.dirname(__file__), ".callback_cache"),
)
background_callback_manager = DiskcacheManager(diskcache.Cache(_background_callback_cache_dir))

app = dash.Dash(
    __name__,
    use_pages=True,
    external_stylesheets=[dbc.themes.FLATLY],
    suppress_callback_exceptions=True,
    title="SoloLab",
    background_callback_manager=background_callback_manager,
)
server = app.server

# Reject oversized uploads before Flask buffers the whole request body in
# memory (dcc.Upload base64-encodes the file client-side, so this is the
# request size, ~1.33x the raw file size). Configurable since some CDF
# files legitimately run into the tens of MB.
server.config["MAX_CONTENT_LENGTH"] = int(os.environ.get("SOLOLAB_MAX_UPLOAD_MB", 200)) * 1024 * 1024


def _start_session_cache_sweeper(interval_seconds=3600):
    """session_store.evict_stale() was defined but never called anywhere -
    diskcache only expires entries lazily on access, so .session_cache
    would otherwise accumulate expired-but-unswept blobs forever on a
    long-running server. Runs as a daemon thread so it never blocks
    shutdown."""

    def _sweep_forever():
        while True:
            threading.Event().wait(interval_seconds)
            try:
                session_store.evict_stale()
            except Exception:  # noqa: BLE001 - best-effort background sweep
                pass

    threading.Thread(target=_sweep_forever, daemon=True).start()


_start_session_cache_sweeper()

STATUS_KEYS = ["stix", "rpw_hfr", "rpw_tnr", "epd"]
# Sidebar's 2x2 status grid order, per the design brief: RPW-HFR, RPW-TNR / STIX, EPD.
# (Independent of STATUS_KEYS, which only orders the callback's Output list below.)
STATUS_GRID_ORDER = ["rpw_hfr", "rpw_tnr", "stix", "epd"]

FOOTER_LINKS = [
    ("STIX Data Center", "https://datacenter.stix.i4ds.net/"),
    ("RPW Data Center", "https://rpw-datacenter.obspm.fr"),
    ("GitHub", "https://github.com/PaipaPsyche/SoloLab"),
    ("Contact", "mailto:david.paipa@obspm.fr"),
]


def _status_grid():
    return html.Div(
        [
            html.Div(
                dbc.Badge(id=f"status-badge-{key.replace('_', '-')}", color="danger", className="status-badge"),
                id=f"status-cell-{key.replace('_', '-')}",
                className="status-cell",
            )
            for key in STATUS_GRID_ORDER
        ],
        className="status-grid",
    )


topbar = dbc.Navbar(
    dbc.Container(
        [
            dbc.NavbarBrand(
                [
                    html.Img(src="/assets/sololab_icon.png", height="36px", className="me-2"),
                    "SoloLab",
                ],
                href="/",
                className="d-flex align-items-center",
            )
        ],
        fluid=True,
    ),
    dark=True,
    className="topbar",
)

sidebar = html.Div(
    [
        html.Div(
            [
                dbc.Button("Import Data", href="/import", color="primary", className="w-100 mb-2"),
                _status_grid(),
            ],
            className="sidebar-section",
        ),
        html.Hr(),
        html.Div(
            [
                html.Div("Data Pack", className="sidebar-section-title"),
                dbc.Button("Save / Load", href="/data-pack", color="secondary", outline=True, className="w-100"),
            ],
            className="sidebar-section",
        ),
        html.Hr(),
        html.Div(
            [
                html.Div("Plot", className="sidebar-section-title"),
                dbc.Nav(
                    [
                        dbc.NavLink("Plot Data", href="/plot-prefs", active="exact"),
                        dbc.NavLink("Combined Plot", href="/combined-plot", active="exact"),
                    ],
                    vertical=True,
                    pills=True,
                ),
            ],
            className="sidebar-section",
        ),
        html.Hr(),
        html.Div(
            [
                html.Div("Imaging", className="sidebar-section-title"),
                dbc.Nav(
                    [dbc.NavLink("STIX Imaging", href="/imaging/stix", active="exact")],
                    vertical=True,
                    pills=True,
                ),
            ],
            className="sidebar-section",
        ),
    ],
    className="sidebar",
)

bottombar = html.Div(
    [dbc.NavLink(label, href=href, target="_blank", className="d-inline-block me-3") for label, href in FOOTER_LINKS],
    className="bottom-bar",
)

app.layout = html.Div(
    [
        dcc.Location(id="url"),
        dcc.Store(id="session-id", storage_type="session"),
        dcc.Store(id="plot-prefs-store", storage_type="session", data=DEFAULT_PLOT_PREFS),
        dcc.Store(id="instrument-status-store", storage_type="session", data=DEFAULT_INSTRUMENT_STATUS),
        topbar,
        html.Div(
            [sidebar, html.Div(dbc.Container(dash.page_container, fluid=True), className="main-content")],
            className="app-body",
        ),
        bottombar,
    ],
    className="app-shell",
)


@callback(
    Output("session-id", "data"),
    Input("url", "pathname"),
    State("session-id", "data"),
)
def ensure_session_id(_pathname, current):
    return current or uuid.uuid4().hex


@callback(
    [Output(f"status-badge-{key.replace('_', '-')}", "children") for key in STATUS_KEYS]
    + [Output(f"status-badge-{key.replace('_', '-')}", "color") for key in STATUS_KEYS]
    + [Output(f"status-cell-{key.replace('_', '-')}", "title") for key in STATUS_KEYS],
    Input("instrument-status-store", "data"),
)
def render_status_badges(status):
    status = status or {}
    labels, colors, tooltips = [], [], []
    for key in STATUS_KEYS:
        label, color, tooltip = status_badge_content(key, status.get(key))
        labels.append(label)
        colors.append(color)
        tooltips.append(tooltip)
    return labels + colors + tooltips


if __name__ == "__main__":
    # debug=True enables Werkzeug's interactive in-browser debugger on
    # unhandled exceptions, which lets a visitor execute arbitrary code on
    # the server - never leave it on for anything but local development.
    # Defaults to off; opt in explicitly with SOLOLAB_DEBUG=1.
    debug = os.environ.get("SOLOLAB_DEBUG", "").lower() in ("1", "true", "yes")
    # use_reloader=False: the stat-based reloader spawns an extra subprocess
    # per restart on Windows without reliably killing the previous one,
    # leaving stale duplicate servers bound to the same port. Restart
    # manually after edits instead.
    app.run(debug=debug, use_reloader=False)
