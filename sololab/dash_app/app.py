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
from dash import Input, Output, State, callback, dcc, html

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

app = dash.Dash(
    __name__,
    use_pages=True,
    external_stylesheets=[dbc.themes.FLATLY],
    suppress_callback_exceptions=True,
    title="SoloLab",
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


def _nav_links():
    return [
        dbc.NavLink(page["name"], href=page["path"], active="exact")
        for page in dash.page_registry.values()
    ]


def _status_badges():
    return html.Div(
        [
            dbc.Badge(id=f"status-badge-{key.replace('_', '-')}", color="danger", className="me-2")
            for key in STATUS_KEYS
        ],
        className="d-flex flex-wrap gap-1",
    )


navbar = dbc.Navbar(
    dbc.Container(
        [
            dbc.NavbarBrand("SoloLab", href="/"),
            dbc.Nav(_nav_links(), navbar=True, className="me-auto"),
            _status_badges(),
        ],
        fluid=True,
    ),
    color="dark",
    dark=True,
    className="mb-3",
)

app.layout = html.Div(
    [
        dcc.Location(id="url"),
        dcc.Store(id="session-id", storage_type="session"),
        dcc.Store(id="plot-prefs-store", storage_type="session", data=DEFAULT_PLOT_PREFS),
        dcc.Store(id="instrument-status-store", storage_type="session", data=DEFAULT_INSTRUMENT_STATUS),
        navbar,
        dbc.Container(dash.page_container, fluid=True),
    ]
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
    + [Output(f"status-badge-{key.replace('_', '-')}", "color") for key in STATUS_KEYS],
    Input("instrument-status-store", "data"),
)
def render_status_badges(status):
    status = status or {}
    texts = []
    colors = []
    for key in STATUS_KEYS:
        text, color = status_badge_content(key, status.get(key))
        texts.append(text)
        colors.append(color)
    return texts + colors


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
