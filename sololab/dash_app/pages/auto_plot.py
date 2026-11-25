"""Auto Plot - pick a start date/time + window length and the instruments,
and this page fetches whatever is needed to cover the WHOLE window
(reusing already-loaded data when it covers the window, otherwise
downloading every file/day that overlaps it and joining them), stores the
result as the loaded data for each instrument, and renders a stacked
quicklook via plotting.quicklook_plot_plotly.

Downloads land in shared server-side caches (public science data, same
reasoning as EPD_CACHE_DIR) so re-running a nearby window doesn't
re-download.
"""
import glob
import logging
import os
import re
from datetime import date, datetime, timedelta

import dash
import dash_bootstrap_components as dbc
import numpy as np
import pandas as pd
from dash import ALL, Input, Output, State, callback, dcc, html
from dash.exceptions import PreventUpdate

from sololab import rpw_read, stix_read
from sololab.dash_app import plotting
from sololab.dash_app.constants import EPD_CACHE_DIR, EPD_PARTICLE_OPTIONS, EPD_RESAMPLE_OPTIONS, rpw_key
from sololab.dash_app.session_store import session_store
from sololab.dash_app.utils import format_dt_input, order_rows

logger = logging.getLogger(__name__)

dash.register_page(__name__, path="/auto-plot", name="Auto Plot")

# ponytail: unbounded public-data caches, like EPD_CACHE_DIR; add an LRU sweep if disk becomes an issue
_CACHE_ROOT = os.environ.get("SOLOLAB_AUTOPLOT_CACHE_DIR", os.path.join(os.path.dirname(os.path.dirname(__file__)), ".autoplot_cache"))
STIX_CACHE_DIR = os.path.join(_CACHE_ROOT, "stix")
RPW_CACHE_DIR = os.path.join(_CACHE_ROOT, "rpw")

LABELS = {"stix": "STIX", "hfr": "RPW-HFR", "tnr": "RPW-TNR", "epd": "EPD"}
DEFAULT_ORDER = ["epd", "tnr", "hfr", "stix"]  # top to bottom
STIX_RANGES = {"6-10": [6, 10], "12-25": [12, 25], "32-50": [32, 50], "55-80": [55, 80]}
STIX_RANGES_DEFAULT = ["6-10", "12-25"]
TNR_MIN_KHZ = 50
EPD_RESAMPLE_DEFAULT = "5min"  # 1min is spiky for quiet-day EPT counts
MAX_WINDOW_MIN = 7 * 24 * 60
AUTO_BKG_WINDOW = "5min"  # STIX
RPW_BKG_WINDOW = "15min"  # wider: RPW bursts/drifts outlast 5 min, which biased the bkg high
EPD_BKG_MIN_POINTS = 5  # EPD window = max(5 min, 5 resampled points)


def layout(**kwargs):
    return dbc.Container(
        [
            html.H3("Auto Plot", className="mt-3"),
            dbc.Alert(id="ap-alert", is_open=False, dismissable=True, className="mt-2"),
            dbc.Row(
                [
                    dbc.Col(dbc.Card(dbc.CardBody(_controls())), xl=3, md=4),
                    dbc.Col(dcc.Loading(dcc.Graph(id="ap-graph"), type="circle"), xl=9, md=8),
                ]
            ),
            dcc.Store(id="ap-order", storage_type="session", data=DEFAULT_ORDER),
            dbc.Modal(
                [dbc.ModalHeader(dbc.ModalTitle(id="ap-bkg-title")), dbc.ModalBody(dcc.Graph(id="ap-bkg-graph"))],
                id="ap-bkg-modal", is_open=False, size="lg",
            ),
        ],
        className="pb-5",
    )


def _controls():
    keep = dict(persistence=True, persistence_type="session")
    return [
        html.Label("Start date (UTC)"),
        dcc.DatePickerSingle(
            id="ap-date", date=(date.today() - timedelta(days=7)).isoformat(),
            display_format="YYYY-MM-DD", className="d-block mb-2", **keep,
        ),
        html.Label("Start time [HH:MM:SS]"),
        dbc.Input(id="ap-time", type="time", step=1, value="00:00:00", className="mb-2", **keep),
        html.Label("Window [minutes]"),
        dbc.Input(id="ap-window", type="number", min=1, max=MAX_WINDOW_MIN, step=1, value=60, className="mb-2", **keep),
        html.Label("Instruments"),
        dbc.Checklist(
            id="ap-instruments",
            options=[{"label": LABELS[c], "value": c} for c in ["stix", "hfr", "tnr", "epd"]],
            value=["stix", "hfr", "tnr", "epd"],
            inline=True, className="mb-3", **keep,
        ),
        dbc.Button("AUTO PLOT", id="ap-plot-btn", className="w-100 btn-autoplot"),
        html.Div(id="ap-notes", className="small text-muted mt-2"),
        html.Hr(),
        dbc.Button("Advanced settings ▾", id="ap-adv-toggle", color="link", className="p-0"),
        dbc.Collapse(
            [
                dbc.Checkbox(id="ap-auto-bkg", label="Auto-bkg", value=False, className="mt-2", **keep),
                html.Div(
                    "Moving median over every channel; the window whose channel medians have the smallest "
                    "norm is the background for all channels "
                    "(window: STIX 5 min, RPW 15 min, EPD max(5 min, 5 resampled points)).",
                    className="small text-muted",
                ),
                html.Div(
                    [
                        dbc.Button("Auto-bkg plots ▾", id="ap-bkg-toggle", color="link", className="p-0 mt-2"),
                        dbc.Collapse(html.Div(id="ap-bkg-buttons", className="d-grid gap-1 mt-1"), id="ap-bkg-panel", is_open=False),
                    ],
                    id="ap-bkg-section",
                    style={"display": "none"},
                ),
                html.H6("Panel order (top to bottom)", className="mt-3"),
                html.Div(id="ap-order-rows"),
                html.H6("STIX", className="mt-3"),
                html.Div("Energy ranges [keV] (nearest bin edges are used)", className="small text-muted"),
                dbc.Checklist(id="ap-stix-ranges", options=list(STIX_RANGES), value=STIX_RANGES_DEFAULT, inline=True, **keep),
                html.Label("Smoothing points", className="mt-1"),
                dbc.Input(id="ap-stix-smooth", type="number", min=1, max=200, step=1, value=1, size="sm", **keep),
                html.H6("RPW", className="mt-3"),
                dbc.Checkbox(id="ap-tnr-extend", label=f"Extend TNR below {TNR_MIN_KHZ} kHz", value=False, **keep),
                dbc.Checkbox(id="ap-rpw-invert", label="Reverse spectrogram frequency axis", value=True, **keep),
                html.H6("EPD", className="mt-3"),
                dbc.Row(
                    [
                        dbc.Col([html.Label("Particle"), dbc.Select(id="ap-epd-particle", options=EPD_PARTICLE_OPTIONS, value="Electron", size="sm", **keep)]),
                        dbc.Col([html.Label("Resample"), dbc.Select(id="ap-epd-resample", options=EPD_RESAMPLE_OPTIONS, value=EPD_RESAMPLE_DEFAULT, size="sm", **keep)]),
                    ],
                    className="g-2",
                ),
                html.Div(
                    "EPT, sun viewing: 3 channels evenly spaced in the lower-energy half of the bins, 2 in the upper half.",
                    className="small text-muted mt-1",
                ),
            ],
            id="ap-adv",
            is_open=False,
        ),
    ]


@callback(Output("ap-adv", "is_open"), Input("ap-adv-toggle", "n_clicks"), State("ap-adv", "is_open"), prevent_initial_call=True)
def toggle_advanced(_n, is_open):
    return not is_open


@callback(Output("ap-order-rows", "children"), Input("ap-order", "data"), Input("ap-instruments", "value"))
def render_order(order, selected):
    return order_rows([c for c in order or DEFAULT_ORDER if c in (selected or [])], LABELS, "ap-order-move")


def move_in_order(order, selected, code, direction):
    """Swap `code` with its visible (selected) neighbour, keeping unselected
    instruments where they were in the full order."""
    visible = [c for c in order if c in selected]
    i = visible.index(code)
    j = i - 1 if direction == "up" else i + 1
    if not 0 <= j < len(visible):
        return order
    visible[i], visible[j] = visible[j], visible[i]
    it = iter(visible)
    return [next(it) if c in selected else c for c in order]


@callback(
    Output("ap-order", "data"),
    Input({"type": "ap-order-move", "code": ALL, "dir": ALL}, "n_clicks"),
    State("ap-order", "data"),
    State("ap-instruments", "value"),
    prevent_initial_call=True,
)
def move_panel(n_clicks_list, order, selected):
    trig = dash.callback_context.triggered_id
    if not any(n_clicks_list) or not trig:
        raise PreventUpdate
    return move_in_order(order or DEFAULT_ORDER, selected or [], trig["code"], trig["dir"])


# --- data fetching --------------------------------------------------------------------


def _ts(x):
    return pd.Timestamp(x).to_pydatetime()


def _covers(times, start, end, tol=timedelta(minutes=2)):
    times = pd.to_datetime(np.asarray(times))
    return len(times) > 0 and times.min() <= start + tol and times.max() >= end - tol


def latest_versions(results):
    """The Data Center lists every processing version (_V02_, _V04_...) and
    repeated requests of the same observation - keep one file per time
    range, the highest version."""
    best = {}
    for r in results:
        key = tuple(r.get("observation_time_range") or [r.get("file_id")])
        m = re.search(r"_V(\d+)", r.get("url", ""))
        ver = int(m.group(1)) if m else 0
        if key not in best or ver > best[key][0]:
            best[key] = (ver, r)
    return [r for _, r in best.values()]


def _stix_fetch(start, end):
    fmt = "%Y-%m-%dT%H:%M:%S"
    results = stix_read.stix_query_science_files(start.strftime(fmt), end.strftime(fmt), "xray-spec")
    if not results:  # no spectrogram files - fall back to L1 pixel data (flare periods only)
        results = stix_read.stix_query_science_files(start.strftime(fmt), end.strftime(fmt), "xray-l1")
    if not results:
        raise FileNotFoundError("no STIX files on the Data Center for this window")
    counts = [stix_read.stix_create_counts(stix_read.stix_download_file(r["file_id"], STIX_CACHE_DIR)) for r in latest_versions(results)]
    joined = counts[0] if len(counts) == 1 else stix_read.stix_combine_counts(counts, force=True)
    joined["time"] = np.asarray(joined["time"])
    return joined


def _rpw_day_path(day, data_type):
    folder = os.path.join(RPW_CACHE_DIR, data_type)
    hits = glob.glob(os.path.join(folder, f"*{data_type}*{day:%Y%m%d}*.cdf"))
    return hits[0] if hits else rpw_read.rpw_download_cdaweb_file(day, data_type, folder)


def join_psds(psds):
    """Concatenate daily PSDs along time, on the frequencies common to all."""
    if len(psds) == 1:
        return psds[0]
    common = sorted(set.intersection(*(set(np.asarray(p["frequency"]).tolist()) for p in psds)))
    blocks = []
    for p in psds:
        f = np.asarray(p["frequency"]).tolist()
        blocks.append(np.asarray(p["v"])[[f.index(x) for x in common], :])
    t = np.concatenate([np.asarray(p["time"], dtype="datetime64[us]") for p in psds])
    v = np.concatenate(blocks, axis=1)
    t, idx = np.unique(t, return_index=True)  # sorted + drops overlap duplicates
    return {**psds[0], "time": t, "frequency": np.array(common), "v": v[:, idx], "bkg": None, "bkg_std": None}


def _rpw_fetch(start, end, data_type):
    last = (end - timedelta(microseconds=1)).date()
    days = [start.date() + timedelta(days=k) for k in range((last - start.date()).days + 1)]
    psds, missing = [], []
    for day in days:
        try:
            data = rpw_read.rpw_get_data(_rpw_day_path(day, data_type))
            psds.append(rpw_read.rpw_create_PSD(data, which_freqs="non_zero"))
        except FileNotFoundError:
            missing.append(f"{day}")
    if not psds:
        raise FileNotFoundError(f"no RPW-{data_type.upper()} L3 files on CDAWeb for {', '.join(missing)}")
    return join_psds(psds)


def epd_channels(n):
    """3 channels evenly spaced in the lower half of n bins, 2 in the upper half."""
    h = n // 2
    return [int(h * (j + 0.5) / 3) for j in range(3)] + [h + int((n - h) * (j + 0.5) / 2) for j in range(2)]


def _epd_fetch(start, end):
    from solo_epd_loader import epd_load

    df_p, df_e, energies = epd_load(
        sensor="ept", level="l2", startdate=int(f"{start:%Y%m%d}"), enddate=int(f"{end:%Y%m%d}"),
        viewing="sun", path=EPD_CACHE_DIR, autodownload=True,
    )
    if isinstance(df_e, list) or isinstance(df_p, list):
        raise FileNotFoundError("no EPD data for this window (viewing='sun')")
    return df_p, df_e, energies


def _get_or_fetch(code, sid, status, start, end, particle="Electron"):
    """Returns (data, status_entry_or_None). Reuses loaded session data when it
    already covers [start, end]; otherwise downloads and stores the result as
    the instrument's loaded data (status entry returned so the caller can
    update instrument-status-store)."""
    if code == "stix":
        cur = session_store.get(sid, "stix_counts_final") if status.get("stix", {}).get("loaded") else None
        if cur is not None and _covers(cur["time"], start, end):
            return cur, None
        counts = _stix_fetch(start, end)
        session_store.set(sid, "stix_counts_final", counts)
        session_store.set(sid, "stix_meta", {"bkg_file_enabled": False, "bkg_time_enabled": False, "bkg_poll_function": "mean"})
        return counts, {
            "loaded": True, "min_time": format_dt_input(min(counts["time"])),
            "max_time": format_dt_input(max(counts["time"])), "bkg_enabled": False,
        }
    if code in ("hfr", "tnr"):
        key = rpw_key(code, "psd_final")
        cur = session_store.get(sid, key) if status.get(f"rpw_{code}", {}).get("loaded") else None
        if cur is not None and _covers(cur["time"], start, end):
            return cur, None
        psd = _rpw_fetch(start, end, code)
        session_store.set(sid, key, psd)
        session_store.set(sid, rpw_key(code, "meta"), {"bkg_option": 0, "bkg_poll_function": "median"})
        return psd, {
            "loaded": True, "min_time": format_dt_input(min(psd["time"])),
            "max_time": format_dt_input(max(psd["time"])), "bkg_enabled": False,
        }
    # epd
    final_key = "epd_electrons_final" if particle == "Electron" else "epd_protons_final"
    meta = session_store.get(sid, "epd_meta") or {}
    cur = session_store.get(sid, final_key) if status.get("epd", {}).get("loaded") and meta.get("particle") == particle else None
    if cur is not None and _covers(cur.index, start, end):
        return (cur, session_store.get(sid, "epd_energies")), None
    df_p, df_e, energies = _epd_fetch(start, end)
    df = df_e if particle == "Electron" else df_p
    for name, val in (("epd_protons_df", df_p), ("epd_electrons_df", df_e), ("epd_energies", energies), (final_key, df)):
        session_store.set(sid, name, val)
    day = f"{start:%Y-%m-%d}"
    session_store.set(sid, "epd_meta", {"date": day, "particle": particle, "resample": EPD_RESAMPLE_DEFAULT, "bkg_enabled": False, "bkg_poll_function": "mean"})
    return (df, energies), {"loaded": True, "date": day, "particle": particle, "resample": EPD_RESAMPLE_DEFAULT, "bkg_enabled": False}


def auto_bkg(times, values, window=AUTO_BKG_WINDOW):
    """values: (time, channel). Moving median of every channel over `window`;
    the ONE window whose vector of channel medians has the smallest L2 norm
    is the background for all channels (plus each channel's std in it).
    Only full windows count, so a lone early sample can't win.
    ponytail: plain norm, so the brightest channels dominate the choice;
    normalise channels first if that picks noisy windows."""
    df = pd.DataFrame(np.asarray(values, dtype=float), index=pd.DatetimeIndex(pd.to_datetime(np.asarray(times)))).sort_index()
    full = df.index >= df.index[0] + pd.Timedelta(window)
    if not full.any():  # window shorter than 5 min - fall back to the whole window
        return np.nan_to_num(df.median().to_numpy()), np.nan_to_num(df.std().to_numpy())
    roll = df.rolling(window)
    med = roll.median().to_numpy()[full]
    std = roll.std().to_numpy()[full]
    norm = np.sqrt(np.nansum(med**2, axis=1))
    norm[np.all(np.isnan(med), axis=1)] = np.inf
    i = int(np.argmin(norm))
    return np.nan_to_num(med[i]), np.nan_to_num(std[i])


def _in_window(times, start, end):
    t = pd.to_datetime(np.asarray(times))
    return np.asarray((t >= start) & (t <= end))


def _apply_auto_bkg(code, d, start, end, particle, resample=None):
    """Returns (background-subtracted copy for plotting, bkg payload for the
    bkg figure). The stored/loaded data is never modified."""
    if code == "stix":
        cps = np.asarray(d["counts_per_sec"], dtype=float)
        m = _in_window(d["time"], start, end)
        bkg, std = auto_bkg(np.asarray(d["time"])[m], cps[m])
        payload = {"mean_energy": d["mean_energy"], "background": bkg, "background_std": std}
        return {**d, "counts_per_sec": np.clip(cps - bkg, 0, None)}, payload
    if code in ("hfr", "tnr"):
        v = np.asarray(d["v"], dtype=float)
        m = _in_window(d["time"], start, end)
        bkg, std = auto_bkg(np.asarray(d["time"])[m], v[:, m].T, RPW_BKG_WINDOW)
        floor = np.nanmin(v[v > 0]) if np.any(v > 0) else 0  # keep the log colour scale valid
        payload = {
            "frequency": d["frequency"], "bkg": bkg[:, None], "bkg_std": std,
            "polling_function": f"auto {RPW_BKG_WINDOW} median", "type": d.get("type", code), "level": d.get("level", "L3"),
        }
        return {**d, "v": np.clip(v - bkg[:, None], floor, None)}, payload
    df, energies = d
    key = f"{particle}_Flux"
    flux = df[key].loc[_in_window(df.index, start, end)]
    freq = plotting._epd_resample_freq(resample)
    window = pd.Timedelta(AUTO_BKG_WINDOW)
    if freq:  # bkg from the same resampled series that gets plotted
        flux = flux.resample(freq).mean()
        window = max(window, EPD_BKG_MIN_POINTS * pd.Timedelta(pd.tseries.frequencies.to_offset(freq)))
    bkg, std = auto_bkg(flux.index, flux.to_numpy(), window)
    out = df.copy()
    # no clip here: resample(raw - bkg) == resample(raw) - bkg only without it;
    # epd_flux_figure floors the plotted values at 0.1 after resampling
    out[key] = df[key].to_numpy() - bkg
    return (out, energies), {"bkg": bkg, "bkg_std": std, "energies": energies, "particle": particle}


def _times(code, data):
    if code == "epd":
        return data[0].index
    return data["time"]


def _gap_note(code, data, start, end):
    t = pd.to_datetime(np.asarray(_times(code, data)))
    t = t[(t >= start) & (t <= end)]
    if len(t) == 0:
        return f"{LABELS[code]}: no data inside the window."
    if not _covers(t, start, end):
        return f"{LABELS[code]}: data only covers {t.min():%H:%M:%S} - {t.max():%H:%M:%S}."
    return None


@callback(
    Output("ap-graph", "figure"),
    Output("ap-notes", "children"),
    Output("ap-alert", "children"),
    Output("ap-alert", "color"),
    Output("ap-alert", "is_open"),
    Output("instrument-status-store", "data", allow_duplicate=True),
    Output("ap-bkg-section", "style"),
    Output("ap-bkg-buttons", "children"),
    Input("ap-plot-btn", "n_clicks"),
    State("ap-date", "date"),
    State("ap-time", "value"),
    State("ap-window", "value"),
    State("ap-instruments", "value"),
    State("ap-order", "data"),
    State("ap-stix-ranges", "value"),
    State("ap-stix-smooth", "value"),
    State("ap-tnr-extend", "value"),
    State("ap-rpw-invert", "value"),
    State("ap-epd-particle", "value"),
    State("ap-epd-resample", "value"),
    State("ap-auto-bkg", "value"),
    State("session-id", "data"),
    State("instrument-status-store", "data"),
    prevent_initial_call=True,
)
def auto_plot(n_clicks, date_str, time_str, window, selected, order, stix_ranges, stix_smooth, tnr_extend, rpw_invert, epd_particle, epd_resample, use_auto_bkg, sid, status):
    if not n_clicks:
        raise PreventUpdate
    nu = dash.no_update

    def fail(msg, color="warning"):
        return nu, nu, msg, color, True, nu, nu, nu

    if not selected:
        return fail("Select at least one instrument.")
    if "stix" in selected and not stix_ranges:
        return fail("Select at least one STIX energy range (Advanced settings).")
    try:
        t = (time_str or "00:00:00").strip()
        start = datetime.strptime(f"{date_str} {t if t.count(':') == 2 else t + ':00'}", "%Y-%m-%d %H:%M:%S")
    except (TypeError, ValueError):
        return fail("Enter a valid start date and time (HH:MM:SS).")
    if not window or int(window) < 1 or int(window) > MAX_WINDOW_MIN:
        return fail(f"Window must be between 1 and {MAX_WINDOW_MIN} minutes.")
    end = start + timedelta(minutes=int(window))

    status = dict(status or {})
    status_changed = False
    data, notes, errors = {}, [], []
    for code in selected:
        try:
            d, entry = _get_or_fetch(code, sid, status, start, end, epd_particle or "Electron")
        except Exception as exc:  # noqa: BLE001 - one instrument failing shouldn't block the others
            logger.exception("Auto plot fetch failed for %s", code)
            errors.append(f"{LABELS[code]}: {exc}")
            continue
        data[code] = d
        if entry:
            status["stix" if code == "stix" else "epd" if code == "epd" else f"rpw_{code}"] = entry
            status_changed = True
        note = _gap_note(code, d, start, end)
        if note:
            notes.append(note)

    display = [c for c in (order or DEFAULT_ORDER) if c in data]
    status_out = status if status_changed else nu
    if not display:
        return nu, nu, "Nothing to plot. " + " | ".join(errors), "danger", True, status_out, nu, nu

    tnr = data.get("tnr")
    if tnr is not None and not tnr_extend:
        f = np.asarray(tnr["frequency"])
        keep = f >= TNR_MIN_KHZ
        tnr = {**tnr, "frequency": f[keep], "v": np.asarray(tnr["v"])[keep]}
    if tnr is not None:
        data["tnr"] = tnr

    bkg_style, bkg_buttons = {"display": "none"}, []
    if use_auto_bkg:
        for code in display:
            data[code], payload = _apply_auto_bkg(code, data[code], start, end, epd_particle, epd_resample or EPD_RESAMPLE_DEFAULT)
            session_store.set(sid, f"ap_bkg_{code}", payload)
        groups = [("stix", "STIX"), ("rpw", "RPW (HFR + TNR)"), ("epd", "EPD")]
        avail = {"rpw" if c in ("hfr", "tnr") else c for c in display}
        bkg_buttons = [
            dbc.Button(f"Plot {label} background", id={"type": "ap-bkg-btn", "inst": g}, size="sm", color="secondary", outline=True)
            for g, label in groups if g in avail
        ]
        session_store.set(sid, "ap_bkg_rpw_codes", [c for c in display if c in ("hfr", "tnr")])
        bkg_style = {"display": "block"}

    epd_df, epd_energies = data.get("epd", (None, None))
    try:
        fig = plotting.quicklook_plot_plotly(
            stix_counts=data.get("stix"),
            hfr_psd=data.get("hfr"),
            tnr_psd=data.get("tnr"),
            epd_data=epd_df,
            epd_energies=epd_energies,
            display=display,
            date_range=(start, end),
            exact_date_range=True,
            stix_energy_bins=[STIX_RANGES[k] for k in STIX_RANGES if k in stix_ranges],
            stix_mode="curve",
            stix_smoothing_points=int(stix_smooth or 1),
            stix_curves_ylogscale=True,
            rpw_mode="spec",
            rpw_overlap="hfr",
            rpw_invert_yaxis=bool(rpw_invert),
            epd_channels=epd_channels(len(epd_energies[f"{epd_particle}_Bins_Text"])) if epd_df is not None else (),
            epd_particle=epd_particle,
            epd_resample=epd_resample or EPD_RESAMPLE_DEFAULT,
            epd_zero_as_gap=True,
            fontsize=11,
            linewidth=1.5,
            height_per_panel=200,
        )
    except Exception as exc:  # noqa: BLE001
        logger.exception("Auto plot render failed")
        return nu, nu, str(exc), "danger", True, status_out, nu, nu

    if use_auto_bkg:
        notes = notes + ["Auto-bkg subtracted (moving-median window with the lowest norm)."]
    if errors:
        return fig, notes, "Skipped - " + " | ".join(errors), "warning", True, status_out, bkg_style, bkg_buttons
    return fig, notes, "", "success", False, status_out, bkg_style, bkg_buttons


@callback(Output("ap-bkg-panel", "is_open"), Input("ap-bkg-toggle", "n_clicks"), State("ap-bkg-panel", "is_open"), prevent_initial_call=True)
def toggle_bkg_panel(_n, is_open):
    return not is_open


@callback(
    Output("ap-bkg-graph", "figure"),
    Output("ap-bkg-title", "children"),
    Output("ap-bkg-modal", "is_open"),
    Input({"type": "ap-bkg-btn", "inst": ALL}, "n_clicks"),
    State("session-id", "data"),
    prevent_initial_call=True,
)
def plot_auto_bkg(n_clicks_list, sid):
    trig = dash.callback_context.triggered_id
    if not any(n_clicks_list or []) or not trig:
        raise PreventUpdate
    inst = trig["inst"]
    if inst == "stix":
        return plotting.stix_bkg_figure(session_store.get(sid, "ap_bkg_stix")), "STIX auto-background", True
    if inst == "epd":
        b = session_store.get(sid, "ap_bkg_epd")
        return plotting.epd_bkg_figure(b["bkg"], b["bkg_std"], b["energies"], b["particle"]), "EPD auto-background", True
    codes = session_store.get(sid, "ap_bkg_rpw_codes") or []
    hfr = session_store.get(sid, "ap_bkg_hfr") if "hfr" in codes else None
    tnr = session_store.get(sid, "ap_bkg_tnr") if "tnr" in codes else None
    return plotting.rpw_bkg_combined_figure(hfr, tnr), "RPW auto-background", True


if __name__ == "__main__":
    assert epd_channels(34) == [2, 8, 14, 21, 29]
    assert move_in_order(["epd", "tnr", "hfr", "stix"], ["epd", "hfr", "stix"], "hfr", "up") == ["hfr", "tnr", "epd", "stix"]
    a = {"time": np.array(["2024-01-01T00:00", "2024-01-01T00:01"], dtype="datetime64[us]"), "frequency": np.array([2.0, 1.0]), "v": np.array([[1, 2], [3, 4]])}
    b = {"time": np.array(["2024-01-01T00:01", "2024-01-01T00:02"], dtype="datetime64[us]"), "frequency": np.array([1.0, 2.0, 3.0]), "v": np.array([[5, 6], [7, 8], [0, 0]])}
    tt = pd.date_range("2024-01-01", periods=20, freq="1min")
    vals = np.column_stack([np.r_[np.full(10, 5.0), np.full(10, 1.0)], np.r_[np.full(10, 2.0), np.full(10, 9.0)]])
    bk, _ = auto_bkg(tt, vals)
    assert bk.tolist() == [5.0, 2.0], bk  # one shared window: (5, 2) has a smaller norm than (1, 9)
    rs = [{"observation_time_range": ["a", "b"], "url": "x_V02_1.fits"}, {"observation_time_range": ["a", "b"], "url": "x_V04_2.fits"}, {"observation_time_range": ["c", "d"], "url": "y_V01_3.fits"}]
    assert [r["url"] for r in latest_versions(rs)] == ["x_V04_2.fits", "y_V01_3.fits"]
    j = join_psds([a, b])
    assert j["frequency"].tolist() == [1.0, 2.0] and j["v"].tolist() == [[3, 4, 6], [1, 2, 8]], j
    print("ok")
