"""Plotly equivalents of sololab/quicklooks.py's matplotlib plotting
functions. Nothing here imports sololab.quicklooks or matplotlib at
runtime - only numpy/plotly/sololab.values, so the Dash server never needs
a display backend.

Each function is documented with the matplotlib function it replaces, for
side-by-side comparison against quicklooks.py while porting behaviour.
Deliberate visual simplifications vs. the original are noted inline (see
also the "Simplificaciones visuales" section of the migration plan).
"""
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from sololab import values

# ---------------------------------------------------------------------------
# shared helpers
# ---------------------------------------------------------------------------


def smooth(y, pts):
    """Port of quicklooks.smooth: simple moving-average via convolution."""
    if not pts or pts <= 1:
        return np.asarray(y)
    ones = np.ones(pts) / pts
    return np.convolve(y, ones, mode="same")


def _as_datetime_array(time_data):
    return np.asarray(time_data)


def _invert_yaxis(fig, secondary_y=False, row=None, col=None):
    """Invert a y-axis whether or not it already has an explicit range set
    (from a frequency_range/energy_range constraint) - reversing the range
    list directly when there is one keeps the constraint, autorange
    "reversed" handles the auto-scaled case."""
    kwargs = dict(secondary_y=secondary_y)
    if row is not None:
        kwargs.update(row=row, col=col)
    target = fig.select_yaxes(secondary_y=secondary_y, row=row, col=col)
    current_range = next((ax.range for ax in target), None)
    if current_range:
        fig.update_yaxes(range=list(reversed(current_range)), **kwargs)
    else:
        fig.update_yaxes(autorange="reversed", **kwargs)


def _filter_time_range(time_arr, date_range):
    """date_range: (start, end), or None. Returns a boolean mask over
    time_arr. time_arr may hold numpy.datetime64 (RPW PSDs) or plain
    datetime.datetime (STIX counts) - everything is normalized through
    numpy.datetime64[us] so the comparison never mixes the two types
    (a naive Python `datetime <= np.datetime64` comparison can be brittle
    across numpy versions)."""
    if date_range is None:
        return np.ones(len(time_arr), dtype=bool)
    start = np.datetime64(date_range[0])
    end = np.datetime64(date_range[1])
    t = np.asarray(time_arr, dtype="datetime64[us]")
    return (t >= start) & (t <= end)


# ---------------------------------------------------------------------------
# STIX  (port of stix_plot_spectrogram / stix_plot_counts / stix_plot_bkg /
# stix_plot_overlay in quicklooks.py)
# ---------------------------------------------------------------------------


def stix_spectrogram_figure(
    counts,
    energy_range=None,
    date_range=None,
    logscale=True,
    ylogscale=False,
    colorscale="Jet",
    height=420,
):
    """Port of stix_plot_spectrogram. counts_per_sec (T, E) -> heatmap
    (E rows, T cols), log10 in Z computed manually (Plotly has no log
    color-axis)."""
    time = counts["time"]
    cts_per_sec = np.asarray(counts["counts_per_sec"])
    min_channels = cts_per_sec.shape[-1]
    mean_e = np.asarray(counts["mean_energy"][:min_channels])

    if logscale:
        z = np.zeros_like(cts_per_sec, dtype=float)
        mask = cts_per_sec > 0
        z[mask] = np.log10(cts_per_sec[mask])
    else:
        z = cts_per_sec
    z = np.nan_to_num(z, nan=0.0)

    fig = go.Figure(
        go.Heatmap(
            x=time,
            y=mean_e,
            z=z.T,
            colorscale=colorscale,
            zmin=0,
            colorbar=dict(
                title="Log10 Counts/s" if logscale else "Counts/s",
            ),
        )
    )
    fig.update_yaxes(title="STIX Energy bins [keV]", type="log" if ylogscale else "linear")
    if energy_range:
        fig.update_yaxes(range=energy_range if not ylogscale else np.log10(energy_range))
    if date_range:
        fig.update_xaxes(range=list(date_range))
    fig.update_layout(height=height, margin=dict(l=60, r=20, t=30, b=40))
    return fig


def stix_counts_traces(
    counts,
    integrate_bins=None,
    e_range=None,
    date_range=None,
    smoothing_pts=1,
    lw=1.5,
):
    """Port of stix_plot_counts (trace-building part only). Returns a list
    of go.Scatter, one per integration bin (or one per energy channel if
    integrate_bins is None)."""
    color_list = ["red", "dodgerblue", "limegreen", "cyan", "magenta"]

    time = np.asarray(counts["time"])
    cts_per_sec = np.asarray(counts["counts_per_sec"])
    min_channels = cts_per_sec.shape[-1]
    energies = counts["energy_bins"][:min_channels]
    mean_e = np.asarray(counts["mean_energy"][:min_channels])

    cts_data = np.nan_to_num(cts_per_sec, nan=0.0)

    if e_range is not None:
        e_idx = np.logical_and(mean_e >= e_range[0], mean_e <= e_range[1])
        cts_data = cts_data[:, e_idx]
        energies = energies[e_idx]
        mean_e = mean_e[e_idx]

    if date_range is not None:
        mask = _filter_time_range(time, date_range)
        cts_data = cts_data[mask, :]
        time = time[mask]

    plot_groups = []
    if integrate_bins:
        for e_low, e_high in integrate_bins:
            e_idx = np.logical_and(mean_e >= e_low, mean_e <= e_high)
            if not np.any(e_idx):
                continue
            energies_g = energies[e_idx]
            cts_sec_g = np.sum(cts_data[:, e_idx], axis=1)
            energy_g = [energies_g[0]["e_low"], energies_g[-1]["e_high"]]
            plot_groups.append([cts_sec_g, energy_g])
    else:
        for e in range(len(energies)):
            plot_groups.append(
                [cts_data[:, e], [energies[e]["e_low"], energies[e]["e_high"]]]
            )

    traces = []
    for g, (counts_plot, e_bounds) in enumerate(plot_groups):
        label = f"{int(e_bounds[0])}-{int(e_bounds[1])} keV"
        traces.append(
            go.Scatter(
                x=time,
                y=smooth(counts_plot, smoothing_pts),
                mode="lines",
                name=label,
                line=dict(width=lw, color=color_list[g % len(color_list)]),
            )
        )
    return traces


def stix_counts_figure(
    counts,
    integrate_bins=None,
    e_range=None,
    date_range=None,
    smoothing_pts=1,
    ylogscale=True,
    lw=1.5,
    height=420,
):
    """Port of stix_plot_counts wrapped as a standalone figure (used by
    Plot Preferences' "time profiles" preview)."""
    traces = stix_counts_traces(
        counts,
        integrate_bins=integrate_bins,
        e_range=e_range,
        date_range=date_range,
        smoothing_pts=smoothing_pts,
        lw=lw,
    )
    fig = go.Figure(traces)
    fig.update_yaxes(
        title="STIX Count Rate [cts/sec]",
        type="log" if ylogscale else "linear",
        rangemode="tozero" if not ylogscale else None,
    )
    fig.update_layout(height=height, margin=dict(l=60, r=20, t=30, b=40))
    return fig


def stix_overlay_figure(
    counts,
    energy_range=None,
    date_range=None,
    stix_energy_bins=None,
    stix_smoothing_points=5,
    stix_spec_zlogscale=True,
    stix_spec_ylogscale=False,
    stix_curves_ylogscale=True,
    linewidth=2,
    colorscale="Jet",
    height=460,
):
    """Port of stix_plot_overlay: spectrogram heatmap + integrated count-rate
    curves on a secondary y-axis."""
    spec_fig = stix_spectrogram_figure(
        counts,
        energy_range=energy_range,
        date_range=date_range,
        logscale=stix_spec_zlogscale,
        ylogscale=stix_spec_ylogscale,
        colorscale=colorscale,
    )
    traces = stix_counts_traces(
        counts,
        integrate_bins=stix_energy_bins,
        date_range=date_range,
        smoothing_pts=stix_smoothing_points,
        lw=linewidth,
    )

    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(spec_fig.data[0], secondary_y=False)
    for tr in traces:
        fig.add_trace(tr, secondary_y=True)
    fig.update_yaxes(
        title="STIX Energy bins [keV]",
        type="log" if stix_spec_ylogscale else "linear",
        secondary_y=False,
    )
    fig.update_yaxes(
        title="STIX Count Rate [cts/sec]",
        type="log" if stix_curves_ylogscale else "linear",
        secondary_y=True,
    )
    if date_range:
        fig.update_xaxes(range=list(date_range))
    fig.update_layout(height=height, margin=dict(l=60, r=60, t=30, b=40))
    return fig


def stix_bkg_figure(counts, height=380):
    """Port of stix_plot_bkg: bkg counts/energy vs mean energy, log-log,
    with calibration lines at 31/81 keV."""
    if "background" not in counts:
        fig = go.Figure()
        fig.add_annotation(text="This dataset has no subtracted background.", showarrow=False)
        return fig

    energies = np.asarray(counts["mean_energy"])
    bkg_counts = np.asarray(counts["background"])
    min_channels = min(len(energies), len(bkg_counts))
    energies = energies[:min_channels]
    bkg_counts = bkg_counts[:min_channels]

    fig = go.Figure(
        go.Scatter(
            x=energies,
            y=bkg_counts / energies,
            mode="lines+markers",
            name="Background",
        )
    )
    fig.update_xaxes(title="Energy [keV]", type="log")
    fig.update_yaxes(title="Counts / sec / keV", type="log")
    if energies[-1] >= 31:
        fig.add_vline(x=31, line_dash="dash", line_color="red")
    if energies[-1] >= 81:
        fig.add_vline(x=81, line_dash="dash", line_color="red")
    fig.update_layout(height=height, margin=dict(l=60, r=20, t=30, b=40))
    return fig


# ---------------------------------------------------------------------------
# RPW (HFR/TNR)  (port of rpw_plot_psd / rpw_plot_curves / rpw_plot_overlay /
# rpw_plot_bkg in quicklooks.py)
# ---------------------------------------------------------------------------


def _rpw_y_ticks(psd_type, f):
    """Port of rpw_plot_psd's per-instrument y-tick selection + dynamic-
    precision kHz->MHz formatter."""
    display = values.display_freqs_tnr if psd_type == "tnr" else values.display_freqs
    ticks = [x for x in display if f[-1] >= x >= f[0]]
    if not ticks:
        return [], []
    text = []
    for y in ticks:
        prec = int(max(-np.log10(y / 1000.0), 2 if psd_type == "tnr" else 1))
        text.append(f"{y / 1000.0:.{prec}f}")
    return ticks, text


def rpw_psd_figure(
    psd,
    frequency_range=None,
    date_range=None,
    vmin=None,
    vmax=None,
    rpw_cbar_units="wmhz",
    colorscale="Jet",
    show_colorbar=True,
    height=420,
):
    """Port of rpw_plot_psd. RPW spectrograms are ALWAYS log-scaled on the Y
    axis in the original (quicklooks.py calls ax.set_yscale('log')
    unconditionally, independent of any "log Y" preference) - replicated
    here as an unconditional log axis, not a toggle."""
    t, f, z = psd["time"], np.asarray(psd["frequency"]), np.asarray(psd["v"])
    multi = 1e-22 if rpw_cbar_units == "wmhz" else 1
    z = z * multi
    z = np.log10(z, out=np.full_like(z, np.nan), where=z > 0)
    zmin = np.log10(vmin) if vmin else None
    zmax = np.log10(vmax) if vmax else None

    if psd["level"] == "L2":
        cbar_title = "Log10 PSD [V^2/Hz]"
    else:
        cbar_title = "Log10 Flux [SFU]" if rpw_cbar_units == "SFU" else "Log10 Flux [W/m^2/Hz]"

    fig = go.Figure(
        go.Heatmap(
            x=t,
            y=f,
            z=z,
            colorscale=colorscale,
            zmin=zmin,
            zmax=zmax,
            showscale=show_colorbar,
            colorbar=dict(title=cbar_title) if show_colorbar else None,
        )
    )
    ticks, ticktext = _rpw_y_ticks(psd["type"], f)
    label = "RPW - TNR Frequency [MHz]" if psd["type"] == "tnr" else "RPW - HFR Frequency [MHz]"
    fig.update_yaxes(type="log", title=label, tickvals=ticks, ticktext=ticktext)
    if frequency_range:
        fig.update_yaxes(range=[np.log10(max(frequency_range[0], f.min())), np.log10(min(frequency_range[1], f.max()))])
    if date_range:
        fig.update_xaxes(range=list(date_range))
    fig.update_layout(height=height, margin=dict(l=70, r=20, t=30, b=40))
    return fig


def rpw_curve_traces(
    psd,
    freqs,
    date_range=None,
    smoothing_pts=None,
    bias_multiplier=None,
    lcolor=None,
    lw=1.5,
):
    """Port of rpw_plot_curves (trace-building part). Returns a list of
    go.Scatter, nearest-frequency lookup per requested freq, optional
    moving-average smoothing and "waterfall" bias-stacking (each curve
    multiplied by bias_multiplier**(n-g) and offset visually so overlapping
    curves stay legible)."""
    color_list = ["red", "dodgerblue", "limegreen", "orange", "cyan", "magenta", "black"]
    t, f, z = np.asarray(psd["time"]), np.asarray(psd["frequency"]), np.asarray(psd["v"])

    freqs = np.asarray(freqs, dtype=float)
    freqs = freqs[(freqs >= f.min()) & (freqs <= f.max())]
    if len(freqs) == 0:
        return []

    mask = _filter_time_range(t, date_range)
    plot_time = t[mask]
    plot_z = z[:, mask]

    if not lcolor:
        lcolor = color_list[: len(freqs)]

    traces = []
    n = len(freqs)
    for g, sel_freq in enumerate(freqs):
        idx_close = int(np.argmin(np.abs(f - sel_freq)))
        close_freq = f[idx_close]
        y = plot_z[idx_close, :]
        if smoothing_pts:
            y = smooth(y, smoothing_pts)
        label = f"{int(close_freq)} kHz"
        color = lcolor[g % len(lcolor)]
        if bias_multiplier:
            y = y * (bias_multiplier ** (n - g))
        traces.append(go.Scatter(x=plot_time, y=y, mode="lines", name=label, line=dict(width=lw, color=color)))
    return traces


def rpw_curve_figure(
    psd,
    freqs,
    date_range=None,
    smoothing_pts=None,
    ylogscale=True,
    lw=1.5,
    height=420,
):
    """Port of rpw_plot_curves wrapped as a standalone figure (Plot
    Preferences' "time profiles" preview, and the import dialogs' fixed-
    frequency preview)."""
    traces = rpw_curve_traces(psd, freqs, date_range=date_range, smoothing_pts=smoothing_pts, lw=lw)
    fig = go.Figure(traces)
    fig.update_yaxes(type="log" if ylogscale else "linear", title="Intensity")
    fig.update_layout(height=height, margin=dict(l=60, r=20, t=30, b=40))
    return fig


def rpw_overlay_figure(
    psd,
    freqs,
    frequency_range=None,
    date_range=None,
    rpw_units="wmhz",
    invert_y=True,
    smoothing_pts=5,
    linewidth=2,
    colorscale="Jet",
    height=460,
):
    """Port of rpw_plot_overlay: spectrogram heatmap + frequency curves on a
    secondary y-axis."""
    spec_fig = rpw_psd_figure(
        psd,
        frequency_range=frequency_range,
        date_range=date_range,
        rpw_cbar_units=rpw_units,
        colorscale=colorscale,
    )
    traces = rpw_curve_traces(psd, freqs, date_range=date_range, smoothing_pts=smoothing_pts, lw=linewidth)

    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(spec_fig.data[0], secondary_y=False)
    for tr in traces:
        fig.add_trace(tr, secondary_y=True)
    fig.layout.yaxis = spec_fig.layout.yaxis
    if invert_y:
        _invert_yaxis(fig, secondary_y=False)
    fig.update_yaxes(title="Intensity", secondary_y=True)
    if date_range:
        fig.update_xaxes(range=list(date_range))
    fig.update_layout(height=height, margin=dict(l=70, r=60, t=30, b=40))
    return fig


def rpw_bkg_figure(psd, height=380):
    """Port of rpw_plot_bkg: used-background profile + per-frequency maxima
    of the data, log-log."""
    frequency = np.asarray(psd["frequency"])
    bkg = np.asarray(psd["bkg"])[:, 0]
    v = np.asarray(psd["v"])
    maxs = np.array([np.max(v[i, :]) for i in range(len(frequency))])

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=frequency,
            y=bkg,
            mode="lines",
            name=f"Used Background ({psd.get('polling_function')})",
            line=dict(color="red"),
        )
    )
    fig.add_trace(
        go.Scatter(x=frequency, y=maxs, mode="lines", name="Max. values in data", line=dict(color="black", dash="dot"))
    )
    fig.update_xaxes(title="Frequency [kHz]", type="log")
    fig.update_yaxes(title="SFU" if psd["level"] == "L3" else "PSD(V)", type="log")
    fig.update_layout(
        title=f"Background {psd['type'].upper()} {psd['level'].upper()} (used {psd.get('polling_function')})",
        height=height,
        margin=dict(l=60, r=20, t=40, b=40),
    )
    return fig


# ---------------------------------------------------------------------------
# EPD (EPT)  (port of plot_ept_data in quicklooks.py)
# ---------------------------------------------------------------------------


def epd_bin_text(energies_ept, particle, channel):
    """energies_ept[f"{particle}_Bins_Text"][channel] -> the "<min> - <max>
    MeV" label string for one channel. Defensive against a
    solo_epd_loader shape difference observed between versions: some
    versions wrap the string in a 1-element array (need a trailing [0]),
    the version this app was tested against (0.4.4) returns the string
    directly."""
    value = energies_ept[f"{particle}_Bins_Text"][channel]
    if isinstance(value, str):
        return value
    return value[0]


def _epd_resample_freq(resample):
    """UI-facing resample label (e.g. "30sec") -> a pandas offset alias
    ("30s"). Values already in pandas format (e.g. "1min") pass through
    unchanged. None/empty disables resampling."""
    if not resample:
        return None
    if resample.endswith("sec"):
        return resample[:-3] + "s"
    return resample


def epd_flux_figure(
    epd_df,
    energies_ept,
    particle="Electron",
    channels=(2, 6, 14, 18, 26),
    date_range=None,
    round_epd_label=True,
    resample=None,
    height=420,
):
    """Port of plot_ept_data, now with real resampling: `resample` (e.g.
    "1min", "30sec") bins the flux series via pandas `.resample(freq).mean()`
    before plotting, instead of the original's fixed 600-point moving
    average (which ignored the "Resample" UI control entirely - see
    Backend Fixes item 14). The jet_r 7-color cycle used by matplotlib's
    set_prop_cycle isn't replicated; Plotly's default qualitative palette is
    used instead (a deliberate visual simplification)."""
    df = epd_df
    if date_range is not None:
        start, end = date_range
        mask = (df.index > start) & (df.index <= end)
        df = df.loc[mask]

    freq = _epd_resample_freq(resample)

    traces = []
    for channel in channels:
        leg_elems = epd_bin_text(energies_ept, particle, channel).split()
        to_round = 0 if round_epd_label else 2
        ktype = int if round_epd_label else float
        low = ktype(round(float(leg_elems[0]) * 1000, to_round))
        high = ktype(round(float(leg_elems[2]) * 1000, to_round))
        label = f"{low} - {high} keV"

        y = df[f"{particle}_Flux"][f"{particle}_Flux_{channel}"]
        if freq:
            y = y.resample(freq).mean()
        traces.append(go.Scatter(x=y.index, y=y, mode="lines", name=label))

    fig = go.Figure(traces)
    fig.update_yaxes(
        title=f"EPD - EPT {particle} flux [(cm^2 sr s MeV)^-1]",
        type="log",
    )
    fig.update_layout(
        height=height,
        margin=dict(l=70, r=20, t=30, b=40),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5),
    )
    return fig


# ---------------------------------------------------------------------------
# Combined plot  (port of quicklook_plot in quicklooks.py)
# ---------------------------------------------------------------------------


def resolve_frequency_ranges(display, hfr_psd, tnr_psd, rpw_frequency_range, rpw_overlap):
    """Port of the hfr_freq_range/tnr_freq_range cross-clipping block in
    quicklook_plot (quicklooks.py ~806-856). Order matters: HFR's range is
    resolved first (using TNR's UNCLIPPED native range for the "tnr"-overlap
    case), then TNR's range is resolved using HFR's now-adjusted range for
    the "hfr"-overlap case - replicated exactly, including that asymmetry."""
    hfr_range = None
    tnr_range = None
    both = "tnr" in display and "hfr" in display
    if both:
        hfr_range = [hfr_psd["frequency"][0], hfr_psd["frequency"][-1]]
        tnr_range = [tnr_psd["frequency"][0], tnr_psd["frequency"][-1]]
    if "hfr" in display:
        hfr_range = [hfr_psd["frequency"][0], hfr_psd["frequency"][-1]]
        if rpw_frequency_range:
            hfr_range = [max(hfr_range[0], rpw_frequency_range[0]), min(hfr_range[-1], rpw_frequency_range[-1])]
        if rpw_overlap == "tnr" and "tnr" in display:
            hfr_range[0] = max(hfr_range[0], tnr_range[1])
    if "tnr" in display:
        tnr_range = [tnr_psd["frequency"][0], tnr_psd["frequency"][-1]]
        if rpw_frequency_range:
            tnr_range = [max(tnr_range[0], rpw_frequency_range[0]), min(tnr_range[-1], rpw_frequency_range[-1])]
        if rpw_overlap == "hfr" and "hfr" in display:
            tnr_range[1] = min(hfr_range[0], tnr_range[1])
    return {"hfr": hfr_range, "tnr": tnr_range}


def resolve_common_time_range(time_arrays, date_range=None):
    """Port of the d_range/label_obstime block in quicklook_plot. time_arrays
    is a dict of instrument name -> time array (datetime64 or datetime).
    Returns (start, end, label) as (datetime, datetime, str)."""
    mins, maxs = [], []
    for arr in time_arrays.values():
        arr = np.asarray(arr, dtype="datetime64[us]")
        mins.append(arr.min())
        maxs.append(arr.max())
    if date_range:
        mins.append(np.datetime64(date_range[0]))
        maxs.append(np.datetime64(date_range[1]))
    start = pd.Timestamp(max(mins)).to_pydatetime()
    end = pd.Timestamp(min(maxs)).to_pydatetime()

    elapsed = (end - start).total_seconds()
    if elapsed > 36000:
        label_elapsed = f"{round(elapsed / 3600)} hrs"
    elif elapsed > 180:
        label_elapsed = f"{round(elapsed / 60)} min"
    else:
        label_elapsed = f"{round(elapsed)} sec"
    label = f"Obs. time [@ SolO]: {start.strftime(values.std_date_fmt)} ({label_elapsed})"
    return start, end, label


def resolve_combined_panels(display, rpw_mode, stix_mode, hfr_frequencies, tnr_frequencies):
    """Port of the plots_todo loop in quicklook_plot (quicklooks.py
    ~889-906) - one panel id per row of the combined figure, in the order
    instruments were selected.

    Deviation from the original (bug fix, not a faithful port): the
    original's curve-mode branch appends BOTH tnr_frequencies AND
    hfr_frequencies panels every time `disp` is 'tnr' OR 'hfr', so
    selecting both HFR and TNR together with mode="curve" silently
    duplicates every curve panel. Here each instrument's curve panels are
    only emitted once, when disp matches that instrument."""
    plots_todo = []
    for disp in display:
        if disp in ("tnr", "hfr"):
            if rpw_mode in ("spec", "overlay"):
                plots_todo.append(f"{disp}_{rpw_mode}_0")
            elif rpw_mode == "curve":
                freqs = tnr_frequencies if disp == "tnr" else hfr_frequencies
                for freq in freqs:
                    plots_todo.append(f"{disp}_curve_{freq}")
        if disp == "stix":
            if stix_mode in ("spec", "overlay"):
                plots_todo.append(f"stix_{stix_mode}_0")
            elif stix_mode == "curve":
                plots_todo.append("stix_curve_0")
        if disp == "epd":
            plots_todo.append("epd_curve_0")
    return plots_todo


def _copy_yaxis(fig, src_yaxis, row, secondary_y=False):
    kwargs = {}
    if src_yaxis.type:
        kwargs["type"] = src_yaxis.type
    if src_yaxis.title and src_yaxis.title.text:
        kwargs["title_text"] = src_yaxis.title.text
    if src_yaxis.tickvals:
        kwargs["tickvals"] = src_yaxis.tickvals
    if src_yaxis.ticktext:
        kwargs["ticktext"] = src_yaxis.ticktext
    if src_yaxis.range:
        kwargs["range"] = src_yaxis.range
    if kwargs:
        fig.update_yaxes(row=row, col=1, secondary_y=secondary_y, **kwargs)


def quicklook_plot_plotly(
    stix_counts=None,
    hfr_psd=None,
    tnr_psd=None,
    epd_data=None,
    epd_energies=None,
    display=("hfr", "stix"),
    date_range=None,
    stix_energy_range=(4, 28),
    stix_energy_bins=([4, 12], [16, 28]),
    stix_mode="curve",
    stix_smoothing_points=5,
    stix_curves_ylogscale=True,
    stix_spec_ylogscale=False,
    stix_spec_zlogscale=True,
    rpw_frequency_range=None,
    hfr_frequencies=(),
    tnr_frequencies=(),
    rpw_mode="spec",
    rpw_overlap="hfr",
    rpw_units="wmhz",
    rpw_invert_yaxis=True,
    rpw_smoothing_points=5,
    epd_channels=(0, 2, 6),
    epd_particle="Electron",
    epd_round_label=True,
    epd_resample=None,
    fontsize=13,
    linewidth=2,
    height_per_panel=140,
):
    """Port of quicklook_plot. Builds one go.Figure with N stacked,
    x-shared subplots (one row per panel in resolve_combined_panels),
    mixing go.Heatmap (spectrograms) and go.Scatter (curves) traces,
    matching the original's per-panel dispatch (hfr/tnr spec|overlay|curve,
    stix spec|overlay|curve, epd curve)."""
    missing = []
    if "hfr" in display and hfr_psd is None:
        missing.append("RPW-HFR")
    if "tnr" in display and tnr_psd is None:
        missing.append("RPW-TNR")
    if "stix" in display and stix_counts is None:
        missing.append("STIX")
    if "epd" in display and (epd_data is None or epd_energies is None):
        missing.append("EPD")
    if missing:
        raise ValueError("Missing data for: " + ", ".join(missing))

    freq_ranges = resolve_frequency_ranges(display, hfr_psd, tnr_psd, rpw_frequency_range, rpw_overlap)

    time_arrays = {}
    if "hfr" in display:
        time_arrays["hfr"] = hfr_psd["time"]
    if "tnr" in display:
        time_arrays["tnr"] = tnr_psd["time"]
    if "stix" in display:
        time_arrays["stix"] = stix_counts["time"]
    if "epd" in display:
        time_arrays["epd"] = epd_data.index.to_numpy()
    start, end, label_obstime = resolve_common_time_range(time_arrays, date_range)
    d_range = (start, end)

    plots_todo = resolve_combined_panels(display, rpw_mode, stix_mode, hfr_frequencies, tnr_frequencies)
    if not plots_todo:
        raise ValueError("Nothing to plot for the current display/mode selection.")
    n_plots = len(plots_todo)

    common_vmin = common_vmax = None
    if "tnr" in display and "hfr" in display:
        multi = 1 if rpw_units == "SFU" else 1e-22
        common_vmin = min(np.min(tnr_psd["v"]), np.min(hfr_psd["v"])) * multi
        common_vmax = max(np.max(tnr_psd["v"]), np.max(hfr_psd["v"])) * multi

    fig = make_subplots(
        rows=n_plots,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=min(0.25 / n_plots, 0.04),
        specs=[[{"secondary_y": True}] for _ in range(n_plots)],
    )

    rpw_colorbar_shown = False
    for i, panel in enumerate(plots_todo):
        row = i + 1
        origin, ptype, detail = panel.split("_", 2)

        if origin in ("hfr", "tnr"):
            psd = hfr_psd if origin == "hfr" else tnr_psd
            frange = freq_ranges[origin]
            freqs = hfr_frequencies if origin == "hfr" else tnr_frequencies
            if ptype in ("spec", "overlay"):
                show_cbar = not rpw_colorbar_shown
                rpw_colorbar_shown = True
                spec_fig = rpw_psd_figure(
                    psd,
                    frequency_range=frange,
                    date_range=d_range,
                    vmin=common_vmin,
                    vmax=common_vmax,
                    rpw_cbar_units=rpw_units,
                    show_colorbar=show_cbar,
                )
                fig.add_trace(spec_fig.data[0], row=row, col=1, secondary_y=False)
                _copy_yaxis(fig, spec_fig.layout.yaxis, row)
                if rpw_invert_yaxis:
                    _invert_yaxis(fig, secondary_y=False, row=row, col=1)
                if ptype == "overlay":
                    for tr in rpw_curve_traces(psd, freqs, smoothing_pts=rpw_smoothing_points, lw=linewidth):
                        fig.add_trace(tr, row=row, col=1, secondary_y=True)
            elif ptype == "curve":
                freq_val = float(detail)
                for tr in rpw_curve_traces(psd, [freq_val], smoothing_pts=rpw_smoothing_points, lcolor=["black"], lw=linewidth):
                    fig.add_trace(tr, row=row, col=1, secondary_y=False)

        elif origin == "stix":
            if ptype in ("spec", "overlay"):
                spec_fig = stix_spectrogram_figure(
                    stix_counts,
                    energy_range=stix_energy_range,
                    date_range=d_range,
                    logscale=stix_spec_zlogscale,
                    ylogscale=stix_spec_ylogscale,
                )
                fig.add_trace(spec_fig.data[0], row=row, col=1, secondary_y=False)
                _copy_yaxis(fig, spec_fig.layout.yaxis, row)
                if ptype == "overlay":
                    for tr in stix_counts_traces(
                        stix_counts, integrate_bins=stix_energy_bins, smoothing_pts=stix_smoothing_points, lw=linewidth
                    ):
                        fig.add_trace(tr, row=row, col=1, secondary_y=True)
                    fig.update_yaxes(
                        type="log" if stix_curves_ylogscale else "linear", row=row, col=1, secondary_y=True
                    )
            elif ptype == "curve":
                for tr in stix_counts_traces(
                    stix_counts,
                    integrate_bins=stix_energy_bins,
                    date_range=d_range,
                    smoothing_pts=stix_smoothing_points,
                    lw=linewidth,
                ):
                    fig.add_trace(tr, row=row, col=1, secondary_y=False)
                fig.update_yaxes(type="log" if stix_curves_ylogscale else "linear", row=row, col=1, secondary_y=False)

        elif origin == "epd":
            epd_fig = epd_flux_figure(
                epd_data,
                epd_energies,
                particle=epd_particle,
                channels=epd_channels,
                date_range=d_range,
                round_epd_label=epd_round_label,
                resample=epd_resample,
            )
            for tr in epd_fig.data:
                fig.add_trace(tr, row=row, col=1, secondary_y=False)
            _copy_yaxis(fig, epd_fig.layout.yaxis, row)

    fig.update_xaxes(range=[start, end], showgrid=True)
    fig.update_xaxes(title_text=label_obstime, row=n_plots, col=1)
    fig.update_layout(
        height=max(320, height_per_panel * n_plots),
        font=dict(size=fontsize),
        showlegend=True,
        margin=dict(l=70, r=60, t=30, b=60),
    )
    return fig
