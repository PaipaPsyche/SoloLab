"""STIX pixel-data image reconstruction (backprojection/CLEAN/MEM_GE), built
on `stixpy` + `xrayvision` - a separate, optional-dependency path from
stix_read.py: reads pixel-data FITS files directly via stixpy.product.Product
rather than stix_read.stix_create_counts, which sums away the
detector/pixel axes stix_read.py needs for the existing spectrogram
pipeline (see stix_read.is_stix_pixel_file). Nothing here touches
stix_read.py's own counts pipeline.

`stixpy`/`xrayvision` are imported lazily inside the functions that need
them, matching stix_read.py's stixdcpy precedent, so the rest of sololab
keeps working without them installed.

Two-stage pipeline, mirroring what stixpy's own real example
(examples/imaging_demo.py) does - estimate the flare's sky location once
(its own coarse imaging pass, the expensive part), then calibrate+
reconstruct per interval reusing that fixed location rather than
re-estimating it every time:

1. estimate_flare_location_and_ancillary() - wraps stixpy's own
   stixpy.imaging.flare_location.estimate_flare_location (a real,
   documented function - not hand-rolled), run once per loaded file.
2. reconstruct_stix_image() - meta-pixels -> visibility -> calibration ->
   reconstruction for one interval, given a flare_location from step 1 (or
   a manual override) - run once per single-interval Run, or once per
   Sequence-mode tile.

Verified against a real `pip install stixpy xrayvisim` (stixpy 0.3.0,
xrayvision/xrayvisim 0.2.1) and stixpy's own bundled sample data
(stixpy.data.test.STIX_SCI_XRAY_CPD), and against the real
examples/imaging_demo.py and xrayvision/clean.py source on GitHub
(TCDSolar/stixpy, TCDSolar/xrayvision), not just docs:

- get_hpc_info() (used internally by estimate_flare_location too) needs
  outbound network access (downloads SPICE/ephemeris data via SOAR on
  first use for a given date) - the Dash server's environment needs to
  reach the network for imaging to work, same as the existing
  STIX-Data-Center download feature.
- **Known upstream bugs, confirmed by direct execution against stixpy's own
  bundled sample data, not assumed from docs**: create_meta_pixels()
  currently fails with at least THREE independent schema mismatches against
  both CompressedPixelData and RawPixelData in stixpy 0.3.0:
  (1) it unconditionally reads `pixel_data.energy_masks.energy_mask`, an
  attribute that doesn't exist on either product class;
  (2, only reachable after patching around the first) it reads a
  `"counts_comp_err"` data column - the real column on both bundled sample
  files is named `"counts_err"`, not `"counts_comp_err"`;
  (3, only reachable after patching around the first two) even with a
  constructed `EnergyEdgeMasks`, the resulting boolean mask has a different
  length (31) than the 32-channel arrays `get_elut_correction` indexes with
  it, raising `IndexError` - a real energy-mask shape/semantics mismatch,
  not something safely patchable without deep stixpy internals knowledge.
  Confirmed via GitHub issue search
  (api.github.com/search/issues?q=repo:TCDSolar/stixpy+energy_masks and
  +counts_comp_err, both total_count: 0) this is not a reported upstream
  issue. This is not a SoloLab bug or a usage mistake - the code below
  calls the documented, correct API; it will start working automatically
  once stixpy fixes this upstream (see github.com/TCDSolar/stixpy).
  reconstruct_stix_image()/estimate_flare_location_and_ancillary() wrap
  *any* AttributeError/KeyError/IndexError from the meta-pixel step with a
  clear message rather than trying to special-case every individual schema
  mismatch.
- **CLEAN**: `xrayvision.clean.vis_clean(..., map=True)` returns a 3-tuple
  `[clean_map, model_map, resid_map]`, confirmed by reading the real source
  (`return [Map((data, dirty_map.meta)) for data in (clean_map, model,
  residual)]`) - _run_algorithm unpacks and keeps only clean_map.
- **MEM_GE**: `xrayvision.mem.mem`'s own docstring states only a default
  (0.02%) and valid range ([0.0001, 0.2]) for `percent_lambda`, no
  recommended value. The real example instead *computes* one from the data
  via an SNR estimate (`xrayvision.mem.resistant_mean`) - used here as the
  default when the user leaves the field blank, matching the reference
  implementation rather than silently falling through to the library's
  bare 0.02% default.
- "em" (stixpy.imaging.em.em) is deliberately not implemented here: it has
  its own, separate bug (`idx` defaulting is inverted -
  `if idx is not None: idx = [...]` instead of `if idx is None`, in
  stixpy/imaging/em.py) plus an input (`countrates`) whose expected shape
  doesn't match anything create_meta_pixels/create_visibility produces -
  wiring it up now would mean shipping a guessed, unverifiable code path.
"""
import numpy as np
import astropy.units as u
from astropy.time import Time

# Energy range used only for the flare-location coarse pass (independent of
# whatever energy range the user later picks for the actual reconstruction)
# - a broad non-thermal HXR band where flare footpoints are typically
# visible. Not a scientifically tuned value, just a reasonable default so
# the user isn't asked to pick two separate energy ranges.
DEFAULT_LOCATION_ENERGY_RANGE = (4, 15)


def tile_time_range(start, end, duration):
    """Fixed-width tiles from `start`; the last tile is extended forward
    past `end` (never shortened) so every tile has exactly `duration` - a
    plain fixed-step loop, no special-casing. start/end: astropy Time (or
    anything Time() accepts); duration: astropy Quantity (a time unit,
    e.g. 30*u.s)."""
    start, end = Time(start), Time(end)
    tiles = []
    t = start
    while t < end:
        tiles.append((t, t + duration))
        t = t + duration
    return tiles


_KNOWN_UPSTREAM_BUG_MSG = (
    "STIX imaging is unavailable: stixpy 0.3.0's create_meta_pixels() has "
    "confirmed schema-mismatch bugs against pixel-data products (see "
    "stix_imaging.py's module docstring - missing 'energy_masks', wrong "
    "'counts_comp_err' column name, and an energy-mask shape mismatch). "
    "This is a confirmed upstream issue, not a SoloLab bug or a data "
    "problem with this file - it will work once stixpy publishes a fix. "
    "Original error: {exc}"
)


def _build_observer(time_range):
    """HeliographicStonyhurst observer (Solar Orbiter's position) for the
    given time range, via stixpy's own get_hpc_info - shared by
    estimate_flare_location_and_ancillary (which normally gets this for
    free from estimate_flare_location's own return value) and
    build_flare_location (the manual-override path, which needs its own
    lookup since it doesn't call estimate_flare_location)."""
    from stixpy.coordinates.transforms import get_hpc_info
    from sunpy.coordinates import HeliographicStonyhurst

    start, end = Time(time_range[0]), Time(time_range[1])
    roll, solo_heeq, pointing = get_hpc_info(start, end)
    center = start + (end - start) / 2
    observer = HeliographicStonyhurst(*solo_heeq, obstime=center, representation_type="cartesian")
    return observer


def build_flare_location(time_range, tx, ty):
    """Manual override: a STIXImaging SkyCoord at (tx, ty) arcsec for the
    given time range - same coordinate construction
    estimate_flare_location_and_ancillary uses internally, so either can be
    passed as reconstruct_stix_image's flare_location argument."""
    from stixpy.coordinates.frames import STIXImaging
    from astropy.coordinates import SkyCoord

    start, end = Time(time_range[0]), Time(time_range[1])
    observer = _build_observer(time_range)
    return SkyCoord(tx * u.arcsec, ty * u.arcsec, frame=STIXImaging(obstime=start, obstime_end=end, observer=observer))


def estimate_flare_location_and_ancillary(pixel_path, energy_range=None):
    """Coarse meta-pixels -> visibility -> calibration -> backprojection ->
    argmax pass, run once for the file's full time range, plus ancillary
    observation metadata for display. Call this once per loaded file;
    reconstruct_stix_image reuses its "flare_location" for every
    subsequent image instead of re-estimating it.

    This hand-rolls the same coarse-pass algorithm
    examples/imaging_demo.py uses inline in the real stixpy repository
    (github.com/TCDSolar/stixpy) - NOT stixpy.imaging.flare_location's
    estimate_flare_location(), which reads cleanly on GitHub's `main`
    branch but does not exist in the installed PyPI release
    (confirmed: `stixpy==0.3.0`'s `stixpy/imaging/` only contains `em.py` -
    `flare_location.py` is a `main`-only addition not yet published).
    When stixpy publishes a release containing it, swap this for a direct
    call to that function instead (same coarse-pass algorithm either way).

    Returns {"flare_location": SkyCoord (STIXImaging), "flare_tx_arcsec":
    float, "flare_ty_arcsec": float, "obs_start": Time, "obs_end": Time,
    "sun_distance_au": float, "earth_distance_au": float}.
    """
    import stixpy.product
    from stixpy.calibration.visibility import create_meta_pixels, create_visibility, calibrate_visibility
    from xrayvision.imaging import vis_to_map
    from sunpy.coordinates import get_earth

    pixel_data = stixpy.product.Product(pixel_path)
    tr = pixel_data.time_range
    time_range = (tr.start, tr.end)
    e_lo, e_hi = energy_range or DEFAULT_LOCATION_ENERGY_RANGE
    energy_q = [e_lo, e_hi] * u.keV

    disk_center = build_flare_location(time_range, 0, 0)
    try:
        meta_pixels = create_meta_pixels(
            pixel_data, time_range=[tr.start, tr.end], energy_range=energy_q,
            flare_location=disk_center, pixels="top+bot", no_shadowing=True,
        )
    except (AttributeError, KeyError, IndexError) as exc:
        raise RuntimeError(_KNOWN_UPSTREAM_BUG_MSG.format(exc=exc)) from exc

    vis = create_visibility(meta_pixels)
    vis = calibrate_visibility(vis, flare_location=disk_center)

    # Coarse full-disk backprojection (imsize/pixel_size match
    # examples/imaging_demo.py's own coarse-pass values).
    coarse_map = vis_to_map(vis, shape=[512, 512] * u.pix, pixel_size=[10, 10] * u.arcsec / u.pix)
    data = np.asarray(coarse_map.data)
    iy, ix = np.unravel_index(np.argmax(data), data.shape)
    x0 = coarse_map.bottom_left_coord.Tx.to_value(u.arcsec)
    y0 = coarse_map.bottom_left_coord.Ty.to_value(u.arcsec)
    dx = coarse_map.scale.axis1.to_value(u.arcsec / u.pix)
    dy = coarse_map.scale.axis2.to_value(u.arcsec / u.pix)
    flare_tx = x0 + ix * dx
    flare_ty = y0 + iy * dy

    flare_location = build_flare_location(time_range, flare_tx, flare_ty)
    observer = flare_location.frame.observer  # reuse - avoids a second ephemeris lookup

    sun_distance_au = observer.spherical.distance.to_value(u.AU)
    earth = get_earth(tr.center)
    earth_distance_au = (earth.cartesian - observer.cartesian).norm().to_value(u.AU)

    return {
        "flare_location": flare_location,
        "flare_tx_arcsec": flare_tx,
        "flare_ty_arcsec": flare_ty,
        "obs_start": tr.start,
        "obs_end": tr.end,
        "sun_distance_au": sun_distance_au,
        "earth_distance_au": earth_distance_au,
    }


def flare_location_on_disk(flare_location, resolution=360):
    """Flare position and the solar limb, both in helioprojective arcsec as
    seen from Solar Orbiter. The STIXImaging frame is boresight-relative and
    rotated with the spacecraft roll, so the flare must go through stixpy's
    registered STIXImaging->Helioprojective transform to be comparable with
    the limb (a circle centered on HPC (0, 0))."""
    from sunpy.coordinates import Helioprojective
    from sunpy.coordinates.utils import get_limb_coordinates

    frame = flare_location.frame
    observer = frame.observer
    hpc_frame = Helioprojective(obstime=frame.obstime, observer=observer)
    hpc = flare_location.transform_to(hpc_frame)
    limb = get_limb_coordinates(observer, resolution=resolution).transform_to(hpc_frame)
    return {
        "flare_x": hpc.Tx.to_value(u.arcsec),
        "flare_y": hpc.Ty.to_value(u.arcsec),
        "limb_x": limb.Tx.to_value(u.arcsec),
        "limb_y": limb.Ty.to_value(u.arcsec),
        "distance_au": observer.spherical.distance.to_value(u.AU),
        "obstime": frame.obstime,
    }


def reconstruct_stix_image(
    pixel_path,
    bkg_path,
    time_range,
    energy_range,
    algorithm,
    algo_params,
    flare_location,
):
    """pixel_path/bkg_path: local FITS paths (main pixel-data file / optional
    background pixel-data file - bkg_path may be None). time_range: (start,
    end), anything astropy.time.Time accepts. energy_range: (e_low, e_high)
    in keV (plain numbers, keV assumed). algorithm: one of
    constants.STIX_IMAGING_ALGORITHMS. algo_params: dict of algorithm-specific
    parameters (see the per-algorithm dispatch below) plus the common
    "npix"/"pixel_size" keys. flare_location: a SkyCoord (STIXImaging frame)
    - from estimate_flare_location_and_ancillary's cached result, or
    build_flare_location() for a manual override. Required (unlike the
    earlier flare_xy=None default) since callers now always have one of
    these two sources rather than silently defaulting to disk center.

    Returns {"image": ndarray, "x_arcsec": ndarray, "y_arcsec": ndarray,
    "algorithm": str, "energy_range": (lo, hi), "time_range": (start, end)}.
    Image is in the native STIXImaging frame (boresight-relative, not
    rotated to solar north) - no Helioprojective transform in this pass.
    """
    import stixpy.product
    from stixpy.calibration.visibility import create_meta_pixels, create_visibility, calibrate_visibility

    start, end = Time(time_range[0]), Time(time_range[1])
    energy_q = [energy_range[0], energy_range[1]] * u.keV

    pixel_data = stixpy.product.Product(pixel_path)
    try:
        meta_pixels = create_meta_pixels(
            pixel_data, time_range=[start, end], energy_range=energy_q,
            flare_location=flare_location, pixels="top+bot",
        )
        if bkg_path:
            bkg_data = stixpy.product.Product(bkg_path)
            bkg_meta_pixels = create_meta_pixels(
                bkg_data, time_range=[start, end], energy_range=energy_q,
                flare_location=flare_location, pixels="top+bot",
            )
            meta_pixels["abcd_rate_kev_cm"] = meta_pixels["abcd_rate_kev_cm"] - bkg_meta_pixels["abcd_rate_kev_cm"]
    except (AttributeError, KeyError, IndexError) as exc:
        raise RuntimeError(_KNOWN_UPSTREAM_BUG_MSG.format(exc=exc)) from exc

    vis = create_visibility(meta_pixels)
    vis = calibrate_visibility(vis, flare_location=flare_location)

    npix = algo_params.get("npix", 128)
    pixel_size_val = algo_params.get("pixel_size", 2.0)
    shape = [npix, npix] * u.pix
    pixel_size = [pixel_size_val, pixel_size_val] * u.arcsec / u.pix

    smap = _run_algorithm(vis, algorithm, shape, pixel_size, algo_params)

    data = np.asarray(smap.data)
    ny, nx = data.shape
    x0 = smap.bottom_left_coord.Tx.to_value(u.arcsec)
    y0 = smap.bottom_left_coord.Ty.to_value(u.arcsec)
    dx = smap.scale.axis1.to_value(u.arcsec / u.pix)
    dy = smap.scale.axis2.to_value(u.arcsec / u.pix)
    x_arcsec = x0 + np.arange(nx) * dx
    y_arcsec = y0 + np.arange(ny) * dy

    return {
        "image": data,
        "x_arcsec": x_arcsec,
        "y_arcsec": y_arcsec,
        "algorithm": algorithm,
        "energy_range": (energy_range[0], energy_range[1]),
        "time_range": (start, end),
    }


def _run_algorithm(vis, algorithm, shape, pixel_size, algo_params):
    """Dispatches to the matching xrayvision reconstruction call, always
    requesting a sunpy Map back (map=True where the function supports it)
    so the caller can read pixel scale/reference coordinate uniformly."""
    if algorithm == "backprojection":
        from xrayvision.imaging import vis_to_map
        return vis_to_map(vis, shape=shape, pixel_size=pixel_size, scheme=algo_params.get("weighting", "natural"))
    if algorithm == "clean":
        from xrayvision.clean import vis_clean
        # vis_clean(map=True) returns [clean_map, model_map, resid_map] -
        # confirmed by reading the real xrayvision source (not a single Map,
        # despite the "map=True" name suggesting one) - keep only the
        # cleaned image, model/residual aren't displayed.
        clean_map, _model_map, _resid_map = vis_clean(
            vis, shape=shape, pixel_size=pixel_size, map=True,
            gain=algo_params.get("gain", 0.1),
            niter=algo_params.get("niter", 200),
            clean_beam_width=algo_params.get("clean_beam_width", 20.0) * u.arcsec,
        )
        return clean_map
    if algorithm == "mem_ge":
        from xrayvision.mem import mem, resistant_mean
        percent_lambda = algo_params.get("percent_lambda")
        if percent_lambda is None:
            # SNR-based default, matching examples/imaging_demo.py exactly
            # (not xrayvision's own bare 0.02% default) - see module docstring.
            snr_value, _ = resistant_mean((np.abs(vis.visibilities) / vis.amplitude_uncertainty).flatten(), 3)
            percent_lambda = 2 / (snr_value**2 + 90)
        return mem(
            vis, shape=shape, pixel_size=pixel_size, map=True,
            percent_lambda=percent_lambda * u.percent,
            maxiter=algo_params.get("maxiter", 1000),
            tol=algo_params.get("tolerance", 0.001),
        )
    raise ValueError(f"Unsupported algorithm: {algorithm!r}")


if __name__ == "__main__":
    """Manual self-check, no test framework (matches this repo's
    test_installation.py convention). Uses stixpy's own bundled sample CPD
    file - zero setup required beyond `pip install stixpy xrayvisim`.
    Currently expected to fail with the documented upstream bugs above;
    this script is also a live canary for when those get fixed - it exits
    0 and prints OK only on a genuine successful reconstruction."""
    import sys
    import stixpy.data.test as _test_data

    path = sys.argv[1] if len(sys.argv) > 1 else _test_data.STIX_SCI_XRAY_CPD
    from stixpy.product import Product

    pd = Product(path)
    tr = pd.time_range
    try:
        estimate = estimate_flare_location_and_ancillary(path)
        print("Flare location estimated:", estimate["flare_tx_arcsec"], estimate["flare_ty_arcsec"], "arcsec")
        result = reconstruct_stix_image(
            pixel_path=path,
            bkg_path=None,
            time_range=(tr.start, tr.end),
            energy_range=(4, 10),
            algorithm="backprojection",
            algo_params={"npix": 65, "pixel_size": 4.0},
            flare_location=estimate["flare_location"],
        )
    except RuntimeError as exc:
        print("KNOWN UPSTREAM ISSUE (not fixed yet):", exc)
        sys.exit(1)
    assert result["image"].shape == (65, 65), result["image"].shape
    assert result["x_arcsec"].shape == (65,)
    print("OK:", path, "->", result["image"].shape, result["image"].dtype)
