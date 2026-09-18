"""Dependency-installation check for SoloLab.

Run this after `pip install -r requirements.txt` (desktop app) or
`pip install -r requirements-dash.txt` (web app) to confirm everything is
importable and minimally functional - no GUI, no network access, and no
sample data files required.

    python test_installation.py

Unlike test_sololabapp.py (which launches the desktop GUI and only proves
"it didn't crash on startup"), this reports pass/fail per package, grouped
by which app needs it, and exits non-zero if anything *required* for the
app(s) actually installed in this environment is missing. It is normal and
expected for the "desktop-only" group to fail in a `requirements-dash.txt`
environment, and vice versa - the summary reflects that instead of treating
it as an error.

See INSTALL.md for setup instructions.
"""
import importlib
import sys
import traceback

# name -> (import name, PyPI/friendly display name)
CORE_PACKAGES = [
    ("numpy", "numpy"),
    ("scipy", "scipy"),
    ("pandas", "pandas"),
    ("matplotlib", "matplotlib"),
    ("astropy", "astropy"),
    ("sunpy", "sunpy"),
    ("h5py", "h5py"),  # not used by sololab directly - avoids a SunpyUserWarning solo-epd-loader triggers on import
    ("cdflib", "cdflib"),
    ("solo_epd_loader", "solo-epd-loader"),
    ("seaborn", "seaborn"),
]
DESKTOP_PACKAGES = [
    ("qtpy", "qtpy"),
    ("PyQt5", "PyQt5"),
]
WEB_PACKAGES = [
    ("dash", "dash"),
    ("dash_bootstrap_components", "dash-bootstrap-components"),
    ("plotly", "plotly"),
    ("diskcache", "diskcache"),
]
DEPLOY_PACKAGES = [
    ("gunicorn", "gunicorn"),  # only used in production deployment, not local runs
]
OPTIONAL_PACKAGES = [
    # Only needed for sololab.stix_read.stix_query_science_files/stix_download_file
    # (the desktop app's "Download from STIX Data Center..." button) - everything
    # else works fine without it.
    ("stixdcpy", "stixdcpy"),
]

GROUPS = [
    ("Core (both apps)", CORE_PACKAGES, True),
    ("Desktop app only (PyQt5/qtpy)", DESKTOP_PACKAGES, False),
    ("Web app only (Dash)", WEB_PACKAGES, False),
    ("Deployment only (gunicorn)", DEPLOY_PACKAGES, False),
    ("Optional features (STIX Data Center download)", OPTIONAL_PACKAGES, False),
]


def _check_import(import_name):
    try:
        mod = importlib.import_module(import_name)
    except Exception as exc:  # noqa: BLE001 - report *any* import failure, not just ImportError
        return False, str(exc)
    version = getattr(mod, "__version__", None)
    return True, version


def _print_group(title, packages, required):
    print(f"\n{title}")
    print("-" * len(title))
    n_ok = 0
    for import_name, display_name in packages:
        ok, info = _check_import(import_name)
        if ok:
            n_ok += 1
            version_str = f" ({info})" if info else ""
            print(f"  [OK]      {display_name}{version_str}")
        else:
            print(f"  [MISSING] {display_name} - {info}")
    return n_ok, len(packages)


def _check_sololab_package():
    print("\nSoloLab package")
    print("-" * len("SoloLab package"))
    results = {}
    for mod_name in ["sololab", "sololab.values", "sololab.stix_read", "sololab.rpw_read", "sololab.quicklooks"]:
        try:
            importlib.import_module(mod_name)
            print(f"  [OK]      {mod_name}")
            results[mod_name] = True
        except Exception:  # noqa: BLE001
            print(f"  [FAILED]  {mod_name}")
            traceback.print_exc(limit=3)
            results[mod_name] = False

    # sololab_app (desktop GUI) - only meaningful if PyQt5/qtpy are present;
    # sololab/__init__.py already degrades gracefully without them, so this
    # is informational, not a failure.
    try:
        import sololab

        has_run_app = hasattr(sololab, "run_app")
        print(f"  [{'OK' if has_run_app else 'INFO'}]      sololab.run_app available: {has_run_app}"
              f"{'' if has_run_app else ' (expected if PyQt5/qtpy are not installed in this environment)'}")
    except Exception:  # noqa: BLE001
        pass

    # dash_app - only meaningful if dash is present.
    try:
        importlib.import_module("dash")
    except Exception:  # noqa: BLE001
        print("  [INFO]    skipping sololab.dash_app import check (dash not installed in this environment)")
    else:
        try:
            importlib.import_module("sololab.dash_app.plotting")
            print("  [OK]      sololab.dash_app.plotting")
            results["sololab.dash_app.plotting"] = True
        except Exception:  # noqa: BLE001
            print("  [FAILED]  sololab.dash_app.plotting")
            traceback.print_exc(limit=3)
            results["sololab.dash_app.plotting"] = False

    return results


def _run_smoke_checks():
    """A handful of tiny, offline computations that exercise the imported
    libraries beyond a bare `import` - catches "importable but broken"
    installs (e.g. a numpy/scipy ABI mismatch) without needing any sample
    data files."""
    print("\nSmoke checks")
    print("-" * len("Smoke checks"))
    checks = []

    try:
        import numpy as np

        assert np.array([1, 2, 3]).sum() == 6
        checks.append(("numpy array math", True, None))
    except Exception as exc:  # noqa: BLE001
        checks.append(("numpy array math", False, str(exc)))

    try:
        from astropy import constants as const
        from astropy import units as u

        assert const.c.to(u.km / u.s).value > 0
        checks.append(("astropy constants/units", True, None))
    except Exception as exc:  # noqa: BLE001
        checks.append(("astropy constants/units", False, str(exc)))

    try:
        import pandas as pd

        idx = pd.date_range("2024-01-01", periods=3, freq="min")
        assert len(idx) == 3
        checks.append(("pandas date range", True, None))
    except Exception as exc:  # noqa: BLE001
        checks.append(("pandas date range", False, str(exc)))

    try:
        from sololab.values import get_poll_func

        assert get_poll_func("mean")([1, 2, 3]) == 2
        checks.append(("sololab.values.get_poll_func", True, None))
    except Exception as exc:  # noqa: BLE001
        checks.append(("sololab.values.get_poll_func", False, str(exc)))

    for name, ok, err in checks:
        print(f"  [{'OK' if ok else 'FAILED'}]  {name}" + (f" - {err}" if err else ""))

    return all(ok for _, ok, _ in checks)


def main():
    print("=" * 60)
    print("SoloLab installation check")
    print(f"Python {sys.version.split()[0]} at {sys.executable}")
    print("=" * 60)

    group_results = []
    for title, packages, required in GROUPS:
        n_ok, n_total = _print_group(title, packages, required)
        group_results.append((title, n_ok, n_total, required))

    sololab_results = _check_sololab_package()
    smoke_ok = _run_smoke_checks()

    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    core_ok = True
    for title, n_ok, n_total, required in group_results:
        status = f"{n_ok}/{n_total} available"
        print(f"  {title}: {status}")
        if required and n_ok < n_total:
            core_ok = False

    sololab_ok = all(sololab_results.values()) if sololab_results else False
    print(f"  SoloLab package imports: {'OK' if sololab_ok else 'FAILED'}")
    print(f"  Smoke checks: {'OK' if smoke_ok else 'FAILED'}")

    desktop_ok = any(_check_import(name)[0] for name, _ in DESKTOP_PACKAGES)
    web_ok = any(_check_import(name)[0] for name, _ in WEB_PACKAGES)
    if not desktop_ok and not web_ok:
        print(
            "\n  [!] Neither the desktop (PyQt5/qtpy) nor the web (Dash) packages "
            "are installed - install requirements.txt or requirements-dash.txt."
        )

    success = core_ok and sololab_ok and smoke_ok
    print(f"\nOverall: {'PASS' if success else 'FAIL'}")
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
