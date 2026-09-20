"""Save/load the current session's loaded instrument data as a pickle
blob - port of MainWindow._save_data_pack / _load_data_pack. Pure
serialization logic; pages/data_pack_page.py wires this to
dcc.Download/dcc.Upload.

Unlike the desktop version's payload (which the migration plan noted as an
optional nice-to-have for cross-compatibility), this payload isn't byte-
identical to the PyQt5 app's .pkl format: it stores each instrument's
{"counts"|"data": ..., "meta": {...}} using the same meta fields this app's
import pages actually track (bkg_file_enabled/bkg_time_enabled/
bkg_poll_function for STIX, bkg_option/bkg_poll_function for RPW), not the
desktop's separate bkg_start/bkg_end/bkg_file keys. A .pkl saved by one is
therefore not loadable by the other.
"""
import gzip
import pickle
from datetime import datetime

from sololab.dash_app.constants import rpw_key
from sololab.dash_app.session_store import session_store
from sololab.dash_app.utils import format_dt_input


def build_payload(sid):
    return {
        "stix": {
            "counts": session_store.get(sid, "stix_counts_final"),
            "meta": session_store.get(sid, "stix_meta"),
        },
        "rpw_hfr": {
            "data": session_store.get(sid, rpw_key("hfr", "psd_final")),
            "meta": session_store.get(sid, rpw_key("hfr", "meta")),
        },
        "rpw_tnr": {
            "data": session_store.get(sid, rpw_key("tnr", "psd_final")),
            "meta": session_store.get(sid, rpw_key("tnr", "meta")),
        },
        "epd": {
            "meta": session_store.get(sid, "epd_meta"),
            "final_protons": session_store.get(sid, "epd_protons_final"),
            "final_electrons": session_store.get(sid, "epd_electrons_final"),
            "energies": session_store.get(sid, "epd_energies"),
            # unlike STIX/RPW, EPD's background isn't embedded in the
            # final data itself - it lives in its own session_store slot
            # (see import_epd.py's epd_load_click/epd_preview_with_bkg),
            # so it needs saving separately too.
            "background": session_store.get(sid, "epd_background"),
            "background_std": session_store.get(sid, "epd_background_std"),
        },
    }


def payload_to_bytes(payload):
    """gzip-compressed pickle - the payload is mostly numpy arrays/pandas
    DataFrames (STIX counts, RPW PSDs, a full day of EPD per-channel flux),
    which compress well (repeated/near-zero values, shared dtypes)."""
    return gzip.compress(pickle.dumps(payload, protocol=pickle.HIGHEST_PROTOCOL))


def bytes_to_payload(blob):
    """Sniffs the gzip magic bytes so a .pkl saved before compression was
    added still loads."""
    if blob[:2] == b"\x1f\x8b":
        blob = gzip.decompress(blob)
    return pickle.loads(blob)


def apply_payload(sid, payload):
    """Writes the payload into session_store and returns the new
    instrument-status-store data (merging is the caller's job, since it
    also holds the current State)."""
    status = {}

    stix = payload.get("stix") or {}
    if stix.get("counts") is not None:
        counts = stix["counts"]
        meta = stix.get("meta") or {}
        session_store.set(sid, "stix_counts_final", counts)
        session_store.set(sid, "stix_meta", meta)
        status["stix"] = {
            "loaded": True,
            "min_time": format_dt_input(min(counts["time"])),
            "max_time": format_dt_input(max(counts["time"])),
            "bkg_enabled": bool(meta.get("bkg_file_enabled")) or bool(meta.get("bkg_time_enabled")),
        }

    for data_type, status_key in (("hfr", "rpw_hfr"), ("tnr", "rpw_tnr")):
        entry = payload.get(status_key) or {}
        if entry.get("data") is not None:
            psd = entry["data"]
            meta = entry.get("meta") or {}
            session_store.set(sid, rpw_key(data_type, "psd_final"), psd)
            session_store.set(sid, rpw_key(data_type, "meta"), meta)
            status[status_key] = {
                "loaded": True,
                "min_time": format_dt_input(min(psd["time"])),
                "max_time": format_dt_input(max(psd["time"])),
                "bkg_enabled": meta.get("bkg_option") == 1,
            }

    epd = payload.get("epd") or {}
    if epd.get("energies") is not None:
        meta = epd.get("meta") or {}
        # "final_*" is the actual plottable data (post background
        # subtraction, if any); fall back to the older pre-fix pack format's
        # raw "df_*" so a pack saved before this fix still loads (without
        # its background subtraction, since the bkg time range was never
        # saved either way).
        final_protons = epd.get("final_protons", epd.get("df_protons"))
        final_electrons = epd.get("final_electrons", epd.get("df_electrons"))
        if final_protons is not None:
            session_store.set(sid, "epd_protons_final", final_protons)
        if final_electrons is not None:
            session_store.set(sid, "epd_electrons_final", final_electrons)
        session_store.set(sid, "epd_energies", epd.get("energies"))
        session_store.set(sid, "epd_meta", meta)
        if epd.get("background") is not None:
            session_store.set(sid, "epd_background", epd["background"])
        if epd.get("background_std") is not None:
            session_store.set(sid, "epd_background_std", epd["background_std"])
        status["epd"] = {
            "loaded": True,
            "date": meta.get("date"),
            "particle": meta.get("particle"),
            "resample": meta.get("resample"),
        }

    return status


def suggested_filename(status):
    status = status or {}
    tags = [
        code
        for code, key in (("stix", "stix"), ("hfr", "rpw_hfr"), ("tnr", "rpw_tnr"), ("epd", "epd"))
        if status.get(key, {}).get("loaded")
    ]
    tag_str = "_".join(tags) if tags else "none"
    return f"sololab_{datetime.now().strftime('%Y%m%dT%H%M')}_{tag_str}.pkl"
