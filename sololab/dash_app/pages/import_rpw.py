"""Registers /import/rpw-hfr and /import/rpw-tnr, both built from the
shared rpw_import_factory (see that module for why HFR/TNR share one
implementation instead of two near-duplicate files)."""
import dash

from sololab.dash_app.rpw_import_factory import make_rpw_import_layout, register_rpw_import_callbacks

dash.register_page(
    "import_rpw_hfr",
    path="/import/rpw-hfr",
    name="Import RPW-HFR",
    layout=lambda **kwargs: make_rpw_import_layout("hfr"),
)
dash.register_page(
    "import_rpw_tnr",
    path="/import/rpw-tnr",
    name="Import RPW-TNR",
    layout=lambda **kwargs: make_rpw_import_layout("tnr"),
)

register_rpw_import_callbacks("hfr")
register_rpw_import_callbacks("tnr")
