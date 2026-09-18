# SoloLab v0.4

**Python tool for multi-instrument analysis of [Solar Orbiter](https://www.esa.int/Science_Exploration/Space_Science/Solar_Orbiter) data**, combining measurements from:

- **[STIX](https://solar-orbiter.cnes.fr/en/SOLO/GP_stix.htm)** — Spectrometer Telescope for Imaging X-rays
- **[RPW](https://rpw.lesia.obspm.fr/)** — Radio and Plasma Waves
- **[EPD](https://espada.uah.es/epd/index.php)** — Energetic Particles Detector

Correlating X-ray, radio, and particle time series from the same event helps trace particle
acceleration and transport during solar flares, and supports broader heliophysics work beyond
transient events.

## Two interfaces, one core

- **Desktop app** (`sololab/sololab_app.py`, PyQt) — the original GUI: import, preview, and plot
  each instrument, including combined multi-instrument plots.
- **Web app** (`sololab/dash_app/`, Dash) — the same workflow in a browser, deployable to a server.

Both sit on top of the same `sololab` package, so scripting/notebook use doesn't require either
GUI — see [`sololab_examples.ipynb`](sololab_examples.ipynb).

## Features

- **Data import** — STIX (FITS: spectrograms, L1 pixel data, background files; direct download
  from the [STIX Data Center](https://datacenter.stix.i4ds.net/)); RPW (CDF: HFR/TNR, L2 and L3;
  direct download from [CDAWeb](https://cdaweb.gsfc.nasa.gov/)); EPD (L2, auto-downloaded via
  [`solo-epd-loader`](https://github.com/jgieseler/solo-epd-loader)).
- **Processing** — STIX background subtraction (BKG file and/or quiet-time interval) with energy
  shifts; RPW background subtraction and polluted-frequency filtering.
- **Visualization** — spectrograms and per-channel time profiles for any instrument, combined into
  one multi-panel plot with a shared time axis.
- **Estimations and fits** — Frequency Drift Rate Analysis (radio burst exciter velocity) and
  electron abundance vs. threshold energy from STIX powerlaws (`sololab.freqs_drift`,
  `sololab.electron_powerlaw`; still uder revision;  called directly, not yet wired into either GUI).

## Quick start

See [`INSTALL.md`](INSTALL.md) for setup on Windows/macOS/Linux (no compiler needed). Then:

```bash
python test_installation.py      # verify your environment
python -m sololab.dash_app.app   # web app, or: python -c "import sololab; sololab.run_app()"
```

Usage examples (reading files, plotting, combined plots): [`sololab_examples.ipynb`](sololab_examples.ipynb).

> `sololab_tutorial.ipynb` is outdated and no longer works — use the notebook above instead.

## Contact

Created by **David Paipa**, LIRA, Observatoire de Meudon — [contact](mailto:david.paipa@obspm.fr).
MIT licensed (see [`LICENSE`](LICENSE)). Not an official Solar Orbiter ground software product;
originally developed for the author's PhD thesis. Questions, suggestions, and bug reports welcome.
