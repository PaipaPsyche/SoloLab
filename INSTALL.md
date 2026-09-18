# Installation Guide

SoloLab ships as **two separate apps** that share the same core Python package
(`sololab/`), each with its own dependency list:

| App | Dependency file | What it needs |
|---|---|---|
| **Desktop app** (`sololab_app.py`, PyQt5/qtpy GUI) | [`requirements.txt`](requirements.txt) | A local display; no server. |
| **Web app** (`sololab/dash_app/`, Dash) | [`requirements-dash.txt`](requirements-dash.txt) | No GUI toolkit - runs in a browser, deployable to a server. |

You only need to install the one you plan to use (or both, in **separate virtual
environments** - `requirements-dash.txt` deliberately excludes PyQt5/qtpy/spacepy
so the web app stays light and deployable to a headless server).

Everything below uses `pip`-installable **binary wheels** - on a supported Python
version (3.10-3.12 recommended; see the note on Windows below) you never need a
C/C++ compiler to install SoloLab's dependencies.

---

## 1. Get Python (only if you don't have it)

Check first - you might already have it:

```bash
python3 --version   # macOS / Linux
python --version    # Windows
```

If that prints `Python 3.10` through `3.12`, skip to [step 2](#2-create-a-virtual-environment).
Anything older than 3.9 won't work; anything newer than what's listed on
[python.org/downloads](https://www.python.org/downloads/) as a stable release may
not have wheels yet for every scientific package SoloLab depends on.

### Windows

**No admin rights needed** for either of these:

- **Recommended:** download the installer from
  [python.org/downloads/windows](https://www.python.org/downloads/windows/) (pick
  the latest **3.12.x** "Windows installer (64-bit)"), run it, and on the first
  screen **check "Add python.exe to PATH"** before clicking Install. Choose
  "Install for me only" if you're offered a per-user vs. all-users choice - that
  path doesn't require admin rights.
- **Alternative:** if `winget` is available (`winget --version` in a terminal),
  `winget install --id Python.Python.3.12 --scope user` installs it without
  admin rights either.

Close and reopen your terminal after installing so `PATH` updates take effect.

> **You do not need a C/C++ compiler.** All of SoloLab's dependencies (numpy,
> scipy, pandas, astropy, PyQt5, cdflib, etc.) publish prebuilt Windows wheels
> for Python 3.10-3.12. If `pip install` ever tries to compile something from
> source and fails with a `Microsoft Visual C++ 14.0 or greater is required`
> error, it almost always means your Python version is *too new* for one of the
> packages to have a wheel yet - switch to Python 3.12 rather than installing a
> compiler.

### macOS

Recent macOS ships Python, but it's often outdated or missing `pip`/`venv`.
Easiest fix, using [Homebrew](https://brew.sh/):

```bash
brew install python@3.12
```

Without Homebrew, use the official installer from
[python.org/downloads/macos](https://www.python.org/downloads/macos/).

On Apple Silicon (M1/M2/M3), all of SoloLab's dependencies have `arm64` wheels
today, so a compiler is normally not needed. If one *is* ever required (rare),
install Xcode's command-line tools: `xcode-select --install`.

### Linux

Use your distribution's package manager. On Debian/Ubuntu, also grab the `venv`
module - some distros ship it as a separate package:

```bash
sudo apt update
sudo apt install python3 python3-venv python3-pip
```

Fedora/RHEL: `sudo dnf install python3 python3-pip`. Arch: `sudo pacman -S python python-pip`.

---

## 2. Create a virtual environment

A virtual environment ("venv") keeps SoloLab's dependencies isolated from the
rest of your system. Run these from the repo root (the folder containing this
file). Pick **one** environment name per app if you're installing both
(`.venv` for the desktop app, `.venv-dash` for the web app is a good
convention - that's what's used below).

### macOS / Linux

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
```

`source .venv/bin/activate` prefixes your prompt with `(.venv)` once it's
active. To leave the environment later: `deactivate`.

### Windows

```powershell
python -m venv .venv
```

Normally you'd activate it with `.venv\Scripts\Activate.ps1` (PowerShell) or
`.venv\Scripts\activate.bat` (cmd.exe). **If your system blocks running that
script** - you'll see something like:

```
File ... cannot be loaded because running scripts is disabled on this system.
```

you have two options, in order of preference:

**Option A - don't activate at all (works even with the strictest policies).**
Every command below just calls the venv's Python directly by path instead of
relying on an activated shell - functionally identical, and it never touches
PowerShell's script execution policy at all:

```powershell
.venv\Scripts\python.exe -m pip install --upgrade pip
.venv\Scripts\python.exe -m pip install -r requirements.txt
.venv\Scripts\python.exe test_installation.py
.venv\Scripts\python.exe test_sololabapp.py
```

(Substitute `.venv-dash\Scripts\python.exe` for the web app.) This is the
approach used throughout this guide's Windows examples.

**Option B - allow activation for your user only, if you're allowed to change
this setting:**

```powershell
Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned
.venv\Scripts\Activate.ps1
```

`-Scope CurrentUser` doesn't require admin rights and doesn't affect other
users on the machine. If group policy blocks this outright (common on managed
corporate laptops), use Option A - it's unaffected by execution policy.

---

## 3. Install dependencies

With the environment created (Option A style shown; drop the `.venv\Scripts\`
prefix on macOS/Linux or if you activated the environment):

**Desktop app:**

```powershell
.venv\Scripts\python.exe -m pip install -r requirements.txt
```

**Web app** (separate environment):

```powershell
python -m venv .venv-dash
.venv-dash\Scripts\python.exe -m pip install -r requirements-dash.txt
```

This step downloads and installs numpy, scipy, pandas, astropy, cdflib, and
the rest - it can take a few minutes on the first run.

`requirements.txt` also includes `stixdcpy`, used only by the desktop app's
"Download from STIX Data Center..." button (in the Import STIX dialog) to
fetch a spectrogram file for a given date/time range directly, instead of
downloading one manually first. Everything else works normally without it.
The equivalent RPW-HFR/RPW-TNR "Download from CDAWeb..." buttons need no
extra package - they use the standard library only.

---

## 4. Verify the install

Run the dependency-check script (see [`test_installation.py`](test_installation.py)):

```powershell
.venv\Scripts\python.exe test_installation.py
```

It imports every dependency, reports pass/fail per package (grouped as
core / desktop-only / web-only, so it works whichever environment you run it
in), and exits non-zero if anything required is missing - no GUI, no network,
no sample data needed. Fix whatever it flags before moving on.

---

## 5. Run it

**Desktop app:**

```powershell
.venv\Scripts\python.exe test_sololabapp.py
```

(or `python -c "import sololab; sololab.run_app()"`).

**Web app:**

```powershell
.venv-dash\Scripts\python.exe -m sololab.dash_app.app
```

then open the printed `http://127.0.0.1:8050/` in a browser.

**Usage examples** (reading files, extracting data, plotting): see
[`sololab_examples.ipynb`](sololab_examples.ipynb) - open it with
`.venv\Scripts\python.exe -m jupyter lab` (desktop env has `jupyterlab`; for
the web env run `.venv-dash\Scripts\python.exe -m pip install jupyterlab`
first, it's not included by default since the web app doesn't need it).

---

## Troubleshooting

- **`pip install` fails trying to build something from source on Windows** -
  you're very likely on a Python version newer than what that package has
  published wheels for yet. Reinstall using Python 3.12 (see step 1) rather
  than installing a compiler.
- **`ModuleNotFoundError: No module named 'PyQt5'` when running the web app** -
  expected and harmless; the web app doesn't need PyQt5.
  `sololab/__init__.py` catches this automatically. If you see this error
  from the *desktop* app instead, you installed `requirements-dash.txt`
  there by mistake - use `requirements.txt`.
- **"Downloading STIX data requires the optional 'stixdcpy' package"** - only
  shown if you click "Download from STIX Data Center..." without `stixdcpy`
  installed (e.g. `pip install -r requirements.txt` ran before it was added,
  or you installed packages individually). `pip install stixdcpy` fixes it;
  everything else in the app works fine without it.
- **Windows: "running scripts is disabled on this system"** - see
  [step 2](#windows), Option A (call `python.exe` directly, skip activation
  entirely).
- **Still stuck?** Run `test_installation.py` and read its per-package
  report - it tells you exactly which import failed and (for the common
  cases) why.
