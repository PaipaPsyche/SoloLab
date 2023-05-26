# SoloLab V 0.1
### Python Tool for multi-instrument studies with Solar Orbiter Data
This tutorial will help you in the installation the requirements for using soloLab, a tool to process and visualize multi-instrument data from Solar Orbiter: A spacecraft launched in February 2020 with 10 instruments onboard dedicated to study the sun from up close (at distances down to 0.27 AU or 60 solar Radii). Among these instruments, the three instruments we are concerned about in the early development of this tool are:
- **STIX:** the **S**pectrometer **T**elescope for **I**maging **X**-rays.
- **RPW:**  the **R**adio and **P**lasma **W**aves instrument.
- **EPD:**  the **E**nergetic **P**articles **D**etector.

Since the spacecraft was launched recently, the scientific teams of each instrument are still figuring out the behavior of the measurements they take therefore, specialized libraries for processing data from the Solar Orbiter instruments are still under development (or revision).

Traditionally, data analysis in heliophysics was done using a custom-made wrapper library called SolarSoftWare(SSW) written in the programming language IDL. Unfortunately, IDL is not free to use and lacks support to many of the modern functionalities that one would need currently - for example, machine-learning libraries. There is growing community support in porting the existing code written in IDL for various instruments to Python. SunPy is currently one of the most active efforts in this direction.

The utility of a pseudo-automatic visualization tool in python comes from the need of correlating the properties of different emissions of solar flares, which can provide information on the physical processes of particle acceleration in flares and their transport across the heliosphere. Comparing the data time series from multiple instruments can help to determine the association between radio, x-ray and particle events, thus allowing to have a broader context to analyze solar transient events. Combined observations can also draw interesting results in other areas of heliophysics; not only measuring the properties of transients, but may also help to get a deeper knowledge of the quiet sun and the beahvior of the heliosphere.

**SoloLab V0.1** uses python libraries and custom made scripts to treat data obtained by Solar Orbiter, helping in the following tasks:

- **Data extraction:** Functions to extract STIX FITS files (L1 pixel data and spectrograms, L1 BKG files) and RPW CDF files (HFR and TNR L2 files).
- **Data processing:** STIX background subtraction (from BKG and/or quiet time intervals) and manual energy shifts. RPW background subtraction (from quiet time interval). Filtering of polluted frequencies for RPW.
- **Data visualization:** Plots of simultaneous measurements of STIX and RPW (spectrograms and time profiles per energy/frequency channel) with the possibility of introducing simultaneous EPD time profiles (EPT/electrons) and X-ray spectroscopy results (injected electron powerlaws, electron abundances at different energy thresholds).
- **Estimations and fits:** Fit of RPW time profiles (per frequency) to estimate the exciter velocity using Frequency Drift Rate Analysis (FDRA) and regression methods. Estimation of electron abundances as a function of threshold energy once given the powerlaws obtained from X-ray spectroscopy.

### Considerations and Contact

Created by **David Paipa** 
*LESIA; Observatoire de Meudon, France*
[(contact)](mailto:david.paipa@obspm.fr)


This code is **NOT** an official Solar Orbiter ground software, moreover is still a work in progress initially developed as a tool for my PhD thesis. If you have any question, suggestion or notice any bug/mistake do not hesitate to contact me.
