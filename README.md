# AAR_THAR_AABR-ELA
This repository contains the code used to reproduce the simulations, data analysis, and figures of
- Yang, W., Mackintosh, A.N., Cooper, E-L., Li, Y., Jone, R.S., Chu, W., and Tielidze, L.G., (2025): A global estimate of Equilibrium-Line Altitude ratios for improved glacier-climate reconstructions.

We provide the estimates of AAR<sub>0</sub>, THAR<sub>0</sub>, and AABR<sub>0</sub> ratios for nearlly all glaciers on Earth using a hybrid of [Python Glacier Evolution Model (PyGEM v0.2.5)](https://github.com/PyGEM-Community/PyGEM/releases/tag/v0.2.0), developed by David Rounce and collaborators, and [Open Global Glacier Model (OGGM v1.6.0)](https://github.com/OGGM/oggm/releases/tag/v1.6.0), developed by the OGGM community. Glacier AAR<sub>0</sub> was calculated by completing linear regression of simulated annual mass balances and AARs from 1995 to 2014. THAR<sub>0</sub> and AABR<sub>0</sub> were estimated for each glacier based on AAR<sub>0</sub> and the glacier geometry from the Randolph Glacier Inventory 6.2.

The files include:
- [`README.md`](README.md) — Description of the repository
- ['data'](data) - The documentation of the data. Download the large data files from [Google Drive](https://drive.google.com/drive/folders/1S8amtzRIEJkixYB_J5qbf-9YW0iHRWg2?usp=sharing).
- ['code'](code) - The documentation of the code for running simulations, analyzing the data, and creating figures and tables.

## Overview of the code
- Run the PyGEM script `run_simulation.py` and `pygem_input.py` to calculate . <br>
  This script replaces the original `run_simulation` file in PyGEM and automatically performs glacier AAR<sub>0</sub> calculations using linear regression and steady-state assumption.

- `compile_pygem_results.py` and `process_errors.py` <br>
  Compile the output of the PyGEM runs of several gdirs into one file. Use the nearest neighbour interpolation to estimate results for the failed glaciers.
  
- `run_oggm.py`. <br>
  Run oggm to calculate glacier ELA<sub>0</sub>, THAR<sub>0</sub> and AABR<sub>0</sub> based on AAR<sub>0</sub> and the glacier geometry.

- `compile_oggm_results.ipynb`, `compile_results.ipynb`, `compile_median_mad.py` and `compile_glacier_statistics.py`. <br>
  Compile the output of the OGGM runs of several gdirs into one file.
  
- `process_region.py`, and `process_griddata.py`. <br>
  Analyze the results based on RGI regions, glacier area, and 0.5°×0.5° grid resolution.

- `wgms_ELA.py` and `run_wgms_AAR.ipynb.` <br>
  Calculate glacier AAR<sub>0</sub>, THAR<sub>0</sub>, and AABR<sub>0</sub> based on the WGMS observations.

- `results_equal_count_bins.py`, `classification.py`, and `GUI_for_paleoglacier.py`. <br>
  Analyze the results based on RGI regions, glacier area, and 0.5°×0.5° grid resolution.

- `Figure_*.py`. <br>
  Create the figures.

## Contact

If you have any questions, please contact:

**Dr. Weilin Yang**  
School of Earth, Atmosphere and Environment, Monash University  <br>
✉️ weilinyang.yang@monash.edu
