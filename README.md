# AAR_THAR_AABR-ELA
This repository contains the code used to reproduce the simulations, data analysis, and figures of
- Yang, W., Mackintosh, A.N., Cooper, EL. et al. Global estimates of glacier equilibrium-line altitude ratios for enhanced paleoclimate reconstructions. Commun Earth Environ (2026). https://doi.org/10.1038/s43247-026-03391-5.

We estimate of AAR<sub>0</sub>, THAR<sub>0</sub>, and AABR<sub>0</sub> ratios for nearlly all modern glaciers on Earth using a hybrid of [Python Glacier Evolution Model (PyGEM v0.2.5)](https://github.com/PyGEM-Community/PyGEM/releases/tag/v0.2.0), developed by David Rounce and collaborators, and [Open Global Glacier Model (OGGM v1.6.0)](https://github.com/OGGM/oggm/releases/tag/v1.6.0), developed by the OGGM community. Glacier AAR<sub>0</sub> was calculated by completing linear regression of simulated annual mass balances and AARs from 1995 to 2014 and a steady-state assumption method. ELA<sub>0</sub>, THAR<sub>0</sub> and AABR<sub>0</sub> were estimated for each glacier based on AAR<sub>0</sub> and the glacier geometry from the Randolph Glacier Inventory 6.2.

We then analyze the influence of climatic, topographic, and glacier-intrinsic factors (e.g., glacier types and glacier area) on AAR<sub>0</sub>, THAR<sub>0</sub>, and AABR<sub>0</sub>, thereby identifying systematic variations in ELA ratios across groups of glaciers with similar characteristics. Based on this analysis, we develope a decision-tree-based classification tool, “Glacier ELA Ratio Calculator” (GERC), to estimate AAR<sub>0</sub>, THAR<sub>0</sub>, and AABR<sub>0</sub> values for groups of glaciers with similar characteristics based on key glacier and climate variables. The web-based version of the GERC is available on [Streamlit](https://gercglacier.streamlit.app/).

Although we also provide the AAR<sub>0</sub>, ELA<sub>0</sub>, THAR<sub>0</sub>, and AABR<sub>0</sub> of global modern glaciers in the NetCDF file `results_all_mad_debriscalving.nc` and in the directory `/results_all_mad_debriscalving_csv`, we still recommend the users to use "GERC" for the more robust estimates at the group level, rather than for individual glaciers.

The files include:
- [`README.md`](README.md) — Description of the repository
- ['data'](data) - The documentation of the data. Download the large data files from [Zenodo](https://doi.org/10.5281/zenodo.18737031).
- ['code'](code) - The documentation of the code for running simulations, analyzing the data, and creating figures and tables.

## Overview of the code
- Run the PyGEM script `run_simulation.py` and `pygem_input.py`. <br>
  This script replaces the original `run_simulation` file in PyGEM and automatically performs glacier AAR<sub>0</sub> calculations using linear regression and steady-state assumption.

- `compile_pygem_results.py` and `process_errors.py` <br>
  Compile the output of the PyGEM runs of several gdirs into one file. Use the nearest neighbour interpolation to estimate results for the failed glaciers.
  
- `run_oggm.py`. <br>
  Run oggm to calculate glacier ELA<sub>0</sub>, THAR<sub>0</sub> and AABR<sub>0</sub> based on AAR<sub>0</sub> and the glacier geometry.

- `compile_oggm_results.ipynb`, `compile_results.ipynb`, `compile_median_mad.py`, `compile_Loibl_results.ipynb` and `compile_glacier_statistics.py`. <br>
  Compile the output of the OGGM runs of several gdirs into one file.
  
- `process_region.py`, and `process_griddata.py`. <br>
  Analyze the results based on RGI regions, glacier area, and 0.5°×0.5° grid resolution.

- `wgms_ELA.py` and `run_wgms_AAR.ipynb.` <br>
  Calculate glacier AAR<sub>0</sub>, THAR<sub>0</sub>, and AABR<sub>0</sub> based on the WGMS observations.

- `Loibl_snowline_ELA.py`, `Loibl_AAR.ipynb.` and `run_Loibl_AAR.py.` <br>
  Calculate glacier AAR<sub>0</sub>, THAR<sub>0</sub>, and AABR<sub>0</sub> based on the transient snowline altitude observations.

- `results_equal_count_bins.py`, `classification.py`, and `GUI_for_paleoglacier.py`. <br>
  Create a user-friendly tool that estimates AAR<sub>0</sub>, THAR<sub>0</sub>, and AABR<sub>0</sub> values for paleoglaciers based on key glacier-specific and climate-topographic variables.

- `Figure_*.py`. <br>
  Create the figures.

## Overview of the data
- Results calculated using the Linear Regression method are named with the prefix intercept_*, for example, intercept_AAR.
  Results calculated using the Steady-State Assumption method are named with the prefix steady_*, for example, steady_AAR.
  The median values derived from both methods are named with the prefix compile_*.

- Outputs for PyGEM calibration: `/sims`

- AAR<sub>0</sub> of global modern glaciers output from a hybrid model of PyGEM and OGGM: `results_AAR_*.nc`.

- AAR<sub>0</sub>, ELA<sub>0</sub>, THAR<sub>0</sub>, and AABR<sub>0</sub> of global modern glaciers: `results_all_mad_*.nc`.
  For users who are not familiar with processing NetCDF files or code, the main corresponding values for modern glaciers can be   directly accessed in the CSV files located in `/results_all_mad_debriscalving_csv`.
  
- Global and regional AAR<sub>0</sub>, THAR<sub>0</sub>, and AABR<sub>0</sub> based on the simulation results: `results_region.nc`.

- Spatial distribution of the grid (0.5° × 0.5°) AAR<sub>0</sub>, (b) THAR<sub>0</sub>, and (c) AABR<sub>0</sub>: `results_0.5.nc`.

- Estimations based on WGMS observations: `/DOI-WGMS-FoG-2024-01`, `WGMS*.csv`, `results_comparison.nc`.

- Estimations based on snowline observations: `Loibl*.csv`, `Loibl*.nc`, `MB_1995_2014_Dussaillant.csv`, `results_comparison.nc`.

- Compiled glacier statistics from RGI 6.2 and ERA5: `/ori`, `/summary`, `glacier_statistics.nc`.

- Classificationtree (also the inputs of GERC if you don't use the online app): `/classificationtree`.

## Contact

If you have any questions, please contact:

**Dr. Weilin Yang**  
School of Earth, Atmosphere and Environment, Monash University  <br>
✉️ weilinyang.yang@monash.edu and ywlcwc@gmail.com
