# AAR_THAR_AABR-ELA
This repository contains the code used to reproduce the simulations, data analysis, and figures of
- Yang, W., Mackintosh, A.N., Cooper, E-L., Li, Y., Jone, R.S., Chu, W., and Tielidze, L.G., (2025): A global estimate of Equilibrium-Line Altitude ratios for improved glacier-climate reconstructions.

We provide the estimates of AAR, THAR, and AABR ratios for nearlly all glaciers on Earth under steady-state conditions (AAR<sub>0</sub>, THAR<sub>0</sub>, and AABR<sub>0</sub>) using a hybrid of [Python Glacier Evolution Model (PyGEM v0.2.5)](https://github.com/PyGEM-Community/PyGEM/releases/tag/v0.2.0), developed by David Rounce and collaborators, and [Open Global Glacier Model (OGGM v1.6.0)](https://github.com/OGGM/oggm/releases/tag/v1.6.0), developed by the OGGM community. Glacier AAR<sub>0</sub> was calculated by completing linear regression of simulated annual mass balances and AARs from 1995 to 2014. THAR<sub>0</sub> and AABR<sub>0</sub> were estimated for each glacier based on AAR0 and the glacier geometry from the Randolph Glacier Inventory 6.2 

The files include:
- [`README.md`](README.md) — Description of the repository
- ['data'](data) - The documentation of the data. Download the large data files (e.g., ERA5_MCMC_ba1_2014_2023_corrected.nc) from [Google Drive](https://drive.google.com/drive/folders/19rjAJm0g4HR1njfsnJD7jJr7Khg-5TN9?usp=share_link).
- ['code'](code) - The documentation of the code for running simulations, analyzing the data, and creating figures and tables.

## Overview of the code
- Run the PyGEM script `run_simulation.py` and `pygem_input.py`. <br>
  This script replaces the original `run_simulation` file in PyGEM and automatically performs glacier climate disequilibrium calculations using both the parameterization approach and the equilibrium experiment.

- `process_disequilibrium.py`. <br>
  Compiles the output of the PyGEM runs of several gdirs into one file.
  
- `process_disequilibrium_errors.py`. <br>
  Uses the nearest neighbour interpolation to estimate results for the failed glaciers.
  
- `process_disequilibrium_by_region.py`, `process_disequilibrium_by_area.py`, `process_disequilibrium_lat_lon_mean.py`, and `process_disequilibrium_griddata.py`. <br>
  Analyze the results based on RGI regions, glacier area, and 2°×2° grid resolution.

- `wgms_disequilibrium.py`. <br>
  Calculate glacier climate disequilbirium based on the WGMS observations.

- `Figure_*.py` and `Table_*.py`. <br>
  Create the figures and tables

## Contact

If you have any questions, please contact:

**Dr. Weilin Yang**  
School of Earth, Atmosphere and Environment, Monash University  <br>
✉️ weilinyang.yang@monash.edu
