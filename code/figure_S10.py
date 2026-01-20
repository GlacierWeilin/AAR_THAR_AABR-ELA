#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 11 11:43:26 2024

@author: Weilin Yang (weilinyang.yang@monash.edu)
"""

import pandas as pd
import numpy as np
import xarray as xr
import os
import glob
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec


errors = pd.Series()

# %% no gdirs
# RGI summary
file_path = '/Users/wyan0065/Desktop/AAR-disequilibrium/AABR/data_and_code/data/'
csv_files = glob.glob(file_path + 'ori/*.csv')
dfs = [pd.read_csv(file, usecols=[0], dtype='str') for file in csv_files]
RGI_all = pd.concat(dfs, ignore_index=True)

# oggm summary
url = file_path + 'summary/'

oggm_all = []
for rgi_reg in range(1, 20):
    fpath = os.path.join(url, f'glacier_statistics_{rgi_reg:02d}.csv')
    oggm_all.append(pd.read_csv(fpath, index_col=0, low_memory=False))

oggm_all = pd.concat(oggm_all, sort=False).sort_index()

find = RGI_all['RGIId'].isin(oggm_all.index)
loc = np.where(find==0)

noglacierdirectories = RGI_all.iloc[loc[0]]
noglacierdirectories.to_csv(file_path+'sims/noglacierdirectories.csv', index=True)

errors['no_glacier_directories'] = np.shape(noglacierdirectories)[0]

#%% oggm tasks failed
find = np.where(oggm_all['error_task']=='simple_glacier_masks')
errors['simple_glacier_masks'] = np.shape(find[0])[0]

find = np.where(oggm_all['error_task']=='define_glacier_region')
errors['define_glacier_region'] = np.shape(find[0])[0]

find = np.where(oggm_all['error_task']=='elevation_band_flowline')
errors['elevation_band_flowline'] = np.shape(find[0])[0]

#%%
data = xr.open_dataset(file_path + 'results_all_mad_debriscalving.nc');
find = np.where(np.isnan(data['compile_AABR'].values[:,0]))
errors['ELA$_0$ = At'] = np.shape(find[0])[0] - errors['simple_glacier_masks'] - errors['define_glacier_region'] - errors['elevation_band_flowline']

data.close()

#%% calibration failed

def list_subdirectories(directory):
    return [d for d in Path(directory).iterdir() if d.is_dir()]

path = file_path +'sims/'
directories = list_subdirectories(path)

def count_files_in_directory(directory):
    return len(list(Path(directory).rglob('*')))

k=0
for i in range(0,len(directories)):
    file_count = count_files_in_directory(directories[i].as_posix())
    k = k+file_count;

errors['Calibration'] = np.shape(oggm_all)[0]- k - errors['elevation_band_flowline']-errors['define_glacier_region']-errors['simple_glacier_masks'];

#%%
data = xr.open_dataset(file_path + 'results_AAR_debriscalving_withnan.nc');
find = np.where(np.isnan(data['intercept_AAR'].values[:,0]))
errors['Linear regression method'] = np.shape(find[0])[0] - errors['Calibration']- errors['elevation_band_flowline']-errors['define_glacier_region']-errors['simple_glacier_masks']

find = np.where(np.isnan(data['steady_AAR'].values[:,0]))
errors['Steady-state assumption'] = np.shape(find[0])[0] - errors['Calibration']- errors['elevation_band_flowline']-errors['define_glacier_region']-errors['simple_glacier_masks']

data.close()

#%%
plt.rcParams.update({'lines.linewidth':1})
plt.rcParams.update({'font.size': 7})
plt.rcParams.update({'font.sans-serif': 'Arial'})

plt.rcParams.update({'axes.linewidth':0.7})
plt.rcParams.update({'axes.titlepad':3})
plt.rcParams.update({'axes.titlesize':8})
plt.rcParams.update({'axes.labelpad':2})
plt.rcParams.update({'xtick.major.pad':2})
plt.rcParams.update({'ytick.major.pad':2})
plt.rcParams.update({'xtick.major.width':0.7})
plt.rcParams.update({'ytick.major.width':0.7})
plt.rcParams.update({'xtick.major.size':1.5})
plt.rcParams.update({'ytick.major.size':1.5})

fig = plt.figure(figsize=(3.6, 2.0), dpi=600)
gs = GridSpec(2, 1, figure=fig, hspace=0.1, height_ratios=[5, 3])
plt.subplots_adjust(left=0.37, right=0.96, top=0.98, bottom=0.13)

ax1 = fig.add_subplot(gs[0, 0], xlim=(0,4000))
bar1 = ax1.barh(errors[:5].index, errors[:5].values, color='orange', edgecolor=None)
ax1.set_ylabel('OGGM errors')
ax1.set_xticklabels([])
ax1.set_xlabel('')
ax1.bar_label(bar1, fmt='%d', padding=3, fontsize=7, color='orange')

ax2 = fig.add_subplot(gs[1, 0], xlim=(0,4000))
bar2 = ax2.barh(errors[5:].index, errors[5:].values, color='#489FE3', edgecolor=None)
ax2.set_ylabel('PyGEM errors')
ax2.set_xlabel('Count')
ax2.bar_label(bar2, fmt='%d', padding=3, fontsize=7, color='#489FE3')

ax1.invert_yaxis()
ax2.invert_yaxis()

out_pdf = '/Users/wyan0065/Desktop/AAR-disequilibrium/AABR/manuscript/' + 'figure_S10.png'
plt.savefig(out_pdf, dpi=600)

plt.show()