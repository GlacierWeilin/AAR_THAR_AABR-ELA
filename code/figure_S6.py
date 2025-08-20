#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 11 14:05:15 2025

@author: Weilin Yang (weilinyang.yang@monash.edu)
"""
import numpy as np
import xarray as xr
import pandas as pd
from scipy.stats import pearsonr

import seaborn as sns
import matplotlib as mpl
import matplotlib.pyplot as plt

file_path = '/Users/wyan0065/Desktop/AAR-disequilibrium/AABR/data_and_code/data/'
data = xr.open_dataset(file_path + 'results_all_mad_debriscalving.nc');

df = pd.DataFrame({
    'AAR$_0$': data['compile_AAR'].values[:,0],
    'THAR$_0$': data['compile_THAR'].values[:,0],
    'AABR$_0$': data['compile_AABR'].values[:,0],
    'Temp': data['glacier_temp'].values,
    'Prcp': data['prcp'].values,
    'Area': data['Area'].values,
    'Slope': data['Slope'].values,
    'Sin (Aspect)': np.sin(np.deg2rad(data['Aspect'].values)),
    'Cos (Aspect)': np.cos(np.deg2rad(data['Aspect'].values)),
})

corr = df.corr()
pvals = pd.DataFrame(np.ones(corr.shape), columns=df.columns, index=df.columns)

for i in df.columns:
    for j in df.columns:
        if i != j:
            x = df[i]
            y = df[j]
            mask = x.notna() & y.notna()
            r, p = pearsonr(x[mask], y[mask])
            pvals.loc[i, j] = p
        else:
            pvals.loc[i, j] = np.nan

annot = corr.round(2).astype(str)

for i in annot.columns:
    for j in annot.index:
        p = pvals.loc[j, i]
        if pd.isna(p):
            continue
        elif p < 0.01:
            annot.loc[j, i] += '*'

plt.rcParams.update({'lines.linewidth':0.5})
plt.rcParams.update({'font.size': 6})
plt.rcParams.update({'font.sans-serif': 'Arial'})

plt.rcParams.update({'axes.linewidth':0.5})
plt.rcParams.update({'axes.titlepad':3})
plt.rcParams.update({'axes.titlesize':6})
plt.rcParams.update({'axes.labelpad':2})
plt.rcParams.update({'xtick.major.pad':2})
plt.rcParams.update({'ytick.major.pad':2})
plt.rcParams.update({'xtick.major.width':0.5})
plt.rcParams.update({'ytick.major.width':0.5})
plt.rcParams.update({'xtick.major.size':1.5})
plt.rcParams.update({'ytick.major.size':1.5})

fig= plt.figure(figsize=(4.72, 2.8), dpi=600)
ax = plt.axes([0.12,0.06,0.97,1.02])

#sns.heatmap(corr, annot=annot, cmap='coolwarm', vmin=-1, vmax=1, fmt='', linewidths=0.5, ax=ax, cbar=False)
mask = np.triu(np.ones_like(corr, dtype=bool))

col_bounds = np.linspace(-0.4,0,7)
col_bounds = np.append(col_bounds, np.linspace(0,0.8,7))
cb = []
cb_val = np.linspace(1, 0, len(col_bounds))
for j in range(len(cb_val)):
    cb.append(mpl.cm.coolwarm_r(cb_val[j])) #'RdYlBu_r'
cmap_cus = mpl.colors.LinearSegmentedColormap.from_list('my_cb', list(zip((col_bounds - min(col_bounds)) / (max(col_bounds - min(col_bounds))), 
                                                                          cb)), N=1000)

norm = mpl.colors.Normalize(vmin=min(col_bounds), vmax=max(col_bounds))

sns.heatmap(
    corr,
    annot=annot,
    cmap=cmap_cus,
    norm=norm,
    fmt='',
    mask=mask,
    linewidths=0.5,
    ax=ax,
    square=False,
    cbar_kws={
        'shrink': 0.89,
        'pad': 0.001,
        'aspect':25,
        'anchor': (0.0, 0.0),
        'orientation': 'horizontal' 
    }
)

ax.set_yticks(ax.get_yticks()[1:])
ax.set_xticks(ax.get_xticks()[0:-1])
ax.set_xticklabels(ax.get_xticklabels(), rotation=0)
ax.set_yticklabels(ax.get_yticklabels(), rotation=0)

out_pdf = '/Users/wyan0065/Desktop/AAR-disequilibrium/AABR/manuscript/' + 'figure_S6.png'
plt.savefig(out_pdf, dpi=600)
plt.show()