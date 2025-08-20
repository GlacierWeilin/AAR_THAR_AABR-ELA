#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 11 16:30:42 2025

@author: Weilin Yang (weilinyang.yang@monash.edu)
"""
import numpy as np
import pandas as pd
import xarray as xr

import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

import seaborn as sns

file_path = '/Users/wyan0065/Desktop/AAR-disequilibrium/AABR/data_and_code/data/'
data = xr.open_dataset(file_path + 'results_all_mad_debriscalving.nc')

y_all = pd.DataFrame({
    'AAR': data['compile_AAR'].values[:,0],
    'THAR': data['compile_THAR'].values[:,0],
    'AABR': data['compile_AABR'].values[:,0]
})

df = pd.DataFrame({
        'temp': data['glacier_temp'].values,
        'prcp': data['prcp'].values,
        'area': data['Area'].values*1000*1000,
        'slope': data['Slope'].values
})

variables = [
    {'x': 'temp',
     'bins': np.linspace(-22,5,10),
     'xlabel': 'Temperature (°C)','scale': 'linear'
     },
    {'x': 'prcp',
     'bins': np.linspace(0,5,10),
     'xlabel': 'Precipitation (m)','scale': 'linear'
     },
    {'x': 'area',
     'bins': np.logspace(4, 7, 10),
     'xlabel': 'Area (m$^2$)','scale': 'log'
     },
    {'x': 'slope',
     'bins': np.linspace(7,45,10),
     'xlabel': 'Slope (°)','scale': 'linear'
    }
]

#%%

plt.rcParams.update({'lines.linewidth':1})
plt.rcParams.update({'font.size': 7})
plt.rcParams.update({'font.sans-serif': 'Arial'})

plt.rcParams.update({'axes.linewidth':0.7})
plt.rcParams.update({'axes.titlepad':3})
plt.rcParams.update({'axes.titlesize':7})
plt.rcParams.update({'axes.labelpad':1})
plt.rcParams.update({'xtick.major.pad':2})
plt.rcParams.update({'ytick.major.pad':2})
plt.rcParams.update({'xtick.major.width':0.7})
plt.rcParams.update({'ytick.major.width':0.7})
plt.rcParams.update({'xtick.major.size':2})
plt.rcParams.update({'ytick.major.size':2})
plt.rcParams.update({'xtick.minor.size': 1})
plt.rcParams.update({'ytick.minor.size': 1})


fig = plt.figure(figsize=(7.09, 4.5))
gs = GridSpec(3, 4, figure=fig, hspace=0.07, wspace=-0.17,
              height_ratios=[1, 1, 1], width_ratios=[1, 1, 1, 1])
plt.subplots_adjust(left=0.05, right=0.99, top=0.98, bottom=0.07)

for i in range(0,3):
    for j, var in enumerate(variables):
        ax = fig.add_subplot(gs[i, j])
        
        if i == 0:
            color = 'Blues'
        elif i== 1:
            color = 'Oranges'
        else:
            color = 'Reds'
        
        y_name = y_all.columns[i]
        y = y_all[y_name].astype(float).values
        
        name = var['x']
        x = df[name].values
        
        bin_min = var['bins'][0]
        bin_max = var['bins'][-1]
        
        mask = (np.isfinite(x) & np.isfinite(y) & (x >= bin_min) & (x <= bin_max))
        
        x_valid = x[mask]
        y_valid = y[mask]
        
        x_bins = pd.cut(x_valid, bins=var['bins'], labels=False)
        bin_edges = var['bins']
        bin_centers = [(bin_edges[j], bin_edges[j + 1]) for j in range(len(bin_edges) - 1)]
    
        temp_df = pd.DataFrame({'bin': x_bins,'y': y_valid})
        mean_vals = temp_df.groupby('bin')['y'].mean();

        step_x, step_y = [], []
        for k, mean in enumerate(mean_vals):
            x0, x1 = bin_centers[k]
            #step_x.extend([x0, x1])
            #step_y.extend([mean, mean])
            
            step_x.append(np.mean([x0,x1]))
            step_y.append(mean)

        #ax.scatter(x_valid, y_valid, alpha=0.5, color='#489FE3', s=1)
        kde = sns.kdeplot(x=x_valid, y=y_valid, ax=ax, cmap=color, fill=True, thresh=0.05, alpha=0.6, levels=10)
        mappable = kde.collections[0]
        vmin = mappable.get_array().min()
        vmax = mappable.get_array().max()
        vmed = mappable.get_array()[4]
        cbar = plt.colorbar(mappable, ax=ax, orientation='vertical')
        cbar.set_ticks([vmin, vmed, vmax])
        cbar.set_ticklabels(['low', 'medium', 'high'])
        cbar.set_label('Density level')
        for tick in cbar.ax.get_yticklabels():
            tick.set_rotation(90)
            tick.set_va('center')
            tick.set_ha('left')
            if j!= 3:
                tick.set_alpha(0)
        #ax.plot(step_x, step_y, color='black', linewidth=0.7)
        ax.scatter(step_x, step_y, alpha=1, color='black', s=5, marker='^', label='Group mean')

        if var['scale'] == 'log':
            ax.set_xscale('log')
        if 'xlim' in var:
            ax.set_xlim(np.min(var['bins']), np.max(var['bins']))
        
        if i == 0:
            ax.set_ylim(0.3, 0.8)
        elif i == 1:
            ax.set_ylim(0.2, 0.9)
        else:
            ax.set_ylim(0, 4.5)
        if i == 0 and j == 0:
            ax.set_ylabel('AAR$_0$')
        elif i == 1 and j == 0:
            ax.set_ylabel('THAR$_0$')
        elif i == 2 and j == 0:
            ax.set_ylabel('AABR$_0$')
        else:
            ax.set_yticklabels([])
        
        if i == 2:
            ax.set_xlabel(var['xlabel'])
        else:
            ax.set_xticklabels([])
        
        if i == 0 and j == 0:
            ax.legend(loc='lower left', frameon=False, fontsize=6, handletextpad=0.2)

        ax.text(0.03, 0.97, chr(ord('a') + 4*i +j), transform=ax.transAxes,
                fontsize=7, fontweight='bold', va='top', ha='left')

out_pdf = '/Users/wyan0065/Desktop/AAR-disequilibrium/AABR/manuscript/' + 'figure_4.png'
plt.savefig(out_pdf, dpi=600)
plt.show()