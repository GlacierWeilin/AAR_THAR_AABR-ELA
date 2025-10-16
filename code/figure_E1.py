#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 28 16:14:41 2025

@author: Weilin Yang (weilinyang.yang@monash.edu)
"""

import numpy as np
import pandas as pd
import xarray as xr
import scipy.stats as st
from sklearn.metrics import mean_squared_error

import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import matplotlib.patches as mpatches
import matplotlib.lines as mlines

file_path = '/Users/wyan0065/Desktop/AAR-disequilibrium/AABR/data_and_code/data/'
ds_all = xr.open_dataset(file_path + 'results_comparison.nc')

#%%
ELA_AAR = pd.DataFrame({
        'AAR_0.58': ds_all['ELA_AAR_0.58'].values,
        'AAR_0.50': ds_all['ELA_AAR_0.50'].values,
})

ELA_THAR = pd.DataFrame({
        'THAR_0.50': ds_all['ELA_THAR_0.50'].values,
})

ELA_AABR = pd.DataFrame({
        'AABR_1.75': ds_all['ELA_AABR_1.75'].values,
        'AABR_1.56': ds_all['ELA_AABR_1.56'].values,
})

titles = ['AAR$_0$', 'THAR$_0$', 'AABR$_0$']
colors = ['#489FE3', '#DC6D57']

alpha = 0.01

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

fig = plt.figure(figsize=(5.91, 3.5), dpi=600)
gs = GridSpec(2, 3, figure=fig, hspace=0.25, wspace=0.05, height_ratios=[1,1], width_ratios=[1,1,1])
plt.subplots_adjust(left=0.065, right=0.995, top=0.95, bottom=0.08)

for i in range(6):
    
    y = ds_all['ELA_median_compile'].values
    
    row = i // 3
    col = i % 3
    ax = fig.add_subplot(gs[row, col])
    color = colors[0]
    
    if col == 0:
        name = ELA_AAR.columns[0]
        x = ELA_AAR[name].values
    elif col == 1:
        name = ELA_THAR.columns[0]
        x = ELA_THAR[name].values
    else:
        name = ELA_AABR.columns[0]
        x = ELA_AABR[name].values
    if row != 0:
        x = x * ds_all['lapserate'].values
        y = y * ds_all['lapserate'].values
    
    mask = ~np.isnan(x) & ~np.isnan(y)
    new_x = x[mask]
    new_y = y[mask]
    
    t_stat, p_value = st.ttest_rel(new_x, new_y)
    rmse = mean_squared_error(new_x, new_y) ** 0.5
    
    data = new_x - new_y;
    if row == 0:
        ax.set_xlim([-110,110])
        ax.set_xticks([-100,-50,0,50,100])
        ax.set_ylim([0,0.05])
        ax.hist(data, bins = np.linspace(-100, 100, 30),
                edgecolor='white', linewidth=0.5, color=color, alpha=0.5, density=True)
    else:
        ax.set_xlim([-1.1,1.1])
        ax.set_xticks(np.arange(-1,1.5,0.5))
        ax.set_ylim([0,5.5])
        ax.hist(data, bins = np.linspace(-1, 1, 30),
                edgecolor='white', linewidth=0.5, color=color, alpha=0.5, density=True)
    
    # Normal
    compile_normx = np.linspace(data.min(), data.max(), 1000)
    compile_normy = st.norm.pdf(compile_normx, data.mean(), data.std())
    ax.plot(compile_normx, compile_normy, color=color, linestyle='-', linewidth=1)

    # Mean and Median
    ax.axvline(x=np.median(data), color=color, linestyle='--', linewidth=1)

    ax.errorbar(data.mean(), max(compile_normy), fmt='o', xerr=data.std(), capsize=2, elinewidth=1, capthick=1,
                  label=r'Mean with 1$\sigma$', c=color, markersize=3)
    if col == 0:
        ax.text(0.67, 0.91, 'AAR$_0$ = 0.580', fontsize=6, color=color, transform=ax.transAxes)
    elif col == 1:
        ax.text(0.67, 0.91, 'THAR$_0$ = 0.500', fontsize=6, color=color, transform=ax.transAxes)
    else:
        ax.text(0.67, 0.91, 'AABR$_0$ = 1.75', fontsize=6, color=color, transform=ax.transAxes)
    if p_value < alpha:
        ax.text(0.67, 0.83, 't: '+f"{t_stat:.0f}"+'$^*$', fontsize=6, color=color, transform=ax.transAxes)
    else:
        ax.text(0.67, 0.83, 't: '+f"{t_stat:.0f}", fontsize=6, color=color, transform=ax.transAxes)
    
    if row == 0:
        ax.text(0.67, 0.75, f'RMSE: {rmse:.0f} m', fontsize=6, color=color,transform=ax.transAxes)
    else:
        ax.text(0.67, 0.75, f'RMSE: {rmse:.2f} °C', fontsize=6, color=color,transform=ax.transAxes)
    
    if (col != 1) > 0:
        
        y = ds_all['ELA_median_compile'].values
        
        color = colors[1]
            
        if col == 0:
            name = ELA_AAR.columns[1]
            x = ELA_AAR[name].values
        else:
            name = ELA_AABR.columns[1]
            x = ELA_AABR[name].values
        
        if row != 0:
            x = x * ds_all['lapserate'].values
            y = y * ds_all['lapserate'].values
        
        mask = ~np.isnan(x) & ~np.isnan(y)
        new_x = x[mask]
        new_y = y[mask]
        
        t_stat, p_value = st.ttest_rel(new_x, new_y)
        rmse = mean_squared_error(new_x, new_y) ** 0.5
        
        data = new_x - new_y;
        
        if row == 0:
            ax.set_xlim([-110,110])
            ax.set_xticks([-100,-50,0,50,100])
            ax.set_ylim([0,0.05])
            ax.hist(data, bins = np.linspace(-100, 100, 30),
                    edgecolor='white', linewidth=0.5, color=color, alpha=0.5, density=True)
        else:
            ax.set_xlim([-1.1,1.1])
            ax.set_xticks(np.arange(-1,1.5,0.5))
            ax.set_ylim([0,5.5])
            ax.hist(data, bins = np.linspace(-1, 1, 30),
                    edgecolor='white', linewidth=0.5, color=color, alpha=0.5, density=True)
        
        # Normal
        compile_normx = np.linspace(data.min(), data.max(), 1000)
        compile_normy = st.norm.pdf(compile_normx, data.mean(), data.std())
        ax.plot(compile_normx, compile_normy, color=color, linestyle='-', linewidth=1)

        # Mean and Median
        ax.axvline(x=np.median(data), color=color, linestyle='--', linewidth=1)

        ax.errorbar(data.mean(), max(compile_normy), fmt='^', xerr=data.std(), capsize=2, elinewidth=1, capthick=1,
                      label=r'Mean with 1$\sigma$', c=color, markersize=3)
        
        if col == 0:
            ax.text(0.67, 0.67, 'AAR$_0$ = 0.500', fontsize=6, color=color, transform=ax.transAxes)
        else:
            ax.text(0.67, 0.67, 'AABR$_0$ = 1.56', fontsize=6, color=color, transform=ax.transAxes)
            
        if p_value < alpha:
            ax.text(0.67, 0.59, 't: '+f"{t_stat:.0f}"+'$^*$', fontsize=6, color=color, transform=ax.transAxes)
        else:
            ax.text(0.67, 0.59, 't: '+f"{t_stat:.0f}", fontsize=6, color=color, transform=ax.transAxes)
        if row == 0:
            ax.text(0.67, 0.51, f'RMSE: {rmse:.0f} m', fontsize=6, color=color,transform=ax.transAxes)
        else:
            ax.text(0.67, 0.51, f'RMSE: {rmse:.2f} °C', fontsize=6, color=color,transform=ax.transAxes)
    if row == 0:
        ax.set(xlabel='∆ELA (m)')
    else:
        ax.set(xlabel='∆T (°C)')
    if col == 0:
        ax.set(ylabel='Probability Density Function')
    else:
        ax.set_yticklabels([])
        ax.set_ylabel('')
    
    if row == 0:
        ax.set_title(titles[col], fontweight='bold')
    
    ax.text(0.03, 0.95, f'{chr(97+i)}', transform=ax.transAxes, fontsize=7, ha='center', va='center', fontweight='bold', color='black')
    
# legend
ax_legend = fig.add_subplot(gs[0,1])
pos = ax_legend.get_position()
ax_legend.set_position([pos.x0+0.006, pos.y0+0.24, pos.width*0.4, pos.height*0.28])
ax_legend.set_xticks([])
ax_legend.set_yticks([])
ax_legend.set_xticklabels([])
ax_legend.set_yticklabels([])
ax_legend.spines['top'].set_visible(False)
ax_legend.spines['right'].set_visible(False)
ax_legend.spines['bottom'].set_visible(False)
ax_legend.spines['left'].set_visible(False)
ax_legend.set_frame_on(True)
ax_legend.set_facecolor('lightgrey')

ax_legend.set_xlim([0,1])
ax_legend.set_ylim([0,1])
ax_legend.errorbar(0.13, 0.15, xerr=0.07, fmt='o', color='grey', ecolor='grey',capsize=2, elinewidth=1, capthick=1, markersize=2)
ax_legend.text(0.28, 0.14, 'Mean with 1σ', transform=ax_legend.transData, fontsize=5, 
        ha='left', va='center', color='black')

patch = mpatches.Patch(facecolor='grey', edgecolor='grey', linewidth=0.5, label='Histogram')
solid_line = mlines.Line2D([], [], color='grey', label='Gaussian DIST')
dashed_line = mlines.Line2D([], [], color='grey', linestyle='--', label='Median')

legend = ax_legend.legend(handles=[patch, solid_line, dashed_line],ncol=1,
                   loc='upper center', fontsize=5, frameon=True, handlelength=1.5, borderpad=0.2, labelspacing=0.3)
legend.get_frame().set_edgecolor('none')
legend.get_frame().set_facecolor('none')

out_pdf = '/Users/wyan0065/Desktop/AAR-disequilibrium/AABR/manuscript/' + 'figure_E1.png'
plt.savefig(out_pdf, dpi=300)
plt.show()

