#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 14 13:03:19 2025

@author: Weilin Yang (weilinyang.yang@monash.edu)
"""
import numpy as np
import xarray as xr
import pandas as pd

import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.gridspec import GridSpec

file_path = '/Users/wyan0065/Desktop/AAR-disequilibrium/AABR/data_and_code/data/'
data = xr.open_dataset(file_path + 'results_figure_S7.nc');
n = [215547, 190512, 25035];

#theta = np.deg2rad([90,135,180,225,270,315,0,45]) ### incorrect
theta = np.deg2rad([90, 45, 0, 315, 270, 225, 180, 135])
df_mean = pd.DataFrame({
        'AAR_global': data['AAR_aspect_global'].values[:,0],
        'AAR_north': data['AAR_aspect_north'].values[:,0],
        'AAR_south': data['AAR_aspect_south'].values[:,0],
        'THAR_global': data['THAR_aspect_global'].values[:,0],
        'THAR_north': data['THAR_aspect_north'].values[:,0],
        'THAR_south': data['THAR_aspect_south'].values[:,0],
        'AABR_global': data['AABR_aspect_global'].values[:,0],
        'AABR_north': data['AABR_aspect_north'].values[:,0],
        'AABR_south': data['AABR_aspect_south'].values[:,0]
})

df_number = pd.DataFrame({
        'AAR_global': data['AAR_aspect_global'].values[:,6]/n[0]*100,
        'AAR_north': data['AAR_aspect_north'].values[:,6]/n[1]*100,
        'AAR_south': data['AAR_aspect_south'].values[:,6]/n[2]*100,
        'THAR_global': data['THAR_aspect_global'].values[:,6]/n[0]*100,
        'THAR_north': data['THAR_aspect_north'].values[:,6]/n[1]*100,
        'THAR_south': data['THAR_aspect_south'].values[:,6]/n[2]*100,
        'AABR_global': data['AABR_aspect_global'].values[:,6]/n[0]*100,
        'AABR_north': data['AABR_aspect_north'].values[:,6]/n[1]*100,
        'AABR_south': data['AABR_aspect_south'].values[:,6]/n[2]*100
})

color_min = [0.535, 0.50, 1.3]
color_max = [0.56, 0.53, 2.0]

titles = ['Global', 'Northern Hemisphere', 'Southern Hemisphere']
row_labels = ['AAR$_0$', 'THAR$_0$', 'AABR$_0$']
row_colors = ['#489FE3', '#FEA909', '#DC6D57']

#%%
plt.rcParams.update({'lines.linewidth':0.5})
plt.rcParams.update({'font.size': 6})
plt.rcParams.update({'font.sans-serif': 'Arial'})

plt.rcParams.update({'axes.linewidth':0.5})
plt.rcParams.update({'axes.titlepad':3})
plt.rcParams.update({'axes.titlesize':7})
plt.rcParams.update({'axes.labelpad':2})
plt.rcParams.update({'xtick.major.pad':2})
plt.rcParams.update({'ytick.major.pad':2})
plt.rcParams.update({'xtick.major.width':0.5})
plt.rcParams.update({'ytick.major.width':0.5})
plt.rcParams.update({'xtick.major.size':1.5})
plt.rcParams.update({'ytick.major.size':1.5})

fig = plt.figure(figsize=(5.91, 5), dpi=600)
gs = GridSpec(3, 3, figure=fig, hspace=0.17, wspace=0.05, height_ratios=[1,1,1], width_ratios=[1,1,1])
plt.subplots_adjust(left=0.08, right=0.9, top=0.95, bottom=0.02)

for i in range(9):
    
    row = i // 3
    col = i % 3
    ax = fig.add_subplot(gs[row, col], polar=True)
    
    name = df_mean.columns[i]
    data = df_mean[name].values
    number = df_number[name].values
    
    norm = plt.Normalize(color_min[row], color_max[row])
    if row == 0:
        cmap = cm.Blues
        colors = cm.Blues(norm(data))
    elif row == 1:
        cmap = cm.Oranges
        colors = cm.Oranges(norm(data))
    else:
        cmap = cm.Reds
        colors = cm.Reds(norm(data))
    
    bars = ax.bar(theta, number, width=np.deg2rad(45), color=colors, edgecolor='none', linewidth=0, align='center')
    #labels = ['N', 'NW', 'W', 'SW', 'S', 'SE', 'E', 'NE']
    labels = ['N', 'NE', 'E', 'SE', 'S', 'SW', 'W', 'NW']
    ax.set_xticks(theta)
    ax.set_xticklabels(labels)
    ax.tick_params(axis='x', pad=-4)
    ax.grid(True, linestyle='--', linewidth=0.3, color='gray')
    
    ax.set_ylim([0,30])
    ax.set_yticks([5, 10, 15, 20, 25, 30])
    ax.set_yticklabels(['5','10','15','20','25','30%'], color='gray')
    
    if row == 0:
        ax.set_title(titles[col], fontweight='bold')
    if col == 0:
        ax.text(-0.25, 0.5, row_labels[row], transform=ax.transAxes, fontsize=7, ha='center', va='center', 
                fontweight='bold', color=row_colors[row])
    if col == 2:
        sm = cm.ScalarMappable(norm=norm, cmap=cmap)
        sm.set_array([])
        ax_colorbar = fig.add_subplot(gs[row, col])
        cbar = fig.colorbar(sm, ax=ax_colorbar, orientation='vertical', extend='both', fraction=0.046, pad=2)
        cbar.ax.tick_params(direction='in')
        cbar.ax.tick_params(labelsize=6)
        
        ax_colorbar.set_xticks([])
        ax_colorbar.set_yticks([])
        ax_colorbar.spines['top'].set_visible(False)
        ax_colorbar.spines['left'].set_visible(False)
        ax_colorbar.spines['right'].set_visible(False)
        ax_colorbar.spines['bottom'].set_visible(False)
        ax_colorbar.set_facecolor('none')
        
        pos = cbar.ax.get_position()
        new_pos = [pos.x0 + 0.05, pos.y0, pos.width, pos.height]
        cbar.ax.set_position(new_pos)
    
    ax.text(0, 1, f'{chr(97+i)}', transform=ax.transAxes, fontsize=7, ha='center', va='center', fontweight='bold', color='black')

out_pdf = '/Users/wyan0065/Desktop/AAR-disequilibrium/AABR/manuscript/' + 'figure_E3.png'
plt.savefig(out_pdf, dpi=600)
plt.show()

