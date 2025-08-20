#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 16 14:06:00 2025

@author: Weilin Yang (weilinyang.yang@monash.edu)
"""

import numpy as np
import pandas as pd
import xarray as xr

import scipy.stats as st
from scipy.stats import wilcoxon, mannwhitneyu
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import matplotlib.patches as mpatches
import matplotlib.lines as mlines

file_path = '/Users/wyan0065/Desktop/AAR-disequilibrium/AABR/data_and_code/data/';
wgms  = pd.read_csv(file_path + 'WGMS_comparison.csv')
wgms_AAR = wgms['wgms_AAR'].values
wgms_THAR = wgms['wgms_THAR'].values
wgms_AABR = wgms['wgms_AABR'].values
obs = pd.DataFrame({
    'AAR': wgms_AAR,
    'THAR': wgms_THAR,
    'AABR': wgms_AABR
})

AAR = wgms['compile_AAR'].values
THAR = wgms['compile_THAR'].values
AABR = wgms['compile_AABR'].values

simu = pd.DataFrame({
    'AAR': AAR,
    'THAR': THAR,
    'AABR': AABR
})

file_path = '/Users/wyan0065/Desktop/AAR-disequilibrium/AABR/data_and_code/data/'
ds_all = xr.open_dataset(file_path + 'results_all_mad_debriscalving.nc')
AAR_all  = ds_all['compile_AAR'].values[:,0]
THAR_all = ds_all['compile_THAR'].values[:,0]
AABR_all = ds_all['compile_AABR'].values[:,0]

AAR_all_clean  = AAR_all[~np.isnan(AAR_all)]
THAR_all_clean = THAR_all[~np.isnan(THAR_all)]
AABR_all_clean = AABR_all[~np.isnan(AABR_all)]

simu_all = pd.DataFrame({
    'AAR': AAR_all_clean,
    'THAR': THAR_all_clean,
    'AABR': AABR_all_clean
})

ds_all.close()

xlabels = ['AAR$_0$', 'THAR$_0$', 'AABR$_0$']
labels = [chr(ord('a') + i) for i in range(9)]
lims = np.array([[0,0,0],[1,1,15]])

alpha = 0.01

#%%

plt.rcParams.update({'lines.linewidth':1})
plt.rcParams.update({'font.size': 7})
plt.rcParams.update({'font.sans-serif': 'Arial'})

plt.rcParams.update({'axes.linewidth':0.7})
plt.rcParams.update({'axes.titlepad':3})
plt.rcParams.update({'axes.titlesize':8})
plt.rcParams.update({'axes.labelpad':1})
plt.rcParams.update({'xtick.major.pad':2})
plt.rcParams.update({'ytick.major.pad':2})
plt.rcParams.update({'xtick.major.width':0.7})
plt.rcParams.update({'ytick.major.width':0.7})
plt.rcParams.update({'xtick.major.size':1.5})
plt.rcParams.update({'ytick.major.size':1.5})

fig = plt.figure(figsize=(5.91, 5.2), dpi=600)
gs = GridSpec(3, 3, figure=fig, hspace=0.25, wspace=0.1, height_ratios=[1,1,1], width_ratios=[1,1,1])
plt.subplots_adjust(left=0.068, right=0.985, top=0.99, bottom=0.175)

j = 0
for i in range(3):
    ax = fig.add_subplot(gs[i, j])
    
    ## all simulated results
    name = obs.columns[i]
    x = obs[name].astype(float).values
    y = simu[name].astype(float).values
    #t_stat, p_value = st.ttest_rel(x, y)
    stat, p_value = wilcoxon(x, y)
    rmse = mean_squared_error(x, y) ** 0.5
    
    color = '#8A8CBF'
    ax.set_xlim([lims[0,i],lims[1,i]])
    ax.set_ylim([lims[0,i],lims[1,i]])
    ax.set_xlabel('WGMS '+ xlabels[i])
    ax.set_ylabel('Simulated '+xlabels[i])
    
    ax.scatter(x, y, c=color, s=2, alpha=0.6)
    x = x.reshape(-1, 1)
    y = y.reshape(-1, 1)
    model = LinearRegression(fit_intercept=False)
    model.fit(x, y)
    new_x = np.linspace(lims[0,i], lims[1,i], 100).reshape(-1, 1)
    y_fit = model.predict(new_x)
    #ax.plot(new_x, y_fit, color=color, linestyle='-')
    ax.plot(np.linspace(lims[0,i],lims[1,i],100), np.linspace(lims[0,i],lims[1,i],100), color='grey', linestyle=':')
    
    #if i == 0 or i == 2:
    #    if p_value < alpha:
    #        ax.text(0.69, 0.03, '\nk: '+f'{model.coef_.item():.3f}'+ '\nt: '+f'{t_stat:.3f}'+'$^*$'+
    #                '\np: '+f'{p_value:.3f}'+'\nRMSE: '+f'{rmse:.3f}', color=color, fontsize=6, transform=ax.transAxes)
    #    else:
    #        ax.text(0.69, 0.03, '\nk: '+f'{model.coef_.item():.3f}'+ '\nt: '+f'{t_stat:.3f}'+
    #                '\np: '+f'{p_value:.3f}'+'\nRMSE: '+f'{rmse:.3f}', color=color, fontsize=6, transform=ax.transAxes)
    #elif i == 1:
    #    if p_value < alpha:
    #        ax.text(0.69, 0.03, '\nk: '+f'{model.coef_.item():.3f}'+ '\nt: '+f'{t_stat:.3f}'+'$^*$'+
    #                '\np: '+f'{p_value:.3f}'+'\nRMSE: '+f'{rmse:.0f}', color=color, fontsize=6, transform=ax.transAxes)
    #    else:
    #        ax.text(0.69, 0.03, '\nk: '+f'{model.coef_.item():.3f}'+ '\nt: '+f'{t_stat:.3f}'+
    #                '\np: '+f'{p_value:.3f}'+'\nRMSE: '+f'{rmse:.0f}', color=color, fontsize=6, transform=ax.transAxes)
    #else:
    #    if p_value < alpha:
    #        ax.text(0.69, 0.03, '\nk: '+f'{model.coef_.item():.3f}'+ '\nt: '+f'{t_stat:.3f}'+'$^*$'+
    #                '\np: '+f'{p_value:.3f}'+'\nRMSE: '+f'{rmse:.2f}', color=color, fontsize=6, transform=ax.transAxes)
    #    else:
    #        ax.text(0.69, 0.03, '\nk: '+f'{model.coef_.item():.3f}'+ '\nt: '+f'{t_stat:.3f}'+
    #                '\np: '+f'{p_value:.3f}'+'\nRMSE: '+f'{rmse:.2f}', color=color, fontsize=6, transform=ax.transAxes)
    
    if i == 0 or i == 1:
        if p_value < alpha:
            ax.text(0.69, 0.03, #'\nstat: '+f'{stat:.3f}'+'$^*$'+
                    '\np: '+f'{p_value:.3f}'+'\nRMSE: '+f'{rmse:.3f}', color=color, fontsize=6, transform=ax.transAxes)
        else:
            ax.text(0.69, 0.03, #'\nstat: '+f'{stat:.3f}'+
                    '\np: '+f'{p_value:.3f}'+'\nRMSE: '+f'{rmse:.3f}', color=color, fontsize=6, transform=ax.transAxes)
    else:
        if p_value < alpha:
            ax.text(0.69, 0.03, #'\nstat: '+f'{stat:.3f}'+'$^*$'+
                    '\np: '+f'{p_value:.3f}'+'\nRMSE: '+f'{rmse:.2f}', color=color, fontsize=6, transform=ax.transAxes)
        else:
            ax.text(0.69, 0.03, #'\nstat: '+f'{stat:.3f}'+
                    '\np: '+f'{p_value:.3f}'+'\nRMSE: '+f'{rmse:.2f}', color=color, fontsize=6, transform=ax.transAxes)
    
    ax.text(0.03, 0.94, labels[j+3*i], transform=ax.transAxes, fontsize=7, ha='center', va='center', fontweight='bold', color='black')
    
lims = np.array([[0,0,0],[1,1,6]])

j = 1
for i in range(3):
    ax = fig.add_subplot(gs[i, j])
    
    ## all simulated results
    name = obs.columns[i]
    data = simu[name].values
    color = 'orange'
    mean_simu = np.mean(data)
    median_simu = np.median(data)
    
    ax.set_xlim([lims[0,i],lims[1,i]])
    ax.set_yticks([])
    ax.set_ylabel('')
    ax.set_xlabel(xlabels[i])
    
    # Bar chart
    ax.hist(data, bins = np.linspace(lims[0,i],lims[1,i], 30),edgecolor='white', linewidth=0.5, color=color, alpha=0.5, density=True)
    
    # Normal
    compile_normx = np.linspace(lims[0,i], lims[1,i], 1000)
    compile_normy = st.norm.pdf(compile_normx, data.mean(), data.std())
    ax.plot(compile_normx, compile_normy, color=color, linestyle='-', linewidth=1)

    # Mean and Median
    ax.axvline(x=np.median(data), color=color, linestyle='--', linewidth=1)

    ax.errorbar(data.mean(), max(compile_normy), fmt='s', xerr=data.std(), capsize=2, elinewidth=1, capthick=1,
                  label=r'Mean with 1$\sigma$', c=color, markersize=3)
    
    if i == 0 or i == 1:
        ax.text(0.68, 0.66, 'Mean: '+f'{mean_simu:.3f}'+'\nMedian: '+f'{median_simu:.3f}', color=color, fontsize=6, transform=ax.transAxes)
    else:
        ax.text(0.68, 0.66, 'Mean: '+f'{mean_simu:.2f}'+'\nMedian: '+f'{median_simu:.2f}', color=color, fontsize=6, transform=ax.transAxes)
    
    ## WGMS
    data = obs[name].values
    color = '#489FE3'
    mean_obs = np.mean(data)
    median_obs = np.median(data)
    
    # Bar chart
    ax.hist(data, bins = np.linspace(lims[0,i],lims[1,i], 30),edgecolor='white', linewidth=0.5, color=color, alpha=0.5, density=True)
    
    # Normal
    compile_normx = np.linspace(lims[0,i], lims[1,i], 1000)
    compile_normy = st.norm.pdf(compile_normx, data.mean(), data.std())
    ax.plot(compile_normx, compile_normy, color=color, linestyle='-', linewidth=1)

    # Mean and Median
    ax.axvline(x=np.median(data), color=color, linestyle='--', linewidth=1)

    ax.errorbar(data.mean(), max(compile_normy), fmt='o', xerr=data.std(), capsize=2, elinewidth=1, capthick=1,
                  label=r'Mean with 1$\sigma$', c=color, markersize=3)
    
    if i == 0 or i == 1:
        ax.text(0.68, 0.82, 'Mean: '+f'{mean_obs:.3f}'+'\nMedian: '+f'{median_obs:.3f}', color=color, fontsize=6, transform=ax.transAxes)
    else:
        ax.text(0.68, 0.82, 'Mean: '+f'{mean_obs:.2f}'+'\nMedian: '+f'{median_obs:.2f}', color=color, fontsize=6, transform=ax.transAxes)
        
    ax.text(0.03, 0.94, labels[j+3*i], transform=ax.transAxes, fontsize=7, ha='center', va='center', fontweight='bold', color='black')
        
j = 2
for i in range(3):
    ax = fig.add_subplot(gs[i, j])
    
    ## all simulated results
    name = obs.columns[i]
    data = simu_all[name].values
    mask = ~np.isnan(data)
    data = data[mask]
    
    x = obs[name].astype(float).values
    stat, p_value = mannwhitneyu(x, data, alternative='two-sided')
    
    color = '#DC6D57'
    mean_simu = np.mean(data)
    median_simu = np.median(data)
    
    ax.set_xlim([lims[0,i],lims[1,i]])
    ax.set_yticks([])
    ax.set_ylabel('')
    ax.set_xlabel(xlabels[i])
    
    # Bar chart
    ax.hist(data, bins = np.linspace(lims[0,i],lims[1,i], 30),edgecolor='white', linewidth=0.5, color=color, alpha=0.5, density=True)
    
    # Normal
    compile_normx = np.linspace(lims[0,i], lims[1,i], 1000)
    compile_normy = st.norm.pdf(compile_normx, data.mean(), data.std())
    ax.plot(compile_normx, compile_normy, color=color, linestyle='-', linewidth=1)

    # Mean and Median
    ax.axvline(x=np.median(data), color=color, linestyle='--', linewidth=1)

    ax.errorbar(data.mean(), max(compile_normy), fmt='^', xerr=data.std(), capsize=2, elinewidth=1, capthick=1,
                  label=r'Mean with 1$\sigma$', c=color, markersize=3)
    
    if i == 0 or i == 1:
        ax.text(0.68, 0.82, 'Mean: '+f'{mean_simu:.3f}'+'\nMedian: '+f'{median_simu:.3f}', color=color, fontsize=6, transform=ax.transAxes)
        ax.text(0.68, 0.74, '\np: '+f'{p_value:.3f}', color='black', fontsize=6, transform=ax.transAxes)
    else:
        ax.text(0.68, 0.82, 'Mean: '+f'{mean_simu:.2f}'+'\nMedian: '+f'{median_simu:.2f}', color=color, fontsize=6, transform=ax.transAxes)
        ax.text(0.68, 0.74, '\np: '+f'{p_value:.3f}', color='black', fontsize=6, transform=ax.transAxes)
        
    ## WGMS
    #data = obs[name].values
    #color = '#489FE3'
    #mean_obs = np.mean(data)
    #median_obs = np.median(data)
    
    # Bar chart
    #ax.hist(data, bins = np.linspace(lims[0,i],lims[1,i], 30),edgecolor='white', linewidth=0.5, color=color, alpha=0.5, density=True)
    
    # Normal
    #compile_normx = np.linspace(lims[0,i], lims[1,i], 1000)
    #compile_normy = st.norm.pdf(compile_normx, data.mean(), data.std())
    #ax.plot(compile_normx, compile_normy, color=color, linestyle='-', linewidth=1)

    # Mean and Median
    #ax.axvline(x=np.median(data), color=color, linestyle='--', linewidth=1)

    #ax.errorbar(data.mean(), max(compile_normy), fmt='o', xerr=data.std(), capsize=2, elinewidth=1, capthick=1,
    #              label=r'Mean with 1$\sigma$', c=color, markersize=3)
    
    #if i == 0 or i == 2:
    #    ax.text(0.68, 0.66, 'Mean: '+f'{mean_obs:.3f}'+'\nMedian: '+f'{median_obs:.3f}', color=color, fontsize=6, transform=ax.transAxes)
    #elif i == 1:
    #    ax.text(0.68, 0.66, 'Mean: '+f'{mean_obs:.0f}'+'\nMedian: '+f'{median_obs:.0f}', color=color, fontsize=6, transform=ax.transAxes)
    #else:
    #    ax.text(0.68, 0.66, 'Mean: '+f'{mean_obs:.2f}'+'\nMedian: '+f'{median_obs:.2f}', color=color, fontsize=6, transform=ax.transAxes)
    
    ax.text(0.03, 0.94, labels[j+3*i], transform=ax.transAxes, fontsize=7, ha='center', va='center', fontweight='bold', color='black')

# legend
ax_legend = fig.add_subplot(gs[2,0])
pos = ax_legend.get_position()
ax_legend.set_position([pos.x0, pos.y0 - 0.18, pos.width, pos.height])
ax_legend.set_xticks([])
ax_legend.set_yticks([])
ax_legend.set_xticklabels([])
ax_legend.set_yticklabels([])
ax_legend.spines['top'].set_visible(False)
ax_legend.spines['right'].set_visible(False)
ax_legend.spines['bottom'].set_visible(False)
ax_legend.spines['left'].set_visible(False)
ax_legend.set_frame_on(False)

scatter_handle = mlines.Line2D([], [], color='#8A8CBF', alpha=0.6, marker='o', linestyle='None',markersize=2, 
                               label='WGMS glaciers (n=' + f'{np.shape(wgms_AAR)[0]:.0f}' + ')')
dashed_line = mlines.Line2D([], [], color='grey', linestyle=':', label='y = x')
solid_line = mlines.Line2D([], [], color='white')


legend = ax_legend.legend(handles=[scatter_handle, dashed_line, solid_line],
                   loc='lower center', fontsize=7, frameon=True, handlelength=1.5, borderpad=0.1)
for i, text in enumerate(legend.get_texts()):
    if i == 1:
        text.set_color('grey')
    else:
        text.set_color('#8A8CBF')
    text.set_fontweight('bold')

legend.get_frame().set_edgecolor('none')
legend.get_frame().set_facecolor('none')

ax_legend = fig.add_subplot(gs[2,1])
pos = ax_legend.get_position()
ax_legend.set_position([pos.x0 + 0.08, pos.y0 - 0.18, pos.width*2, pos.height])
ax_legend.set_xticks([])
ax_legend.set_yticks([])
ax_legend.set_xticklabels([])
ax_legend.set_yticklabels([])
ax_legend.spines['top'].set_visible(False)
ax_legend.spines['right'].set_visible(False)
ax_legend.spines['bottom'].set_visible(False)
ax_legend.spines['left'].set_visible(False)
ax_legend.set_frame_on(False)

colors = ['#489FE3', 'orange', '#DC6D57']
ax_legend.set_xlim([0,1])
ax_legend.set_ylim([0,1])
ax_legend.errorbar(0.83, 0.35, xerr=0.03, fmt='o', color=colors[0], ecolor=colors[0],capsize=2, elinewidth=1, capthick=1, markersize=3)
ax_legend.text(-0.14, 0.35, 'WGMS glaciers (n=' + f'{np.shape(wgms_AAR)[0]:.0f}' + ')', transform=ax_legend.transData, fontsize=7, 
        ha='left', va='center', fontweight='bold', color=colors[0])
ax_legend.errorbar(0.83, 0.23, xerr=0.03, fmt='s', color=colors[1], ecolor=colors[1],capsize=2, elinewidth=1, capthick=1, markersize=3)
ax_legend.text(-0.14, 0.23, 'Simulated glaciers (n=' + f'{np.shape(wgms_AAR)[0]:.0f}' + ')', transform=ax_legend.transData, fontsize=7, 
        ha='left', va='center', fontweight='bold', color=colors[1])
ax_legend.errorbar(0.83, 0.1, xerr=0.03, fmt='^', color=colors[2], ecolor=colors[2],capsize=2, elinewidth=1, capthick=1, markersize=3)
ax_legend.text(-0.14, 0.1, 'Simulated glaciers (n=' + f'{np.sum(~np.isnan(AAR_all)):.0f}' + ')', transform=ax_legend.transData, fontsize=7, 
        ha='left', va='center', fontweight='bold', color=colors[2])

ax_legend.text(0.24, 0.45, 'Histogram  Gaussian DIST  Median   Mean with 1σ', transform=ax_legend.transData, fontsize=7, 
        ha='left', va='center', fontweight='bold', color='black')

handles = []
for color in colors:
    handle = mpatches.Patch(facecolor=color, edgecolor='none', linewidth=0.5, label='')
    handles = np.append(handles, handle)
    handle = mlines.Line2D([], [], color=color, label='')
    handles = np.append(handles, handle)
    handle = mlines.Line2D([], [], color=color, linestyle='--', label='')
    handles = np.append(handles, handle)

handles = [handles[0], handles[3], handles[6], handles[1], handles[4], handles[7], handles[2], handles[5], handles[8]]
legend = ax_legend.legend(handles=handles,ncol=3,
                   loc='lower center', fontsize=7, frameon=True, handlelength=1.5, borderpad=0.1, columnspacing=4)
legend.get_frame().set_edgecolor('none')
legend.get_frame().set_facecolor('none')

out_pdf = '/Users/wyan0065/Desktop/AAR-disequilibrium/AABR/manuscript/' + 'figure_2.png'
plt.savefig(out_pdf, dpi=600)
plt.show()