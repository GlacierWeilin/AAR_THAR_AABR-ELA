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
from matplotlib.gridspec import GridSpec

file_path = '/Users/wyan0065/Desktop/AAR-disequilibrium/AABR/data_and_code/data/'
data = xr.open_dataset(file_path + 'results_all_mad_debriscalving.nc');

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

fig= plt.figure(figsize=(6.5, 4.5), dpi=600)
gs = GridSpec(2, 3, figure=fig, wspace=-0.01, hspace=0.01, height_ratios=[1,1], width_ratios=[1,1,1])
plt.subplots_adjust(left=0.09, right=1.05, top=1.05, bottom=0.03)


for kj in range(0,3):
    for ki in range(0,2):
        
        ax = fig.add_subplot(gs[ki, kj])
        
        if kj == 0:
            if ki == 0:
                loc = np.where((data['CenLat'].values >= 0) & (data['CenLat'].values < 30))[0]
            else:
                loc = np.where((data['CenLat'].values < 0) & (data['CenLat'].values >= -30))[0]
        elif kj == 1:
            if ki == 0:
                loc = np.where((data['CenLat'].values >= 30) & (data['CenLat'].values < 60))[0]
            else:
                loc = np.where((data['CenLat'].values < -30) & (data['CenLat'].values >= -60))[0]
        else:
            if ki == 0:
                loc = np.where(data['CenLat'].values >= 60)[0]
            else:
                loc = np.where(data['CenLat'].values < -60)[0]

        df = pd.DataFrame({
            'AAR$_0$': data['compile_AAR'].values[loc,0],
            'THAR$_0$': data['compile_THAR'].values[loc,0],
            'AABR$_0$': data['compile_AABR'].values[loc,0],
            'u (10m)': data['u10'].values[loc],
            'Sin (Aspect)': np.sin(np.deg2rad(data['Aspect'].values[loc])),
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

        mask = np.triu(np.ones_like(corr, dtype=bool))

        col_bounds = np.linspace(-0.4,0,7)
        col_bounds = np.append(col_bounds, np.linspace(0,0.8,7))
        cb = []
        cb_val = np.linspace(1, 0, len(col_bounds))
        for j in range(len(cb_val)):
            cb.append(mpl.cm.coolwarm_r(cb_val[j])) #'RdYlBu_r'
        cmap_cus = mpl.colors.LinearSegmentedColormap.from_list('my_cb', list(zip((col_bounds - min(col_bounds)) / (max(col_bounds - min(col_bounds))), 
                                                                                  cb)), N=1000)
        cbar_ticks = [-0.4, -0.2, 0, 0.2, 0.4, 0.6, 0.8]
        norm = mpl.colors.Normalize(vmin=min(col_bounds), vmax=max(col_bounds))
        if ki == 1:
            sns.heatmap(corr,annot=annot,cmap=cmap_cus,norm=norm,fmt='',mask=mask,linewidths=0.5,ax=ax,square=False,  
                        cbar_kws={'shrink': 0.8,'pad': 0.02,'aspect':30,'anchor': (0.0, 0.0),'orientation': 'horizontal', 
                                  'extend': 'both', 'ticks': cbar_ticks})
        else:
            sns.heatmap(corr,annot=annot,cmap=cmap_cus,norm=norm,fmt='',mask=mask,linewidths=0.5,ax=ax,square=False,  
                        cbar=False)

        ax.set_yticks(ax.get_yticks()[1:])
        ax.set_yticklabels(ax.get_yticklabels(), rotation=0)
        ax.set_xticks(ax.get_xticks()[0:-1])
        ax.set_xticklabels(ax.get_xticklabels(), rotation=30)
        if kj != 0:
            ax.set_yticklabels([])
        if ki != 1:
            ax.set_xticklabels([])
            

        n = np.sum(~np.isnan(df['AAR$_0$']))
        hh = 0.88
        if kj == 0:
            if ki == 0:
                ax.text(0.5, hh,f'(a) Low-altitude glaciers\n(0°–30° N; n = {n})',transform=ax.transAxes,
                        ha='center',va='top', fontweight='bold',fontsize=7)
            else:
                ax.text(0.5, hh,f'(d) Low-altitude glaciers\n(0°–30° S; n = {n})',transform=ax.transAxes,
                        ha='center',va='top', fontweight='bold',fontsize=7)
        elif kj == 1:
            if ki == 0:
                ax.text(0.5, hh,f'(b) Mid-altitude glaciers\n(30°–60° N; n = {n})',transform=ax.transAxes,
                        ha='center',va='top',fontweight='bold',fontsize=7)
            else:
                ax.text(0.5, hh,f'(e) Mid-altitude glaciers\n(30°–60° S; n = {n})',transform=ax.transAxes,
                        ha='center',va='top',fontweight='bold',fontsize=7)
        else:
            if ki == 0:
                ax.text(0.5, hh,f'(c) High-altitude glaciers\n(60°–90° N; n = {n})',transform=ax.transAxes,
                        ha='center',va='top',fontweight='bold',fontsize=7)
            else:
                ax.text(0.5, hh,f'(f) High-altitude glaciers\n(60°–90° S; n = {n})',transform=ax.transAxes,
                        ha='center',va='top',fontweight='bold',fontsize=7)

out_pdf = '/Users/wyan0065/Desktop/AAR-disequilibrium/AABR/manuscript/' + 'figure_S7.png'
plt.savefig(out_pdf, dpi=600)
plt.show()