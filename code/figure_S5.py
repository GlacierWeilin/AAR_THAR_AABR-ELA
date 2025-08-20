#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
Created on Wed Nov 13 23:03:18 2024

@author: Weilin Yang (weilinyang.yang@monash.edu)
'''

import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from sklearn.linear_model import LinearRegression
from scipy import stats

file_path = '/Users/wyan0065/Desktop/AAR-disequilibrium/AABR/data_and_code/data/'
calving = xr.open_dataset(file_path + 'results_AAR_debriscalving_withnan.nc');
debris  = xr.open_dataset(file_path + 'results_AAR_debrisonly_withnan.nc');
control = xr.open_dataset(file_path + 'results_AAR_control_withnan.nc');

find_id = np.where((np.isnan(control['steady_AAR'].values[:,0]) + np.isnan(debris['steady_AAR'].values[:,0])+
                   np.isnan(calving['steady_AAR'].values[:,0]))!=0)[0]
calving.close()
debris.close()
control.close()

calving = xr.open_dataset(file_path + 'results_all_mad_debriscalving.nc');
debris  = xr.open_dataset(file_path + 'results_all_mad_debrisonly.nc');
control = xr.open_dataset(file_path + 'results_all_mad_control.nc');

calving['is_debris'].values[find_id] = 0
calving['is_tidewater'].values[find_id] = 0

loc = np.where(np.isnan(calving['compile_AABR'].values[:,0]))[0]
calving['glac_acc'].values[loc,:] = np.nan
calving['glac_melt'].values[loc,:] = np.nan
loc = np.where(np.isnan(debris['compile_AABR'].values[:,0]))[0]
debris['glac_acc'].values[loc,:] = np.nan
debris['glac_melt'].values[loc,:] = np.nan

loc = np.where(np.isnan(control['compile_AABR'].values[:,0]))[0]
control['glac_acc'].values[loc,:] = np.nan
control['glac_melt'].values[loc,:] = np.nan

find_id = np.where(calving['is_debris']==1)[0]
control_debris = xr.DataArray(np.column_stack((
            control['glac_acc'].values[find_id,0], debris['glac_acc'].values[find_id,0],
            control['glac_melt'].values[find_id,0], debris['glac_melt'].values[find_id,0])),
            coords=[np.arange(0,len(find_id)), ['acc_control', 'acc_debris', 'melt_control', 'melt_debris']], dims=['id1', 'features1'])
                   
find_id = np.where(calving['is_tidewater']==1)[0]
debris_calving = xr.DataArray(np.column_stack((
            debris['glac_acc'].values[find_id,0], calving['glac_acc'].values[find_id,0],
            debris['glac_melt'].values[find_id,0], calving['glac_melt'].values[find_id,0])),
            coords=[np.arange(0,len(find_id)), ['acc_debris', 'acc_calving', 'melt_debris', 'melt_calving']], dims=['id2', 'features2'])


m = np.shape(calving['RGIId'].values)
find_id = np.where((calving['is_debris'].values + calving['is_tidewater'].values)>0)[0]
control_calving = xr.DataArray(np.column_stack((
            control['glac_acc'].values[find_id,0], calving['glac_acc'].values[find_id,0],
            control['glac_melt'].values[find_id,0], calving['glac_melt'].values[find_id,0])),
            coords=[np.arange(0,len(find_id)), ['acc_control', 'acc_calving', 'melt_control', 'melt_calving']], dims=['id3', 'features3'])

ds = xr.Dataset({
    'control_debris': control_debris,
    'debris_calving': debris_calving,
    'control_calving': control_calving
})

index = list(ds.data_vars)
title = ['Accumulation (km$^3$)', 'Melt (km$^3$)']
labelx = ['Control experiment', 'Debris-only experiment', 'Control experiment']
labely = ['Debris-only experiment', 'Debris-calving experiment', 'Debris-calving experiment']
lim = np.array([[0, 0],[1,1]])

alpha = 0.01

#%%
plt.rcParams.update({'lines.linewidth':1})
plt.rcParams.update({'font.size': 7})
plt.rcParams.update({'font.sans-serif': 'Arial'})

plt.rcParams.update({'axes.linewidth':0.7})
plt.rcParams.update({'axes.titlepad':3})
plt.rcParams.update({'axes.titlesize':7})
plt.rcParams.update({'axes.labelpad':2})
plt.rcParams.update({'xtick.major.pad':2})
plt.rcParams.update({'ytick.major.pad':2})
plt.rcParams.update({'xtick.major.width':0.7})
plt.rcParams.update({'ytick.major.width':0.7})
plt.rcParams.update({'xtick.major.size':1.5})
plt.rcParams.update({'ytick.major.size':1.5})

fig = plt.figure(figsize=(7.09, 4.5), dpi=600)
gs = GridSpec(3, 4, figure=fig, hspace=0.35, wspace=0.35, height_ratios=[1,1,1], width_ratios=[1,1,1,1])
plt.subplots_adjust(left=0.05, right=0.985, top=0.97, bottom=0.06)

for j in range(0,2):
    for i in range(0,3):
        ax = fig.add_subplot(gs[i, j], xlim=(lim[0,j],lim[1,j]), ylim=(lim[0,j],lim[1,j]))
        x = ds[index[i]].values[:,j*2]/1e9
        y = ds[index[i]].values[:,1+j*2]/1e9
        mask = ~np.isnan(x) & ~np.isnan(y)
        x = x[mask]
        y = y[mask]
        ax.scatter(x, y, c='#489FE3', s=0.1, alpha=0.8)
        x = x.reshape(-1, 1)
        y = y.reshape(-1, 1)
        model = LinearRegression(fit_intercept=False)
        model.fit(x, y)
        new_x = np.linspace(lim[0,j], lim[1,j], 100).reshape(-1, 1)
        y_fit = model.predict(new_x)
        ax.plot(new_x, y_fit, color='orange', linestyle='--')
        ax.plot(np.linspace(lim[0,j],lim[1,j],100), np.linspace(lim[0,j],lim[1,j],100), color='black', linestyle=':')
        t_stat, p_value = stats.ttest_rel(x, y)
        if p_value < alpha:
            ax.text(0.69, 0.37, '*', color='red', fontweight='bold', fontsize=10, transform=ax.transAxes)
            ax.text(0.69, 0.05, 'n='+str(len(x))+'\nk='+f'{model.coef_.item():.3f}'+
                    '\nx̅='+f'{np.mean(x):.3f}'+'\ny̅='+f'{np.mean(y):.3f}', transform=ax.transAxes)
        else:
            ax.text(0.69, 0.05, 'n='+str(len(x))+'\nk='+f'{model.coef_.item():.3f}'+
                    '\nx̅='+f'{np.mean(x):.3f}'+'\ny̅='+f'{np.mean(y):.3f}', transform=ax.transAxes)
                
        if i==0:
            ax.set_title(title[j], pad=0.5, fontweight='bold')
        ax.set_xlabel(labelx[i])
        ax.set_ylabel(labely[i])

out_pdf = '/Users/wyan0065/Desktop/AAR-disequilibrium/AABR/manuscript/' + 'figure_S5.png'
plt.savefig(out_pdf, dpi=600)
plt.show()