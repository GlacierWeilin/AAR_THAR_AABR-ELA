#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 10 13:14:07 2025

@author: Weilin Yang (weilinyang.yang@monash.edu)
"""

import numpy as np
import xarray as xr

def calc_stats(output):

    data = output[:,0];
    if np.isnan(data).all():
        return np.full(7, np.nan)
    
    n = np.sum(~np.isnan(data))
    mean = np.nanmean(data)
    std = np.nanstd(data)
    data = output[:,1];
    uncertainty = np.sqrt(np.nansum(data * data))/n
    
    return np.array([n, mean, std, uncertainty])

def equal_count_bins(data, n_bins, var_key=None):
    quantiles = np.linspace(0, 100, n_bins + 1)
    bin_edges = np.nanpercentile(data, quantiles)
    bin_edges = np.unique(bin_edges)

    if var_key == 'temp' or var_key == 'slope':
        bin_edges = np.round(bin_edges, 1)
    else:
        bin_edges = np.round(bin_edges, 2)

    return bin_edges


def calc_bin_stats(output_ds_all, var_name, target_var, n_bins=3):

    data = output_ds_all[var_name].values
    target_data = output_ds_all[target_var].values

    bin_edges = equal_count_bins(data, n_bins, var_key=var_key)

    n_bins = len(bin_edges) - 1

    stats_arr = np.full((n_bins, 4), np.nan)

    for i in range(n_bins):
        if i < n_bins - 1:
            mask = (data >= bin_edges[i]) & (data < bin_edges[i+1])
        else:
            mask = (data >= bin_edges[i]) & (data <= bin_edges[i+1])
        selected = target_data[mask]
        if selected.size > 0:
            stats_arr[i,:] = calc_stats(selected)
    return stats_arr, bin_edges

#%%

file_path = '/Users/wyan0065/Desktop/AAR-disequilibrium/AABR/data_and_code/data/'
output_ds_all = xr.open_dataset(file_path + 'results_all_mad_debriscalving.nc')

n_bins = 3

variables = {
    'temp': 'glacier_temp',
    'prcp': 'prcp',
    'slope': 'Slope',
    'area': 'Area'
}

target_vars = {
    'AAR': 'compile_AAR',
    'THAR': 'compile_THAR',
    'AABR': 'compile_AABR'
}

ds = xr.Dataset()

for target_key, target_var in target_vars.items():
    for var_key, var_name in variables.items():
        stats, bin_edges = calc_bin_stats(output_ds_all, var_name, target_var, n_bins=n_bins)
        dim_name = f'{var_key}_dim'
        ds.coords[dim_name] = np.arange(stats.shape[0])
        ds.coords[f'{dim_name}_bin_edges'] = (('bin_edge',), bin_edges)

        var_stats_name = f'{target_key}_{var_key}'
        ds.coords['stat_dim'] = np.arange(4)
        ds['stat_dim'].attrs['description'] = '0-n, 1-mean,2-std,3-mean uncertainty'

        ds[var_stats_name] = ( (dim_name, 'stat_dim'), stats )

ds.attrs['description'] = 'Each bin has roughly equal number of glaciers. Stats: n,mean,std,mean uncertainty'

output_filename = '/Users/wyan0065/Desktop/AAR-disequilibrium/AABR/data_and_code/data/classificationtree/results_equal_count_bins'+ str(n_bins) + '.nc'
ds.to_netcdf(output_filename)

output_ds_all.close()
ds.close()
