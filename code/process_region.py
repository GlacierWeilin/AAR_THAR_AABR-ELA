#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr  8 11:42:56 2025

@author: Weilin Yang (weilinyang.yang@monash.edu)
"""

import numpy as np
import xarray as xr
from scipy.stats import median_abs_deviation

def calc_stats(output):
    """
    Calculate stats for a given variable

    Parameters
    ----------
    vn : str
        variable name
    ds : xarray dataset
        dataset of output with all ensemble simulations

    Returns
    -------
    stats : np.array
        Statistics related to a given variable
    """
    if output.ndim == 2:
        data = output[:,0];
        stats = None
        stats = np.nanmean(data) # 'mean'
        stats = np.append(stats, np.nanstd(data)) # 'std'
        stats = np.append(stats, np.nanpercentile(data, 5)) # '5%'
        stats = np.append(stats, np.nanpercentile(data, 17)) # '17%'
        stats = np.append(stats, np.nanpercentile(data, 25)) # '25%'
        stats = np.append(stats, np.nanmedian(data)) # 'median'
        stats = np.append(stats, np.nanpercentile(data, 75)) # '75%'
        stats = np.append(stats, np.nanpercentile(data, 83)) # '83%'
        stats = np.append(stats, np.nanpercentile(data, 95)) # '95%'
        stats = np.append(stats, median_abs_deviation(data, nan_policy='omit')) # Compute the median absolute deviation of the data
        
        n = np.sum(~np.isnan(data));
        stats = np.append(stats, n);
        data  = output[:,1]; data = data*data;
        stats = np.append(stats, np.sqrt(np.nansum(data))/n)

    elif output.ndim == 1:
        data = output[:];
        stats = None
        stats = np.nanmean(data) # 'mean'
        stats = np.append(stats, np.nanstd(data)) # 'std'
        stats = np.append(stats, np.nanpercentile(data, 5)) # '5%'
        stats = np.append(stats, np.nanpercentile(data, 17)) # '17%'
        stats = np.append(stats, np.nanpercentile(data, 25)) # '25%'
        stats = np.append(stats, np.nanmedian(data)) # 'median'
        stats = np.append(stats, np.nanpercentile(data, 75)) # '75%'
        stats = np.append(stats, np.nanpercentile(data, 83)) # '83%'
        stats = np.append(stats, np.nanpercentile(data, 95)) # '95%'
        stats = np.append(stats, median_abs_deviation(data, nan_policy='omit')) # Compute the median absolute deviation of the data
        n = np.sum(~np.isnan(data));
        stats = np.append(stats, n);
        stats = np.append(stats, np.nan)
        
    return stats

#%% ===== Regional analysis =====

# prepare output
ds = xr.Dataset();
    
# Attributes
ds.attrs['Source'] = 'PyGEMv0.2.5 developed by David Rounce (drounce@alaska.edu)'
ds.attrs['Further developed by'] = 'Weilin Yang (weilinyang.yang@monash.edu)'

# Coordinates
ds.coords['dim'] = ('dim', np.arange(12))
ds['dim'].attrs['description'] = '0-mean, 1-std, 2-5%, 3-17%, 4-25%, 5-median, 6-75%, 7-83%, 8-95%, 9-mad, 10-n, 11-mean_std(std)'

ds.coords['exp_dim'] = ('exp_dim', np.arange(3))
ds['exp_dim'].attrs['description'] = '0-compile, 1-intercept, 2-steady'

ds.coords['region'] = ('region', np.arange(20))
ds['region'].attrs['description'] = '0-all, 1-19-01Region'
ds.coords['falseortrue'] = ('falseortrue', np.arange(2))
ds['falseortrue'].attrs['description'] = '0-False, 1-True'

# ELA_results
ds['ELA_region'] = (('region', 'exp_dim','dim'), np.zeros([20,3,12])*np.nan)
ds['ELA_region'].attrs['description'] = 'ELA results of each region'
ds['ELA_icecap'] = (('falseortrue', 'exp_dim','dim'), np.zeros([2,3,12])*np.nan)
ds['ELA_icecap'].attrs['description'] = 'ELA results of ice caps and glaciers'
ds['ELA_debris'] = (('falseortrue', 'exp_dim','dim'), np.zeros([2,3,12])*np.nan)
ds['ELA_debris'].attrs['description'] = 'ELA results of debris covered and no debris coverd glaciers'
ds['ELA_tidewater'] = (('falseortrue', 'exp_dim','dim'), np.zeros([2,3,12])*np.nan)
ds['ELA_tidewater'].attrs['description'] = 'ELA results of marine and land terminating glaciers'

# AAR_results
ds['AAR_region'] = (('region', 'exp_dim','dim'), np.zeros([20,3,12])*np.nan)
ds['AAR_region'].attrs['description'] = 'AAR results of each region'
ds['AAR_icecap'] = (('falseortrue', 'exp_dim','dim'), np.zeros([2,3,12])*np.nan)
ds['AAR_icecap'].attrs['description'] = 'AAR results of ice caps and glaciers'
ds['AAR_debris'] = (('falseortrue', 'exp_dim','dim'), np.zeros([2,3,12])*np.nan)
ds['AAR_debris'].attrs['description'] = 'AAR results of debris covered and no debris coverd glaciers'
ds['AAR_tidewater'] = (('falseortrue', 'exp_dim','dim'), np.zeros([2,3,12])*np.nan)
ds['AAR_tidewater'].attrs['description'] = 'AAR results of marine and land terminating glaciers'

# AABR_results
ds['AABR_region'] = (('region', 'exp_dim','dim'), np.zeros([20,3,12])*np.nan)
ds['AABR_region'].attrs['description'] = 'AABR results of each region'
ds['AABR_icecap'] = (('falseortrue', 'exp_dim','dim'), np.zeros([2,3,12])*np.nan)
ds['AABR_icecap'].attrs['description'] = 'AABR results of ice caps and glaciers'
ds['AABR_debris'] = (('falseortrue', 'exp_dim','dim'), np.zeros([2,3,12])*np.nan)
ds['AABR_debris'].attrs['description'] = 'AABR results of debris covered and no debris coverd glaciers'
ds['AABR_tidewater'] = (('falseortrue', 'exp_dim','dim'), np.zeros([2,3,12])*np.nan)
ds['AABR_tidewater'].attrs['description'] = 'AABR results of marine and land terminating glaciers'

# THAR_results
ds['THAR_region'] = (('region', 'exp_dim','dim'), np.zeros([20,3,12])*np.nan)
ds['THAR_region'].attrs['description'] = 'THAR results of each region'
ds['THAR_icecap'] = (('falseortrue', 'exp_dim','dim'), np.zeros([2,3,12])*np.nan)
ds['THAR_icecap'].attrs['description'] = 'THAR results of ice caps and glaciers'
ds['THAR_debris'] = (('falseortrue', 'exp_dim','dim'), np.zeros([2,3,12])*np.nan)
ds['THAR_debris'].attrs['description'] = 'THAR results of debris covered and no debris coverd glaciers'
ds['THAR_tidewater'] = (('falseortrue', 'exp_dim','dim'), np.zeros([2,3,12])*np.nan)
ds['THAR_tidewater'].attrs['description'] = 'THAR results of marine and land terminating glaciers'

# %% ===== all ======
file_path = '/Users/wyan0065/Desktop/AAR-disequilibrium/AABR/data_and_code/data/'
output_ds_all = xr.open_dataset(file_path + 'results_all_mad_debriscalving.nc');
i=0

ds['ELA_region'].values[i,0,:]  = calc_stats(output_ds_all['compile_ELA'].values);
ds['ELA_region'].values[i,1,:]  = calc_stats(output_ds_all['intercept_ELA'].values);
ds['ELA_region'].values[i,2,:]  = calc_stats(output_ds_all['steady_ELA'].values);

ds['AAR_region'].values[i,0,:]  = calc_stats(output_ds_all['compile_AAR'].values);
ds['AAR_region'].values[i,1,:]  = calc_stats(output_ds_all['intercept_AAR'].values);
ds['AAR_region'].values[i,2,:]  = calc_stats(output_ds_all['steady_AAR'].values);

ds['AABR_region'].values[i,0,:]  = calc_stats(output_ds_all['compile_AABR'].values);
ds['AABR_region'].values[i,1,:]  = calc_stats(output_ds_all['intercept_AABR'].values);
ds['AABR_region'].values[i,2,:]  = calc_stats(output_ds_all['steady_AABR'].values);

ds['THAR_region'].values[i,0,:]  = calc_stats(output_ds_all['compile_THAR'].values);
ds['THAR_region'].values[i,1,:]  = calc_stats(output_ds_all['intercept_THAR'].values);
ds['THAR_region'].values[i,2,:]  = calc_stats(output_ds_all['steady_THAR'].values);

#%% ===== Each region (01 region) =====
for i in range(1,20):
    # glac_results
    find_id = np.where(output_ds_all['O1Region'].values==i)[0];

    ds['ELA_region'].values[i,0,:]  = calc_stats(output_ds_all['compile_ELA'].values[find_id,:]);
    ds['ELA_region'].values[i,1,:]  = calc_stats(output_ds_all['intercept_ELA'].values[find_id,:]);
    ds['ELA_region'].values[i,2,:]  = calc_stats(output_ds_all['steady_ELA'].values[find_id,:]);
    
    ds['AAR_region'].values[i,0,:]  = calc_stats(output_ds_all['compile_AAR'].values[find_id,:]);
    ds['AAR_region'].values[i,1,:]  = calc_stats(output_ds_all['intercept_AAR'].values[find_id,:]);
    ds['AAR_region'].values[i,2,:]  = calc_stats(output_ds_all['steady_AAR'].values[find_id,:]);
    
    ds['AABR_region'].values[i,0,:]  = calc_stats(output_ds_all['compile_AABR'].values[find_id,:]);
    ds['AABR_region'].values[i,1,:]  = calc_stats(output_ds_all['intercept_AABR'].values[find_id,:]);
    ds['AABR_region'].values[i,2,:]  = calc_stats(output_ds_all['steady_AABR'].values[find_id,:]);
    
    ds['THAR_region'].values[i,0,:]  = calc_stats(output_ds_all['compile_THAR'].values[find_id,:]);
    ds['THAR_region'].values[i,1,:]  = calc_stats(output_ds_all['intercept_THAR'].values[find_id,:]);
    ds['THAR_region'].values[i,2,:]  = calc_stats(output_ds_all['steady_THAR'].values[find_id,:]);

#%% ===== is ice cap =====
for i in range(0,2):
    find_id = np.where(output_ds_all['is_icecap'].values==i)[0];
    n = len(find_id);
    if n!=0:
        ds['ELA_icecap'].values[i,0,:]  = calc_stats(output_ds_all['compile_ELA'].values[find_id,:]);
        ds['ELA_icecap'].values[i,1,:]  = calc_stats(output_ds_all['intercept_ELA'].values[find_id,:]);
        ds['ELA_icecap'].values[i,2,:]  = calc_stats(output_ds_all['steady_ELA'].values[find_id,:]);

        ds['AAR_icecap'].values[i,0,:]  = calc_stats(output_ds_all['compile_AAR'].values[find_id,:]);
        ds['AAR_icecap'].values[i,1,:]  = calc_stats(output_ds_all['intercept_AAR'].values[find_id,:]);
        ds['AAR_icecap'].values[i,2,:]  = calc_stats(output_ds_all['steady_AAR'].values[find_id,:]);

        ds['AABR_icecap'].values[i,0,:]  = calc_stats(output_ds_all['compile_AABR'].values[find_id,:]);
        ds['AABR_icecap'].values[i,1,:]  = calc_stats(output_ds_all['intercept_AABR'].values[find_id,:]);
        ds['AABR_icecap'].values[i,2,:]  = calc_stats(output_ds_all['steady_AABR'].values[find_id,:]);

        ds['THAR_icecap'].values[i,0,:]  = calc_stats(output_ds_all['compile_THAR'].values[find_id,:]);
        ds['THAR_icecap'].values[i,1,:]  = calc_stats(output_ds_all['intercept_THAR'].values[find_id,:]);
        ds['THAR_icecap'].values[i,2,:]  = calc_stats(output_ds_all['steady_THAR'].values[find_id,:]);

#%% ===== is debris covered glacier =====
for i in range(0,2):
    find_id = np.where(output_ds_all['is_debris'].values==i)[0];
    n = len(find_id);
    if n!=0:
        ds['ELA_debris'].values[i,0,:]  = calc_stats(output_ds_all['compile_ELA'].values[find_id,:]);
        ds['ELA_debris'].values[i,1,:]  = calc_stats(output_ds_all['intercept_ELA'].values[find_id,:]);
        ds['ELA_debris'].values[i,2,:]  = calc_stats(output_ds_all['steady_ELA'].values[find_id,:]);

        ds['AAR_debris'].values[i,0,:]  = calc_stats(output_ds_all['compile_AAR'].values[find_id,:]);
        ds['AAR_debris'].values[i,1,:]  = calc_stats(output_ds_all['intercept_AAR'].values[find_id,:]);
        ds['AAR_debris'].values[i,2,:]  = calc_stats(output_ds_all['steady_AAR'].values[find_id,:]);

        ds['AABR_debris'].values[i,0,:]  = calc_stats(output_ds_all['compile_AABR'].values[find_id,:]);
        ds['AABR_debris'].values[i,1,:]  = calc_stats(output_ds_all['intercept_AABR'].values[find_id,:]);
        ds['AABR_debris'].values[i,2,:]  = calc_stats(output_ds_all['steady_AABR'].values[find_id,:]);

        ds['THAR_debris'].values[i,0,:]  = calc_stats(output_ds_all['compile_THAR'].values[find_id,:]);
        ds['THAR_debris'].values[i,1,:]  = calc_stats(output_ds_all['intercept_THAR'].values[find_id,:]);
        ds['THAR_debris'].values[i,2,:]  = calc_stats(output_ds_all['steady_THAR'].values[find_id,:]);

#%% ===== is tidewater glacier =====
for i in range(0,2):
    find_id = np.where(output_ds_all['is_tidewater'].values==i)[0];
    n = len(find_id);
    if n!=0:
        ds['ELA_tidewater'].values[i,0,:]  = calc_stats(output_ds_all['compile_ELA'].values[find_id,:]);
        ds['ELA_tidewater'].values[i,1,:]  = calc_stats(output_ds_all['intercept_ELA'].values[find_id,:]);
        ds['ELA_tidewater'].values[i,2,:]  = calc_stats(output_ds_all['steady_ELA'].values[find_id,:]);

        ds['AAR_tidewater'].values[i,0,:]  = calc_stats(output_ds_all['compile_AAR'].values[find_id,:]);
        ds['AAR_tidewater'].values[i,1,:]  = calc_stats(output_ds_all['intercept_AAR'].values[find_id,:]);
        ds['AAR_tidewater'].values[i,2,:]  = calc_stats(output_ds_all['steady_AAR'].values[find_id,:]);

        ds['AABR_tidewater'].values[i,0,:]  = calc_stats(output_ds_all['compile_AABR'].values[find_id,:]);
        ds['AABR_tidewater'].values[i,1,:]  = calc_stats(output_ds_all['intercept_AABR'].values[find_id,:]);
        ds['AABR_tidewater'].values[i,2,:]  = calc_stats(output_ds_all['steady_AABR'].values[find_id,:]);

        ds['THAR_tidewater'].values[i,0,:]  = calc_stats(output_ds_all['compile_THAR'].values[find_id,:]);
        ds['THAR_tidewater'].values[i,1,:]  = calc_stats(output_ds_all['intercept_THAR'].values[find_id,:]);
        ds['THAR_tidewater'].values[i,2,:]  = calc_stats(output_ds_all['steady_THAR'].values[find_id,:]);

#%% To file
path = '/Users/wyan0065/Desktop/AAR-disequilibrium/AABR/data_and_code/data/results_region.nc';
enc_var = {'dtype': 'float32'}
encoding = {v: enc_var for v in ds.data_vars}
ds.to_netcdf(path, encoding=encoding);
output_ds_all.close()
ds.close()