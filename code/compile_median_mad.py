#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun  9 15:30:13 2025

@author: Weilin Yang (weilinyang.yang@monash.edu)
"""

import numpy as np
import xarray as xr

from scipy.stats import median_abs_deviation

path = '/Users/wyan0065/Desktop/AAR-disequilibrium/AABR/data_and_code/data/';
#fn = 'results_all_debriscalving.nc';
#fn = 'results_all_control.nc';
fn = 'results_all_debrisonly.nc';
ds = xr.open_dataset(path+fn);

#%% compile
# AAR
ds['compile_AAR'] = (('RGIId', 'dim'), np.zeros([np.shape(ds['RGIId'])[0],2])*np.nan)
ds['compile_AAR'].values[:, 0] = np.nanmedian(
    np.stack([
        ds['AAR_min_compile'].values,
        ds['AAR_median_compile'].values,
        ds['AAR_max_compile'].values
    ], axis=1),
    axis=1
)

ds['compile_AAR'].values[:, 1] = median_abs_deviation(
    np.stack([
        ds['AAR_min_compile'].values,
        ds['AAR_median_compile'].values,
        ds['AAR_max_compile'].values
    ], axis=1),
    axis=1,
    nan_policy='omit'
)
ds['compile_AAR'].attrs['long_name'] = 'glacier steady-state Accumulation Area Ratio'

# ELA
ds['compile_ELA'] = (('RGIId', 'dim'), np.zeros([np.shape(ds['RGIId'])[0],2])*np.nan)
ds['compile_ELA'].values[:, 0] = np.nanmedian(
    np.stack([
        ds['ELA_min_compile'].values,
        ds['ELA_median_compile'].values,
        ds['ELA_max_compile'].values
    ], axis=1),
    axis=1
)

ds['compile_ELA'].values[:, 1] = median_abs_deviation(
    np.stack([
        ds['ELA_min_compile'].values,
        ds['ELA_median_compile'].values,
        ds['ELA_max_compile'].values
    ], axis=1),
    axis=1,
    nan_policy='omit'
)
ds['compile_ELA'].attrs['long_name'] = 'glacier steady-state equilibrium-line altitude'

# THAR
ds['compile_THAR'] = (('RGIId', 'dim'), np.zeros([np.shape(ds['RGIId'])[0],2])*np.nan)
ds['compile_THAR'].values[:, 0] = np.nanmedian(
    np.stack([
        ds['THAR_min_compile'].values,
        ds['THAR_median_compile'].values,
        ds['THAR_max_compile'].values
    ], axis=1),
    axis=1
)

ds['compile_THAR'].values[:, 1] = median_abs_deviation(
    np.stack([
        ds['THAR_min_compile'].values,
        ds['THAR_median_compile'].values,
        ds['THAR_max_compile'].values
    ], axis=1),
    axis=1,
    nan_policy='omit'
)
ds['compile_THAR'].attrs['long_name'] = 'glacier steady-state terminus-to-head altitude ratio'

# AABR
ds['compile_AABR'] = (('RGIId', 'dim'), np.zeros([np.shape(ds['RGIId'])[0],2])*np.nan)
ds['compile_AABR'].values[:, 0] = np.nanmedian(
    np.stack([
        ds['AABR_min_compile'].values,
        ds['AABR_median_compile'].values,
        ds['AABR_max_compile'].values
    ], axis=1),
    axis=1
)

ds['compile_AABR'].values[:, 1] = median_abs_deviation(
    np.stack([
        ds['AABR_min_compile'].values,
        ds['AABR_median_compile'].values,
        ds['AABR_max_compile'].values
    ], axis=1),
    axis=1,
    nan_policy='omit'
)
ds['compile_AABR'].attrs['long_name'] = 'glacier steady-state Accumulation Area Balance Ratio'


#%% intercept
# AAR
ds['intercept_AAR'] = (('RGIId', 'dim'), np.zeros([np.shape(ds['RGIId'])[0],2])*np.nan)
ds['intercept_AAR'].values[:, 0] = np.nanmedian(
    np.stack([
        ds['AAR_min_intercept'].values,
        ds['AAR_median_intercept'].values,
        ds['AAR_max_intercept'].values
    ], axis=1),
    axis=1
)

ds['intercept_AAR'].values[:, 1] = median_abs_deviation(
    np.stack([
        ds['AAR_min_intercept'].values,
        ds['AAR_median_intercept'].values,
        ds['AAR_max_intercept'].values
    ], axis=1),
    axis=1,
    nan_policy='omit'
)
ds['intercept_AAR'].attrs['long_name'] = 'glacier steady-state Accumulation Area Ratio'

# ELA
ds['intercept_ELA'] = (('RGIId', 'dim'), np.zeros([np.shape(ds['RGIId'])[0],2])*np.nan)
ds['intercept_ELA'].values[:, 0] = np.nanmedian(
    np.stack([
        ds['ELA_min_intercept'].values,
        ds['ELA_median_intercept'].values,
        ds['ELA_max_intercept'].values
    ], axis=1),
    axis=1
)

ds['intercept_ELA'].values[:, 1] = median_abs_deviation(
    np.stack([
        ds['ELA_min_intercept'].values,
        ds['ELA_median_intercept'].values,
        ds['ELA_max_intercept'].values
    ], axis=1),
    axis=1,
    nan_policy='omit'
)
ds['intercept_ELA'].attrs['long_name'] = 'glacier steady-state equilibrium-line altitude'

# THAR
ds['intercept_THAR'] = (('RGIId', 'dim'), np.zeros([np.shape(ds['RGIId'])[0],2])*np.nan)
ds['intercept_THAR'].values[:, 0] = np.nanmedian(
    np.stack([
        ds['THAR_min_intercept'].values,
        ds['THAR_median_intercept'].values,
        ds['THAR_max_intercept'].values
    ], axis=1),
    axis=1
)

ds['intercept_THAR'].values[:, 1] = median_abs_deviation(
    np.stack([
        ds['THAR_min_intercept'].values,
        ds['THAR_median_intercept'].values,
        ds['THAR_max_intercept'].values
    ], axis=1),
    axis=1,
    nan_policy='omit'
)
ds['intercept_THAR'].attrs['long_name'] = 'glacier steady-state terminus-to-head altitude ratio'

# AABR
ds['intercept_AABR'] = (('RGIId', 'dim'), np.zeros([np.shape(ds['RGIId'])[0],2])*np.nan)
ds['intercept_AABR'].values[:, 0] = np.nanmedian(
    np.stack([
        ds['AABR_min_intercept'].values,
        ds['AABR_median_intercept'].values,
        ds['AABR_max_intercept'].values
    ], axis=1),
    axis=1
)

ds['intercept_AABR'].values[:, 1] = median_abs_deviation(
    np.stack([
        ds['AABR_min_intercept'].values,
        ds['AABR_median_intercept'].values,
        ds['AABR_max_intercept'].values
    ], axis=1),
    axis=1,
    nan_policy='omit'
)
ds['intercept_AABR'].attrs['long_name'] = 'glacier steady-state Accumulation Area Balance Ratio'

#%% steady
# AAR
ds['steady_AAR'] = (('RGIId', 'dim'), np.zeros([np.shape(ds['RGIId'])[0],2])*np.nan)
ds['steady_AAR'].values[:, 0] = np.nanmedian(
    np.stack([
        ds['AAR_min_steady'].values,
        ds['AAR_median_steady'].values,
        ds['AAR_max_steady'].values
    ], axis=1),
    axis=1
)

ds['steady_AAR'].values[:, 1] = median_abs_deviation(
    np.stack([
        ds['AAR_min_steady'].values,
        ds['AAR_median_steady'].values,
        ds['AAR_max_steady'].values
    ], axis=1),
    axis=1,
    nan_policy='omit'
)
ds['steady_AAR'].attrs['long_name'] = 'glacier steady-state Accumulation Area Ratio'

# ELA
ds['steady_ELA'] = (('RGIId', 'dim'), np.zeros([np.shape(ds['RGIId'])[0],2])*np.nan)
ds['steady_ELA'].values[:, 0] = np.nanmedian(
    np.stack([
        ds['ELA_min_steady'].values,
        ds['ELA_median_steady'].values,
        ds['ELA_max_steady'].values
    ], axis=1),
    axis=1
)

ds['steady_ELA'].values[:, 1] = median_abs_deviation(
    np.stack([
        ds['ELA_min_steady'].values,
        ds['ELA_median_steady'].values,
        ds['ELA_max_steady'].values
    ], axis=1),
    axis=1,
    nan_policy='omit'
)
ds['steady_ELA'].attrs['long_name'] = 'glacier steady-state equilibrium-line altitude'

# THAR
ds['steady_THAR'] = (('RGIId', 'dim'), np.zeros([np.shape(ds['RGIId'])[0],2])*np.nan)
ds['steady_THAR'].values[:, 0] = np.nanmedian(
    np.stack([
        ds['THAR_min_steady'].values,
        ds['THAR_median_steady'].values,
        ds['THAR_max_steady'].values
    ], axis=1),
    axis=1
)

ds['steady_THAR'].values[:, 1] = median_abs_deviation(
    np.stack([
        ds['THAR_min_steady'].values,
        ds['THAR_median_steady'].values,
        ds['THAR_max_steady'].values
    ], axis=1),
    axis=1,
    nan_policy='omit'
)
ds['steady_THAR'].attrs['long_name'] = 'glacier steady-state terminus-to-head altitude ratio'

# AABR
ds['steady_AABR'] = (('RGIId', 'dim'), np.zeros([np.shape(ds['RGIId'])[0],2])*np.nan)
ds['steady_AABR'].values[:, 0] = np.nanmedian(
    np.stack([
        ds['AABR_min_steady'].values,
        ds['AABR_median_steady'].values,
        ds['AABR_max_steady'].values
    ], axis=1),
    axis=1
)

ds['steady_AABR'].values[:, 1] = median_abs_deviation(
    np.stack([
        ds['AABR_min_steady'].values,
        ds['AABR_median_steady'].values,
        ds['AABR_max_steady'].values
    ], axis=1),
    axis=1,
    nan_policy='omit'
)
ds['steady_AABR'].attrs['long_name'] = 'glacier steady-state Accumulation Area Balance Ratio'
#%%
variables_to_remove = ['ELA_median_compile',
 'ELA_min_compile',
 'ELA_max_compile',
 'AAR_median_compile',
 'AAR_min_compile',
 'AAR_max_compile',
 'AABR_median_compile',
 'AABR_min_compile',
 'AABR_max_compile',
 'THAR_median_compile',
 'THAR_min_compile',
 'THAR_max_compile',
 'ELA_median_intercept',
 'ELA_min_intercept',
 'ELA_max_intercept',
 'AAR_median_intercept',
 'AAR_min_intercept',
 'AAR_max_intercept',
 'AABR_median_intercept',
 'AABR_min_intercept',
 'AABR_max_intercept',
 'THAR_median_intercept',
 'THAR_min_intercept',
 'THAR_max_intercept',
 'ELA_median_steady',
 'ELA_min_steady',
 'ELA_max_steady',
 'AAR_median_steady',
 'AAR_min_steady',
 'AAR_max_steady',
 'AABR_median_steady',
 'AABR_min_steady',
 'AABR_max_steady',
 'THAR_median_steady',
 'THAR_min_steady',
 'THAR_max_steady']

ds = ds.drop_vars(variables_to_remove)

ds.attrs = {'Source' : 'PyGEMv0.2.5 developed by David Rounce (drounce@alaska.edu)',
                       'Further developed by': 'Weilin Yang (weilinyang.yang@monash.edu)'}

#output_path = path+'results_all_mad_debriscalving.nc'
#output_path = path+'results_all_mad_control.nc'
output_path = path+'results_all_mad_debrisonly.nc'
ds.to_netcdf(output_path)