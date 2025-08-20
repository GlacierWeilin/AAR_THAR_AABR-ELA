#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 14 13:41:48 2025

@author: Weilin Yang (weilinyang.yang@monash.edu)
"""

import numpy as np
import xarray as xr

from scipy.stats import median_abs_deviation

def calc_stats(output):
    '''
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
    '''
    
    data = output[:];
    stats = None
    if (np.isnan(data)).all():
        stats = np.zeros(3) * np.nan
    else:
        stats = np.nanmean(data) # 'mean'
        stats = np.append(stats, np.nanstd(data)) # 'std'
        stats = np.append(stats, np.nanmedian(data)) # 'median'
        stats = np.append(stats, median_abs_deviation(data, nan_policy='omit'))
        stats = np.append(stats, np.nanpercentile(data, 17)) # '17%'
        stats = np.append(stats, np.nanpercentile(data, 83)) # '83%'
        stats = np.append(stats,np.sum(~np.isnan(data)))
        
    return stats

# prepare output
ds = xr.Dataset();
    
# Attributes
ds.attrs['description'] = 'OGGM & PyGEM output'
ds.attrs['version'] = 'OGGM1.6.0 & PyGEM0.2.0'
ds.attrs['author'] = 'Weilin Yang (weilinyang.yang@monash.edu)'

# Coordinates
ds.coords['dim'] = ('dim', np.arange(7))
ds['dim'].attrs['description'] = '0-mean, 1-std, 2-median, 3-mad, 4-17%, 5-83%, 6-n'

ds.coords['aspect_dim'] = ('temp_dim', np.arange(22.5,360,45))

ds['AAR_aspect_global']  = (('temp_dim','dim'), np.zeros([8,7])*np.nan)
ds['AAR_aspect_north'] = (('temp_dim','dim'), np.zeros([8,7])*np.nan)
ds['AAR_aspect_south'] = (('temp_dim','dim'), np.zeros([8,7])*np.nan)

ds['THAR_aspect_global']  = (('temp_dim','dim'), np.zeros([8,7])*np.nan)
ds['THAR_aspect_north'] = (('temp_dim','dim'), np.zeros([8,7])*np.nan)
ds['THAR_aspect_south'] = (('temp_dim','dim'), np.zeros([8,7])*np.nan)

ds['AABR_aspect_global']  = (('temp_dim','dim'), np.zeros([8,7])*np.nan)
ds['AABR_aspect_north'] = (('temp_dim','dim'), np.zeros([8,7])*np.nan)
ds['AABR_aspect_south'] = (('temp_dim','dim'), np.zeros([8,7])*np.nan)

file_path = '/Users/wyan0065/Desktop/AAR-disequilibrium/AABR/data_and_code/data/'
output_ds_all = xr.open_dataset(file_path + 'results_all_mad_debriscalving.nc');

#%%
n = np.shape(ds.coords['aspect_dim'].values)[0]
for i in range(0, n):
    
    if i == 0:
        aspect = ds.coords['aspect_dim'].values[i]
        find = np.where(output_ds_all['Aspect'] < aspect)[0]
        aspect = ds.coords['aspect_dim'].values[-1]
        find = np.append(find, np.where(output_ds_all['Aspect'] >= aspect)[0])
        
        if len(find)!=0:
            ds['AAR_aspect_global'].values[i,:] = calc_stats(output_ds_all['compile_AAR'].values[find,0])
        
        aspect = ds.coords['aspect_dim'].values[i]
        find = np.where((output_ds_all['Aspect'] < aspect) & 
                        (output_ds_all['CenLat'] >= 0))[0]
        aspect = ds.coords['aspect_dim'].values[-1]
        find = np.append(find, np.where((output_ds_all['Aspect'] >= aspect) & 
                                        (output_ds_all['CenLat'] >= 0))[0])
        if len(find)!=0:
            ds['AAR_aspect_north'].values[i,:] = calc_stats(output_ds_all['compile_AAR'].values[find,0])
        
        aspect = ds.coords['aspect_dim'].values[i]
        find = np.where((output_ds_all['Aspect'] < aspect) &
                        (output_ds_all['CenLat'] < 0))[0]
        aspect = ds.coords['aspect_dim'].values[-1]
        find = np.append(find, np.where((output_ds_all['Aspect'] >= aspect) & 
                                        (output_ds_all['CenLat'] < 0))[0])
        if len(find)!=0:
            ds['AAR_aspect_south'].values[i,:] = calc_stats(output_ds_all['compile_AAR'].values[find,0])
        
    else:
        bounds = np.array([ds.coords['aspect_dim'].values[i-1], ds.coords['aspect_dim'].values[i]])
        find = np.where(((output_ds_all['Aspect'].values >= bounds[0]) & 
                         (output_ds_all['Aspect'].values < bounds[1])))[0]
        if len(find)!=0:
            ds['AAR_aspect_global'].values[i,:] = calc_stats(output_ds_all['compile_AAR'].values[find,0])
        find = np.where((output_ds_all['Aspect'].values >= bounds[0]) & 
                         (output_ds_all['Aspect'].values < bounds[1]) & 
                         (output_ds_all['CenLat'] >= 0))[0]
        if len(find)!=0:
            ds['AAR_aspect_north'].values[i,:] = calc_stats(output_ds_all['compile_AAR'].values[find,0])

        find = np.where((output_ds_all['Aspect'].values >= bounds[0]) & 
                         (output_ds_all['Aspect'].values < bounds[1]) & 
                         (output_ds_all['CenLat'] < 0))[0]
        if len(find)!=0:
            ds['AAR_aspect_south'].values[i,:] = calc_stats(output_ds_all['compile_AAR'].values[find,0])

#%%
n = np.shape(ds.coords['aspect_dim'].values)[0]
for i in range(0, n):
    
    if i == 0:
        aspect = ds.coords['aspect_dim'].values[i]
        find = np.where(output_ds_all['Aspect'] < aspect)[0]
        aspect = ds.coords['aspect_dim'].values[-1]
        find = np.append(find, np.where(output_ds_all['Aspect'] >= aspect)[0])
        
        if len(find)!=0:
            ds['THAR_aspect_global'].values[i,:] = calc_stats(output_ds_all['compile_THAR'].values[find,0])
        
        aspect = ds.coords['aspect_dim'].values[i]
        find = np.where((output_ds_all['Aspect'] < aspect) & 
                        (output_ds_all['CenLat'] >= 0))[0]
        aspect = ds.coords['aspect_dim'].values[-1]
        find = np.append(find, np.where((output_ds_all['Aspect'] >= aspect) & 
                                        (output_ds_all['CenLat'] >= 0))[0])
        if len(find)!=0:
            ds['THAR_aspect_north'].values[i,:] = calc_stats(output_ds_all['compile_THAR'].values[find,0])
        
        aspect = ds.coords['aspect_dim'].values[i]
        find = np.where((output_ds_all['Aspect'] < aspect) &
                        (output_ds_all['CenLat'] < 0))[0]
        aspect = ds.coords['aspect_dim'].values[-1]
        find = np.append(find, np.where((output_ds_all['Aspect'] >= aspect) & 
                                        (output_ds_all['CenLat'] < 0))[0])
        if len(find)!=0:
            ds['THAR_aspect_south'].values[i,:] = calc_stats(output_ds_all['compile_THAR'].values[find,0])
        
    else:
        bounds = np.array([ds.coords['aspect_dim'].values[i-1], ds.coords['aspect_dim'].values[i]])
        find = np.where(((output_ds_all['Aspect'].values >= bounds[0]) & 
                         (output_ds_all['Aspect'].values < bounds[1])))[0]
        if len(find)!=0:
            ds['THAR_aspect_global'].values[i,:] = calc_stats(output_ds_all['compile_THAR'].values[find,0])
        find = np.where((output_ds_all['Aspect'].values >= bounds[0]) & 
                         (output_ds_all['Aspect'].values < bounds[1]) & 
                         (output_ds_all['CenLat'] >= 0))[0]
        if len(find)!=0:
            ds['THAR_aspect_north'].values[i,:] = calc_stats(output_ds_all['compile_THAR'].values[find,0])

        find = np.where((output_ds_all['Aspect'].values >= bounds[0]) & 
                         (output_ds_all['Aspect'].values < bounds[1]) & 
                         (output_ds_all['CenLat'] < 0))[0]
        if len(find)!=0:
            ds['THAR_aspect_south'].values[i,:] = calc_stats(output_ds_all['compile_THAR'].values[find,0])            

#%%
n = np.shape(ds.coords['aspect_dim'].values)[0]
for i in range(0, n):
    
    if i == 0:
        aspect = ds.coords['aspect_dim'].values[i]
        find = np.where(output_ds_all['Aspect'] < aspect)[0]
        aspect = ds.coords['aspect_dim'].values[-1]
        find = np.append(find, np.where(output_ds_all['Aspect'] >= aspect)[0])
        
        if len(find)!=0:
            ds['AABR_aspect_global'].values[i,:] = calc_stats(output_ds_all['compile_AABR'].values[find,0])
        
        aspect = ds.coords['aspect_dim'].values[i]
        find = np.where((output_ds_all['Aspect'] < aspect) & 
                        (output_ds_all['CenLat'] >= 0))[0]
        aspect = ds.coords['aspect_dim'].values[-1]
        find = np.append(find, np.where((output_ds_all['Aspect'] >= aspect) & 
                                        (output_ds_all['CenLat'] >= 0))[0])
        if len(find)!=0:
            ds['AABR_aspect_north'].values[i,:] = calc_stats(output_ds_all['compile_AABR'].values[find,0])
        
        aspect = ds.coords['aspect_dim'].values[i]
        find = np.where((output_ds_all['Aspect'] < aspect) &
                        (output_ds_all['CenLat'] < 0))[0]
        aspect = ds.coords['aspect_dim'].values[-1]
        find = np.append(find, np.where((output_ds_all['Aspect'] >= aspect) & 
                                        (output_ds_all['CenLat'] < 0))[0])
        if len(find)!=0:
            ds['AABR_aspect_south'].values[i,:] = calc_stats(output_ds_all['compile_AABR'].values[find,0])
        
    else:
        bounds = np.array([ds.coords['aspect_dim'].values[i-1], ds.coords['aspect_dim'].values[i]])
        find = np.where(((output_ds_all['Aspect'].values >= bounds[0]) & 
                         (output_ds_all['Aspect'].values < bounds[1])))[0]
        if len(find)!=0:
            ds['AABR_aspect_global'].values[i,:] = calc_stats(output_ds_all['compile_AABR'].values[find,0])
        find = np.where((output_ds_all['Aspect'].values >= bounds[0]) & 
                         (output_ds_all['Aspect'].values < bounds[1]) & 
                         (output_ds_all['CenLat'] >= 0))[0]
        if len(find)!=0:
            ds['AABR_aspect_north'].values[i,:] = calc_stats(output_ds_all['compile_AABR'].values[find,0])

        find = np.where((output_ds_all['Aspect'].values >= bounds[0]) & 
                         (output_ds_all['Aspect'].values < bounds[1]) & 
                         (output_ds_all['CenLat'] < 0))[0]
        if len(find)!=0:
            ds['AABR_aspect_south'].values[i,:] = calc_stats(output_ds_all['compile_AABR'].values[find,0])

#%%            
path = '/Users/wyan0065/Desktop/AAR-disequilibrium/AABR/data_and_code/data/results_figure_S7.nc';
enc_var = {'dtype': 'float32'}
encoding = {v: enc_var for v in ds.data_vars}
ds.to_netcdf(path, encoding=encoding);
output_ds_all.close()
ds.close()