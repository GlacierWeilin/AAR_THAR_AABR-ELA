#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 30 21:38:59 2025

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
        stats = np.append(stats, np.nanmedian(data)) # 'median'
        stats = np.append(stats, np.nanpercentile(data, 83)) # '83%'
        stats = np.append(stats, np.nanpercentile(data, 95)) # '95%'
        stats = np.append(stats, median_abs_deviation(data, nan_policy='omit')) # Compute the median absolute deviation of the data
    
        data  = output[:,1]; data = data*data;
        stats = np.append(stats, np.sqrt(np.nansum(data))/np.sum(~np.isnan(data)))

    elif output.ndim == 1:
        data = output[:];
        stats = None
        stats = np.nanmean(data) # 'mean'
        stats = np.append(stats, np.nanstd(data)) # 'std'
        stats = np.append(stats, np.nanpercentile(data, 5)) # '5%'
        stats = np.append(stats, np.nanpercentile(data, 17)) # '17%'
        stats = np.append(stats, np.nanmedian(data)) # 'median'
        stats = np.append(stats, np.nanpercentile(data, 83)) # '83%'
        stats = np.append(stats, np.nanpercentile(data, 95)) # '95%'
        stats = np.append(stats, median_abs_deviation(data, nan_policy='omit')) # Compute the median absolute deviation of the data
    
        stats = np.append(stats, np.nan)
        
    return stats

def regional_avg(n=None, data=None, path=''):
    """Calculate the regional AAR, THAR, AABR with the spatial resolution of 0.25°*0.25°

    Parameters
    ----------
    n : float
        Spatial resolution. The default is 0.25 (ERA5).
    ela: array
        ERA5_MCMC__ba1_100sets_2000_2019_all.nc
    filesuffix : str
        add suffix to output file.
    path : str, bool
        Set to "True" in order  to store the info in the working directory
        Set to a path to store the file to your chosen location (file
        extension matters)
    """
    
    latbnd = np.arange(-90-n/2, 90+n, n)
    lat    = int(180/n+1)
    lon    = int(360/n)
    if n==0.25:
        lonbnd = np.arange(0-n/2, 360+n/2, n)
    else:
        lonbnd = np.arange(-180, 180+n/2, n)
    
    # prepare output
    ds = xr.Dataset();
    
    # Attributes
    ds.attrs['description'] = 'OGGM & PyGEM output'
    ds.attrs['version'] = 'OGGM1.6.2 & PyGEM0.2.0'
    ds.attrs['author'] = 'Weilin Yang (weilinyang.yang@monash.edu)'
    
    # Coordinates
    ds.coords['dim'] = ('dim', np.arange(9))
    ds['dim'].attrs['description'] = '0-mean, 1-std, 2-5%, 3-17%, 4-median, 5-83%, 6-95%, 7-mad, 8-mean_std(std)'
    
    ds.coords['latitude'] = np.arange(-90, 90+n, n)
    ds['latitude'].attrs['long_name'] = 'latitude'
    ds['latitude'].attrs['units'] = 'degrees_north'
    if n==0.25:
        ds.coords['longitude'] = np.arange(0, 360, n)
        ds['longitude'].attrs['long_name'] = 'longitude'
        ds['longitude'].attrs['units'] = 'degrees_east'
    else:
        ds.coords['longitude'] = np.arange(-180+n/2, 180, n)
        ds['longitude'].attrs['long_name'] = 'longitude'
        ds['longitude'].attrs['units'] = 'degrees_east'
    
    ds['ELA_compile']      = (('latitude', 'longitude','dim'), np.zeros([lat,lon,9])*np.nan)
    ds['ELA_intercept']    = (('latitude', 'longitude','dim'), np.zeros([lat,lon,9])*np.nan)
    ds['ELA_steady']       = (('latitude', 'longitude','dim'), np.zeros([lat,lon,9])*np.nan)
    
    ds['AABR_compile']      = (('latitude', 'longitude','dim'), np.zeros([lat,lon,9])*np.nan)
    ds['AABR_intercept']    = (('latitude', 'longitude','dim'), np.zeros([lat,lon,9])*np.nan)
    ds['AABR_steady']       = (('latitude', 'longitude','dim'), np.zeros([lat,lon,9])*np.nan)
    
    ds['AAR_compile']      = (('latitude', 'longitude','dim'), np.zeros([lat,lon,9])*np.nan)
    ds['AAR_intercept']    = (('latitude', 'longitude','dim'), np.zeros([lat,lon,9])*np.nan)
    ds['AAR_steady']       = (('latitude', 'longitude','dim'), np.zeros([lat,lon,9])*np.nan)
    
    ds['THAR_compile']      = (('latitude', 'longitude','dim'), np.zeros([lat,lon,9])*np.nan)
    ds['THAR_intercept']    = (('latitude', 'longitude','dim'), np.zeros([lat,lon,9])*np.nan)
    ds['THAR_steady']       = (('latitude', 'longitude','dim'), np.zeros([lat,lon,9])*np.nan)
    
    for i in range(0, lat):
        bottom = latbnd[i]; up = latbnd[i+1]
        for j in range(0, lon):
            left = lonbnd[j]; right = lonbnd[j+1];
            find_id = np.where((data['CenLat']>=bottom) & (data['CenLat']<up) & (data['CenLon']>=left) & (data['CenLon']<right))[0];
            
            if sum(find_id)!=0:
                ds['ELA_compile'].values[i,j,:]      = calc_stats(data['compile_ELA'].values[find_id,:]);
                ds['ELA_intercept'].values[i,j,:]    = calc_stats(data['intercept_ELA'].values[find_id,:]);
                ds['ELA_steady'].values[i,j,:]       = calc_stats(data['steady_ELA'].values[find_id,:]);
                
                ds['AAR_compile'].values[i,j,:]      = calc_stats(data['compile_AAR'].values[find_id,:]);
                ds['AAR_intercept'].values[i,j,:]    = calc_stats(data['intercept_AAR'].values[find_id,:]);
                ds['AAR_steady'].values[i,j,:]       = calc_stats(data['steady_AAR'].values[find_id,:]);
                
                ds['THAR_compile'].values[i,j,:]      = calc_stats(data['compile_THAR'].values[find_id,:]);
                ds['THAR_intercept'].values[i,j,:]    = calc_stats(data['intercept_THAR'].values[find_id,:]);
                ds['THAR_steady'].values[i,j,:]       = calc_stats(data['steady_THAR'].values[find_id,:]);
                
                ds['AABR_compile'].values[i,j,:]      = calc_stats(data['compile_AABR'].values[find_id,:]);
                ds['AABR_intercept'].values[i,j,:]    = calc_stats(data['intercept_AABR'].values[find_id,:]);
                ds['AABR_steady'].values[i,j,:]       = calc_stats(data['steady_AABR'].values[find_id,:]);
                
                
    # To file
    path = path + 'results_' + str(n) + '.nc';
    enc_var = {'dtype': 'float32'}
    encoding = {v: enc_var for v in ds.data_vars}
    ds.to_netcdf(path, encoding=encoding);                

#%% ===== Main codes =====

path = '/Users/wyan0065/Desktop/AAR-disequilibrium/AABR/data_and_code/data/';
fn = 'results_all_mad_debriscalving.nc';
data = xr.open_dataset(path+fn);
data.close()

n=0.5;
if n==0.25:
    find_id = np.where(data['CenLon'].values<-n/2);
    data['CenLon'].values[find_id] = data['CenLon'].values[find_id] + 360;
else:
    pass

regional_avg(n=n, data=data, path=path)

