#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 30 16:47:13 2024

@author: Weilin Yang (weilinyang.yang@monash.edu)
"""

import pandas as pd
import numpy as np
import xarray as xr
import os
import glob
import collections
from scipy.interpolate import griddata

csv_files = glob.glob('/Users/wyan0065/Desktop/AAR-disequilibrium/AABR/data_and_code/data/ori/*.csv')
csv_files = np.sort(csv_files)
dfs = [pd.read_csv(file) for file in csv_files]
RGI_all = pd.concat(dfs, ignore_index=True)
m = len(RGI_all);

#%% Output
glac_values = np.arange(m);

dim = np.arange(2)

# Variable coordinates dictionary
output_coords_dict = collections.OrderedDict()
output_coords_dict['RGIId']        = collections.OrderedDict([('glac', glac_values)])
output_coords_dict['CenLon']       = collections.OrderedDict([('glac', glac_values)])
output_coords_dict['CenLat']       = collections.OrderedDict([('glac', glac_values)])
output_coords_dict['O1Region']     = collections.OrderedDict([('glac', glac_values)])
output_coords_dict['O2Region']     = collections.OrderedDict([('glac', glac_values)])
output_coords_dict['is_tidewater'] = collections.OrderedDict([('glac', glac_values)])
output_coords_dict['is_icecap']    = collections.OrderedDict([('glac', glac_values)])
output_coords_dict['is_debris']    = collections.OrderedDict([('glac', glac_values)])

output_coords_dict['glac_prec']            = collections.OrderedDict([('glac', glac_values),('dim', dim)])
output_coords_dict['glac_temp']            = collections.OrderedDict([('glac', glac_values),('dim', dim)])
output_coords_dict['glac_acc']             = collections.OrderedDict([('glac', glac_values),('dim', dim)])
output_coords_dict['glac_refreeze']        = collections.OrderedDict([('glac', glac_values),('dim', dim)])
output_coords_dict['glac_melt']            = collections.OrderedDict([('glac', glac_values),('dim', dim)])
output_coords_dict['glac_frontalablation'] = collections.OrderedDict([('glac', glac_values)])
output_coords_dict['glac_massbaltotal']    = collections.OrderedDict([('glac', glac_values),('dim', dim)])

output_coords_dict['intercept_AAR'] = collections.OrderedDict([('glac', glac_values),('dim', dim)])
output_coords_dict['steady_AAR']    = collections.OrderedDict([('glac', glac_values),('dim', dim)])
output_coords_dict['compile_AAR']   = collections.OrderedDict([('glac', glac_values),('dim', dim)])

# Attributes dictionary
output_attrs_dict = {
        'glac': {
                'long_name': 'glacier index',
                 'comment': 'glacier index referring to glaciers properties and model results'},
        'dim': {
                'long_name': 'stats for a given variable',
                'comment': '0-median, 1-mad'},
        'RGIId': {
                'long_name': 'Randolph Glacier Inventory ID',
                'comment': 'RGIv6.0'},
        'CenLon': {
                'long_name': 'center longitude',
                'units': 'degrees E',
                'comment': 'value from RGIv6.0'},
        'CenLat': {
                'long_name': 'center latitude',
                'units': 'degrees N',
                'comment': 'value from RGIv6.0'},
        'O1Region': {
                'long_name': 'RGI order 1 region',
                'comment': 'value from RGIv6.0'},
        'O2Region': {
                'long_name': 'RGI order 2 region',
                'comment': 'value from RGIv6.0'},
        'is_tidewater': {
                'long_name': 'is marine-terminating glacier',
                'comment': 'value from RGIv6.0'},
        'is_icecap': {
                'long_name': 'is ice cap',
                'comment': 'value from RGIv6.0'},
        'is_debris': {
                'long_name': 'is debris covered glacier',
                'comment': 'value from Rounce et al. (2021)'},
        'glac_temp': {
                'standard_name': 'air_temperature',
                'long_name': 'glacier-wide mean air temperature',
                'units': '$^\circ$C',
                'comment': ('each elevation bin is weighted equally to compute the mean temperature, and '
                            'bins where the glacier no longer exists due to retreat have been removed')},
        'glac_prec': {
                'long_name': 'glacier-wide precipitation (liquid)',
                'units': 'm3',
                'comment': 'only the liquid precipitation, solid precipitation excluded'},
        'glac_acc': {
                'long_name': 'glacier-wide accumulation, in water equivalent',
                'units': 'm3',
                'comment': 'only the solid precipitation'},
        'glac_refreeze': {
                'long_name': 'glacier-wide refreeze, in water equivalent',
                'units': 'm3'},
        'glac_melt': {
                'long_name': 'glacier-wide melt, in water equivalent',
                'units': 'm3'},
        'glac_frontalablation': {
                'long_name': 'observed frontalablation, in water equivalent',
                'units': 'm3'},
        'glac_massbaltotal': {
                'long_name': 'glacier-wide total mass balance, in water equivalent',
                'units': 'm3',
                'comment': 'total mass balance is the sum of the climatic mass balance and frontal ablation'},
        'intercept_AAR': {
                'long_name': 'glacier steady-state Accumulation Area Ratio based on linear regression'},
        'steady_AAR': {
                'long_name': 'glacier steady-state Accumulation Area Ratio based on steady state assumption'},
        'compile_AAR': {
                'long_name': 'glacier steady-state Accumulation Area Ratio'},
        }

count_vn = 0
encoding = {}
for vn in output_coords_dict.keys():
    count_vn += 1
    empty_holder = np.zeros([len(output_coords_dict[vn][i]) for i in list(output_coords_dict[vn].keys())])
    output_ds = xr.Dataset({vn: (list(output_coords_dict[vn].keys()), empty_holder)},
                               coords=output_coords_dict[vn])
    # Merge datasets of stats into one output
    if count_vn == 1:
        output_ds_all = output_ds
    else:
        output_ds_all = xr.merge((output_ds_all, output_ds))
noencoding_vn = ['RGIId']
# Add attributes
for vn in output_ds_all.variables:
    try:
        output_ds_all[vn].attrs = output_attrs_dict[vn]
    except:
        pass
    # Encoding (specify _FillValue, offsets, etc.)
       
    if vn not in noencoding_vn:
        encoding[vn] = {'_FillValue': None,
                        'zlib':True,
                        'complevel':9
                        }
            
output_ds_all.attrs = {'Source' : 'PyGEMv0.2.5 developed by David Rounce (drounce@alaska.edu)',
                       'Further developed by': 'Weilin Yang (weilinyang.yang@monash.edu)'}

#%% all
output_ds_all['RGIId'].values        = RGI_all['RGIId'].values;
output_ds_all['CenLon'].values       = RGI_all['CenLon'].values;
output_ds_all['CenLat'].values       = RGI_all['CenLat'].values;
output_ds_all['O1Region'].values     = RGI_all['O1Region'].values;
output_ds_all['O2Region'].values     = RGI_all['O2Region'].values;
output_ds_all['is_tidewater'].values = RGI_all['IsTidewater'].values;
output_ds_all['is_icecap'].values    = np.where(RGI_all['GlacierType']=='Glacier', 0, 1);

output_ds_all['glac_prec'].values            = np.zeros([m,2])*np.nan
output_ds_all['glac_temp'].values            = np.zeros([m,2])*np.nan
output_ds_all['glac_acc'].values             = np.zeros([m,2])*np.nan
output_ds_all['glac_refreeze'].values        = np.zeros([m,2])*np.nan
output_ds_all['glac_melt'].values            = np.zeros([m,2])*np.nan
output_ds_all['glac_frontalablation'].values = np.zeros(m)*np.nan
output_ds_all['glac_massbaltotal'].values    = np.zeros([m,2])*np.nan

output_ds_all['intercept_AAR'].values          = np.zeros([m,2])*np.nan

output_ds_all['steady_AAR'].values        = np.zeros([m,2])*np.nan

output_ds_all['compile_AAR'].values        = np.zeros([m,2])*np.nan

#%% simulated glaciers
#data = xr.open_dataset('/Users/wyan0065/Desktop/AAR-disequilibrium/AABR/data_and_code/data/results_AAR_debriscalving_withnan.nc');
#data = xr.open_dataset('/Users/wyan0065/Desktop/AAR-disequilibrium/AABR/data_and_code/data/results_AAR_control_withnan.nc');
data = xr.open_dataset('/Users/wyan0065/Desktop/AAR-disequilibrium/AABR/data_and_code/data/results_AAR_debrisonly_withnan.nc');

find = np.where(RGI_all['RGIId'].isin(data['RGIId'].values)==1)[0]

output_ds_all['is_debris'].values[find] = data['is_debris'].values
output_ds_all['is_tidewater'].values[find] = data['is_tidewater'].values

output_ds_all['glac_prec'].values[find,:]            = data['glac_prec'].values[:,[0,1]]
output_ds_all['glac_temp'].values[find,:]            = data['glac_temp'].values[:,[0,1]]
output_ds_all['glac_acc'].values[find,:]             = data['glac_acc'].values[:,[0,1]]
output_ds_all['glac_refreeze'].values[find,:]        = data['glac_refreeze'].values[:,[0,1]]
output_ds_all['glac_melt'].values[find,:]            = data['glac_melt'].values[:,[0,1]]
output_ds_all['glac_frontalablation'].values[find]   = data['glac_frontalablation'].values
output_ds_all['glac_massbaltotal'].values[find,:]    = data['glac_massbaltotal'].values[:,[0,1]]

output_ds_all['intercept_AAR'].values[find,:]     = data['intercept_AAR'].values[:,[0,1]]
output_ds_all['steady_AAR'].values[find,:]        = data['steady_AAR'].values[:,[0,1]]

#%%
# is_debris
loc = np.where(np.isnan(output_ds_all['is_debris'].values))[0]
RGIId = output_ds_all['RGIId'].values[loc]
for i,rgi_id in enumerate(RGIId):
    if int(rgi_id[6:8])<10:
        debris_ed = '/Users/wyan0065/Desktop/PyGEM/calving/debris_data/ed_tifs/'+rgi_id[6:8]+'/'+rgi_id[7:14]+'_meltfactor.tif';
        debris_hd = '/Users/wyan0065/Desktop/PyGEM/calving/debris_data/hd_tifs/'+rgi_id[6:8]+'/'+rgi_id[7:14]+'_hdts_m.tif';
    else:
        debris_ed = '/Users/wyan0065/Desktop/PyGEM/calving/debris_data/ed_tifs/'+rgi_id[6:8]+'/'+rgi_id[6:14]+'_meltfactor.tif';
        debris_hd = '/Users/wyan0065/Desktop/PyGEM/calving/debris_data/hd_tifs/'+rgi_id[6:8]+'/'+rgi_id[6:14]+'_hdts_m.tif';
    
    output_ds_all['is_debris'].values[loc[i]] = os.path.exists(debris_hd);

# intercept_AAR
loc = np.where(np.isnan(output_ds_all['intercept_AAR'].values[:,0]))[0]
loc_lon = output_ds_all['CenLon'].values[loc]
loc_lat = output_ds_all['CenLat'].values[loc]

find_loc = np.where(~np.isnan(output_ds_all['intercept_AAR'].values[:,0]))[0]
find_lon = output_ds_all['CenLon'].values[find_loc]
find_lat = output_ds_all['CenLat'].values[find_loc]

output_ds_all['intercept_AAR'].values[loc,:] = griddata((find_lon, find_lat),
                                                        output_ds_all['intercept_AAR'].values[find_loc,:], 
                                                        (loc_lon, loc_lat), method='nearest')

# steady_AAR
loc = np.where(np.isnan(output_ds_all['steady_AAR'].values[:,0]))[0]
loc_lon = output_ds_all['CenLon'].values[loc]
loc_lat = output_ds_all['CenLat'].values[loc]

find_loc = np.where(~np.isnan(output_ds_all['steady_AAR'].values[:,0]))[0]
find_lon = output_ds_all['CenLon'].values[find_loc]
find_lat = output_ds_all['CenLat'].values[find_loc]

output_ds_all['steady_AAR'].values[loc,:] = griddata((find_lon, find_lat),
                                                     output_ds_all['steady_AAR'].values[find_loc,:], 
                                                     (loc_lon, loc_lat), method='nearest')


output_ds_all['compile_AAR'].values[:,:]        = np.column_stack(((np.median(np.column_stack((
    output_ds_all['intercept_AAR'].values[:,0], output_ds_all['steady_AAR'].values[:,0])),axis=1)),
    (np.median(np.column_stack((
        output_ds_all['intercept_AAR'].values[:,1], output_ds_all['steady_AAR'].values[:,1])), axis=1))))

#output_ds_all.to_netcdf('/Users/wyan0065/Desktop/AAR-disequilibrium/AABR/data_and_code/data/results_AAR_debriscalving.nc');
#output_ds_all.to_netcdf('/Users/wyan0065/Desktop/AAR-disequilibrium/AABR/data_and_code/data/results_AAR_control.nc');
output_ds_all.to_netcdf('/Users/wyan0065/Desktop/AAR-disequilibrium/AABR/data_and_code/data/results_AAR_debrisonly.nc');
# Close datasets
output_ds_all.close();

