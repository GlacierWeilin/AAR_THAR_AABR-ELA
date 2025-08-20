#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb  8 09:42:22 2024

Source : PyGEMv0.2.5 developed by David Rounce (drounce@alaska.edu)
Further developed by: Weilin Yang (weilinyang.yang@monash.edu)
Code reviewed by: Wenchao Chu (peterchuwenchao@foxmail.com)

"""
import numpy as np
import pandas as pd
import xarray as xr
import os
import collections

import pygem.pygem_input as pygem_prms
import pygem.pygem_modelsetup as modelsetup

#%% ====== Read RGI tables =====

regions = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19]

main_glac_rgi_all = pd.DataFrame();
for reg in regions:
    main_glac_rgi = modelsetup.selectglaciersrgitable(rgi_regionsO1=[reg], rgi_regionsO2='all', rgi_glac_number='all', 
                                                      glac_no=None, rgi_fp=pygem_prms.rgi_fp);
    main_glac_rgi_all = pd.concat([main_glac_rgi_all, main_glac_rgi], axis=0);

noresults = pd.read_csv(pygem_prms.output_filepath+'noresults.csv')
main_glac_rgi_all = main_glac_rgi_all[~main_glac_rgi_all['RGIId'].isin(noresults['RGIId'])]
[m,n]=np.shape(main_glac_rgi_all);

#%% ===== Create empty xarray dataset =====

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

output_coords_dict['intercept_AAR']     = collections.OrderedDict([('glac', glac_values),('dim', dim)])
output_coords_dict['steady_AAR']        = collections.OrderedDict([('glac', glac_values),('dim', dim)])

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
                       'Further developed by': 'Weilin Yang (weilinyang.yang@monash.edu)',
                       'Code reviewed by': 'Wenchao Chu (peterchuwenchao@foxmail.com)'}

#%% ===== Compile simulation results =====
filepath = pygem_prms.output_filepath + '/simulations/';
RGIId=[];

for i in range(0,m):
    rgi_id = main_glac_rgi_all.RGIId.values[i];
    stats_fp = filepath + rgi_id[6:8] + '/ERA5/stats/';
    
    if pygem_prms.include_debris:
        debris_fp = pygem_prms.debris_fp;
    else:
        debris_fp = pygem_prms.output_filepath + '/../debris_data/';
    
    if int(rgi_id[6:8]) <10:
        stats_fn = rgi_id[7:14] + '_ERA5_MCMC_ba1_100sets_1995_2014_all.nc';
        debris_fn = debris_fp + 'ed_tifs/' + rgi_id[6:8] + '/' + rgi_id[7:14] + '_meltfactor.tif';
    else:
        stats_fn = rgi_id[6:14] + '_ERA5_MCMC_ba1_100sets_1995_2014_all.nc';
        debris_fn = debris_fp + 'ed_tifs/' + rgi_id[6:8] + '/' + rgi_id[6:14] + '_meltfactor.tif';
    
    stats = xr.open_dataset(stats_fp+stats_fn);
    RGIId = np.append(RGIId, stats['RGIId'].values);

    output_ds_all['CenLon'].values[i]       = stats['CenLon'].values;
    output_ds_all['CenLat'].values[i]       = stats['CenLat'].values;
    output_ds_all['O1Region'].values[i]     = stats['O1Region'].values;
    output_ds_all['O2Region'].values[i]     = stats['O2Region'].values;
    output_ds_all['is_tidewater'].values[i] = stats['is_tidewater'].values;
    output_ds_all['is_icecap'].values[i]    = stats['is_icecap'].values;
    output_ds_all['is_debris'].values[i]    = os.path.exists(debris_fn);

    output_ds_all['glac_prec'].values[i,:]          = stats['glac_prec'].values;
    output_ds_all['glac_temp'].values[i,:]          = stats['glac_temp'].values;
    output_ds_all['glac_acc'].values[i,:]           = stats['glac_acc'].values;
    output_ds_all['glac_refreeze'].values[i,:]      = stats['glac_refreeze'].values;
    output_ds_all['glac_melt'].values[i,:]          = stats['glac_melt'].values;
    output_ds_all['glac_frontalablation'].values[i] = stats['glac_frontalablation'].values;
    output_ds_all['glac_massbaltotal'].values[i,:]  = stats['glac_massbaltotal'].values;
    
    output_ds_all['intercept_AAR'].values[i,:]     = stats['intercept_AAR'].values;
    output_ds_all['steady_AAR'].values[i,:]        = stats['steady_AAR'].values;

AAR = np.column_stack(((np.median(np.column_stack((
    output_ds_all['intercept_AAR'].values[:,0], output_ds_all['steady_AAR'].values[:,0])),axis=1)),
    (np.median(np.column_stack((
        output_ds_all['intercept_AAR'].values[:,1], output_ds_all['steady_AAR'].values[:,1])), axis=1))))

output_ds_all['compile_AAR'] = (output_ds_all['steady_AAR'].dims, AAR)
output_ds_all['compile_AAR'].attrs['long_name'] = 'glacier steady-state Accumulation Area Ratio'

output_ds_all['RGIId'].values = RGIId;

#%% ===== Export Results =====

output_fp = pygem_prms.output_filepath;
output_fn = ('results_AAR_debriscalving.nc');
output_ds_all.to_netcdf(output_fp+output_fn);
            
# Close datasets
output_ds_all.close();