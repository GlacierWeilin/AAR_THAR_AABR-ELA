#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
Created on Tue Jan 13 15:30:03 2026

@author: Weilin Yang (weilinyang.yang@monash.edu)
'''

import xarray as xr

file_path = '/Users/wyan0065/Desktop/AAR-disequilibrium/AABR/data_and_code/data/'
#exp = 'debrisonly'
#exp = 'control'
exp = 'debriscalving'
glacier = xr.open_dataset(file_path+'glacier_statistics.nc')
results = xr.open_dataset(file_path+f'results_all_mad_{exp}_0.nc')

u10 = glacier['u10']
v10 = glacier['v10']

results['u10'] = u10
results['v10'] = v10

results.to_netcdf(file_path+f'results_all_mad_{exp}.nc')