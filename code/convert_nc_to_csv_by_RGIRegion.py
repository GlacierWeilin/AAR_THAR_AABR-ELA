#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
Created on Mon Mar 23 21:46:26 2026

@author: Weilin Yang (weilinyang.yang@monash.edu; ywlcwc@gmail.com)
'''

import xarray as xr
import pandas as pd
import os

path = '/Users/wyan0065/Desktop/AAR-disequilibrium/AABR/data_and_code/data/'
output_dir = path + 'results_all_mad_debriscalving_csv'
os.makedirs(output_dir, exist_ok=True)

ds = xr.open_dataset(path + 'results_all_mad_debriscalving.nc')

o1 = ds['O1Region'].values

rgiid = ds['RGIId'].values
aabr = ds['compile_AABR'][:, 0].values
aar  = ds['compile_AAR'][:, 0].values
ela  = ds['compile_ELA'][:, 0].values
thar = ds['compile_THAR'][:, 0].values

df = pd.DataFrame({
    'O1Region': o1,
    'RGIId': rgiid,
    'compile_AABR': aabr,
    'compile_AAR': aar,
    'compile_ELA': ela,
    'compile_THAR': thar
})


df['O1Region'] = df['O1Region'].astype(int)

for region, group in df.groupby('O1Region'):
    output_path = os.path.join(output_dir, f'O1Region_{region}.csv')
    group.to_csv(output_path, index=False)