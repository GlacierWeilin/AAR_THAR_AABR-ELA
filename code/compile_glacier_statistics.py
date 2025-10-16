#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr  1 15:25:49 2025

@author: Weilin Yang (weilinyang.yang@monash.edu)
"""
import os
import csv
import calendar
import numpy as np
import pandas as pd
import xarray as xr

# elev_bands
url = '/Users/wyan0065/Desktop/AAR-disequilibrium/AABR/data_and_code/data/ori/'
csv_files = [f for f in os.listdir(url) if f.endswith('.csv')]
csv_files_sorted = sorted(csv_files, key=lambda x: int(x.split('_')[0]))
glacier = []
for i in range(0, 19):
    filename = url + csv_files_sorted[i]
    df = pd.read_csv(filename)
    glacier.append(df)
glacier = pd.concat(glacier, axis=0, ignore_index=True)
glacier = glacier.set_index('RGIId');

glacier = glacier[['CenLon', 'CenLat', 'O1Region', 'O2Region', 'Area', 'Slope','Aspect', 'Zmin', 'Zmax', 'Zmed', 'Lmax']]

#%%
url = '/Users/wyan0065/Desktop/AAR-disequilibrium/AABR/data_and_code/data/summary/'
csv_files = [f for f in os.listdir(url) if f.endswith('.csv')]
csv_files_sorted = sorted(csv_files, key=lambda x: int(x[19:21]))
gdf_sel = []
for i in range(0, 19):
    fpath = url + csv_files_sorted[i]
    with open(fpath, 'r') as csvfile:
        reader = csv.DictReader(csvfile)
        column = [row['rgi_id'] for row in reader]
    gdf_sel = np.append(gdf_sel, column)

glacier = glacier.loc[gdf_sel]
#%%
# temp, prcp, lapserate,wind speed (10m)
ds = xr.open_dataset('/Users/wyan0065/Desktop/PyGEM/calving/climate_data/ERA5/ERA5_temp_monthly.nc')
lat = ds['latitude'].values
lon = ds['longitude'].values
temp = ds['t2m'].values
temp = np.nanmean(temp.reshape(85,12,721,1440), axis=1)
temp = temp[55:74,:,:]
temp = np.nanmean(temp,axis=0)-273.15
ds.close()

ds = xr.open_dataset('/Users/wyan0065/Desktop/PyGEM/calving/climate_data/ERA5/ERA5_totalprecip_monthly.nc')

prcp = ds['tp'].values
[time,m,n] = np.shape(prcp)
mday = np.zeros([time,1])
k=0
for year in range(1940, 2024):
    for month in range(1,13):
        monthRange = calendar.monthrange(year, month)
        mday[k] = monthRange[1]
        k += 1

mday = np.tile(mday,m*n)
mday = mday.reshape(time,m,n)
prcp = prcp*mday
prcp = np.nansum(prcp.reshape(85,12,721,1440), axis=1)
prcp = prcp[55:74,:,:]
prcp = np.nanmean(prcp,axis=0)
ds.close()

ds = xr.open_dataset('/Users/wyan0065/Desktop/PyGEM/calving/climate_data/ERA5/ERA5_lapserates_monthly.nc')
lapserate = ds['lapserate'].values
lapserate = np.nanmean(lapserate.reshape(85,12,721,1440), axis=1)
lapserate = lapserate[55:74,:,:]
lapserate = np.nanmean(lapserate,axis=0)
ds.close()

ds = xr.open_dataset('/Users/wyan0065/Desktop/PyGEM/calving/climate_data/ERA5/ERA5_geopotential.nc')
ele = ds['z'].values
ele = np.nanmean(ele.reshape(85,12,721,1440), axis=1)
ele = ele[55:74,:,:]
ele = np.nanmean(ele,axis=0) / 9.80665
ds.close()

ds = xr.open_dataset('/Users/wyan0065/Desktop/PyGEM/calving/climate_data/ERA5/ERA5_10mwindspeed_monthly_1995_2014.nc')
si10 = ds['si10'].values
si10 = np.nanmean(si10.reshape(20,12,721,1440), axis=1)
si10 = np.nanmean(si10,axis=0) # m/s
ds.close()

# %%
CenLons = np.where(glacier['CenLon'] < 0, glacier['CenLon'] + 360, glacier['CenLon'])

grid_positions = []
for CenLat, CenLon in zip(glacier['CenLat'], CenLons):
    lat_idx = np.digitize(CenLat, lat) - 1
    lon_idx = np.digitize(CenLon, lon) - 1
    grid_positions.append((lat_idx, lon_idx))

grid_positions = np.array(grid_positions)
glacier['temp']      = temp[grid_positions[:,0], grid_positions[:,1]]
glacier['prcp']      = prcp[grid_positions[:,0], grid_positions[:,1]]
glacier['lapserate'] = lapserate[grid_positions[:,0], grid_positions[:,1]]
glacier['ele']       = ele[grid_positions[:,0], grid_positions[:,1]]
glacier['si10']       = si10[grid_positions[:,0], grid_positions[:,1]]

file_path = '/Users/wyan0065/Desktop/AAR-disequilibrium/AABR/data_and_code/data/'
ds = xr.Dataset.from_dataframe(glacier)
ds.to_netcdf(file_path + 'glacier_statistics.nc')