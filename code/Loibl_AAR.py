#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
Created on Mon Sep 29 22:01:21 2025

@author: wyan0065
'''

import pandas as pd
import numpy as np
import scipy.stats as st

# period: 1995–2014 is selected to be consistent with the AAR calculation method used in PyGEM.
# Calcluate steady-state AAR by linear regression
def cal_AAR(AAR=None, MB=None):
    
    x=[]
    y=[]
    
    for i in range(1995,2015):
        year = str(i)
        if np.isnan(AAR[year]) == False and np.isnan(MB[year]) == False:
            x = np.append(x, MB[year])
            y = np.append(y, AAR[year])
                    
    if len(x) >= 5:
        slope, intercept, r_value, p_value, mad_err = st.linregress(x, y);
        if intercept < 0.05 or intercept > 0.95:
            result = pd.Series(np.zeros(7) * np.nan,
                               index=['n', 'slope', 'intercept', 'r_value', 'p_value', 'mad_err', 'AAR_mean'])
        else:
            result = pd.Series([len(x), slope, intercept, r_value, p_value, mad_err, np.nanmean(y)],
                               index=['n', 'slope', 'intercept', 'r_value', 'p_value', 'mad_err', 'AAR_mean'])
    else:
        result = pd.Series(np.zeros(7) * np.nan,
                           index=['n', 'slope', 'intercept', 'r_value', 'p_value', 'mad_err', 'AAR_mean'])
    
    return result

#%%
# Dussaillant, I., Hugonnet, R., Huss, M., Berthier, E., Bannwart, J., Paul, F., and Zemp, M.: 
# Annual mass change of the world's glaciers from 1976 to 2024 by temporal downscaling of satellite data with in situ observations, 
# Earth Syst. Sci. Data, 17, 1977–2006, https://doi.org/10.5194/essd-17-1977-2025, 2025.

# ASC_gla_MEAN-CAL-mass-change-series_obs_unobs.csv: 13
# ASW_gla_MEAN-CAL-mass-change-series_obs_unobs.csv: 14
# ASE_gla_MEAN-CAL-mass-change-series_obs_unobs.csv: 15

AARs = pd.read_csv('/Users/wyan0065/Desktop/AAR-disequilibrium/AABR/data_and_code/data/AAR_1995_2014_Loibl.csv')
rgi_ids = AARs['RGIId'].values

path = '/Users/wyan0065/Desktop/AAR-disequilibrium/AABR/obsdata/wgms-amce-2025-02b/individual-glacier/'
asc = pd.read_csv(path + 'ASC_gla_MEAN-CAL-mass-change-series_obs_unobs.csv')
asw = pd.read_csv(path + 'ASW_gla_MEAN-CAL-mass-change-series_obs_unobs.csv')
ase = pd.read_csv(path + 'ASE_gla_MEAN-CAL-mass-change-series_obs_unobs.csv')

df_all = pd.concat([asc, asw, ase], ignore_index=True)

years = [str(y) for y in range(1995, 2015)]
columns_to_keep = ['RGIId'] + [c for c in df_all.columns if c in years]
df_all = df_all[columns_to_keep]

df_all = df_all[df_all['RGIId'].isin(rgi_ids)]

df_all.to_csv('/Users/wyan0065/Desktop/AAR-disequilibrium/AABR/data_and_code/data/MB_1995_2014_Dussaillant.csv', index=False)

#%%
param = pd.DataFrame()
for i in range(0, len(AARs)):
    rgi_id = AARs.iloc[i]['RGIId']
    AAR = AARs.iloc[i]
    AAR = AAR.iloc[1:]
    
    MB = df_all[df_all['RGIId'] == rgi_id]
    MB = MB.iloc[0, 1:]
    
    result = cal_AAR(AAR=AAR, MB=MB)
    result = pd.DataFrame(result).T
    result.insert(0, 'RGIId', rgi_id)
    
    param = pd.concat([param, result])

param.set_index('RGIId', inplace=True)
param = param.dropna(how='all')   

param.to_csv('/Users/wyan0065/Desktop/AAR-disequilibrium/AABR/data_and_code/data/AAR0_Loibl_Dussaillant.csv', index=True)

#%% ===== Compare simulation results with observations =====
import pandas as pd
import xarray as xr

filepath = '/Users/wyan0065/Desktop/AAR-disequilibrium/AABR/data_and_code/data/';
loibl  = pd.read_csv(filepath + 'Loibl_AAR-ELA-AABR.csv')
RGIId = (loibl['RGIId']).tolist()
n = len(RGIId)

fn = 'results_all_mad_debriscalving.nc';
output_ds_all = xr.open_dataset(filepath + fn)
find_id = np.where(np.isin(output_ds_all['RGIId'].values, RGIId)==True);
loibl['CenLon']       = output_ds_all['CenLon'].values[find_id]
loibl['CenLat']       = output_ds_all['CenLat'].values[find_id]

loibl['compile_AAR']   = (output_ds_all['compile_AAR'].values[find_id,0]).reshape(n)
loibl['intercept_AAR'] = (output_ds_all['intercept_AAR'].values[find_id,0]).reshape(n)
loibl['steady_AAR']    = (output_ds_all['steady_AAR'].values[find_id,0]).reshape(n)

loibl['compile_ELA']   = (output_ds_all['compile_ELA'].values[find_id,0]).reshape(n)
loibl['intercept_ELA'] = (output_ds_all['intercept_ELA'].values[find_id,0]).reshape(n)
loibl['steady_ELA']    = (output_ds_all['steady_ELA'].values[find_id,0]).reshape(n)

loibl['compile_THAR']   = (output_ds_all['compile_THAR'].values[find_id,0]).reshape(n)
loibl['intercept_THAR'] = (output_ds_all['intercept_THAR'].values[find_id,0]).reshape(n)
loibl['steady_THAR']    = (output_ds_all['steady_THAR'].values[find_id,0]).reshape(n)

loibl['compile_AABR']   = (output_ds_all['compile_AABR'].values[find_id,0]).reshape(n)
loibl['intercept_AABR'] = (output_ds_all['intercept_AABR'].values[find_id,0]).reshape(n)
loibl['steady_AABR']    = (output_ds_all['steady_AABR'].values[find_id,0]).reshape(n)

output_ds_all.close()

loibl.to_csv(filepath + '/Loibl_comparison.csv', index=False)

#%%
import pandas as pd
import numpy as np
import xarray as xr

filepath = '/Users/wyan0065/Desktop/AAR-disequilibrium/AABR/data_and_code/data/';
AAR  = pd.read_csv(filepath + 'AAR0_Loibl_Dussaillant.csv') 
RGIId = (AAR['RGIId']).tolist()
n = len(RGIId)

fn = 'results_all_mad_debriscalving.nc';
output_ds_all = xr.open_dataset(filepath + fn)
find_id = np.where(np.isin(output_ds_all['RGIId'].values, RGIId)==True);
area = output_ds_all['Area'].values[find_id]

#area_weighted_aar = np.nansum(area * AAR['AAR_mean']) / np.nansum(area);
#area_weighted_aar = np.nansum(area * AAR['intercept']) / np.nansum(area);
area_weighted_aar = np.nansum(area * (output_ds_all['compile_AAR'].values[find_id,0]).reshape(n)) / np.nansum(area);

#%%





