#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 15 14:17:57 2024

@author: Weilin Yang (weilinyang.yang@monash.edu)
"""

import pandas as pd
import numpy as np
import scipy.stats as st

# period: 1995–2014 is selected to be consistent with the AAR calculation method used in PyGEM.
# Calcluate steady-state AAR by linear regression
def cal_AAR(AAR=None, smb=None):
    
    smb = smb.set_index('YEAR')
    AAR = AAR.set_index('YEAR')
    
    find_id = np.where(smb['LOWER_BOUND'] == 9999)
    smb = smb.iloc[find_id]
    
    selected_years = np.arange(1995, 2015, 1)
    smb = smb.loc[smb.index.isin(selected_years)]
    AAR = AAR.loc[AAR.index.isin(selected_years)]
    
    year = AAR.index
    x=[]
    y=[]
    for t in year:
        if t in smb.index:
            if np.isnan(smb['ANNUAL_BALANCE'][t]) == False and np.isnan(AAR['AAR'][t]) == False \
                and AAR['AAR'][t] >=5 and AAR['AAR'][t] <= 95:
                    x = np.append(x, smb['ANNUAL_BALANCE'][t])
                    y = np.append(y, AAR['AAR'][t])
                    
    if len(x) >= 5:
        slope, intercept, r_value, p_value, mad_err = st.linregress(x, y);
        intercept = intercept/100;
        if intercept < 0.05 or intercept > 0.95:
            intercept = np.nan;
        else:
            result = pd.Series([len(x), slope, intercept, r_value, p_value, mad_err, np.nanmean(y)/100, np.nanmean(x)],
                               index=['n', 'slope', 'intercept', 'r_value', 'p_value', 'mad_err', 'wgms_AAR_mean', 'wgms_SMB_mean'])
    else:
        result = pd.Series(np.zeros(8) * np.nan,
                           index=['n', 'slope', 'intercept', 'r_value', 'p_value', 'mad_err', 'wgms_AAR_mean', 'wgms_SMB_mean'])
    
    return result

filepath = '/Users/wyan0065/Desktop/AAR-disequilibrium/AABR/data_and_code/data/DOI-WGMS-FoG-2024-01/data/';
wgms_id  = pd.read_csv(filepath + 'WGMS_ID_AAR.csv')
RGIId    = pd.read_csv(filepath + 'glacier_id_lut.csv')
latlon   = pd.read_csv(filepath + 'glacier.csv');
_AAR     = pd.read_csv(filepath + 'mass_balance_overview.csv')
_smb     = pd.read_csv(filepath + 'mass_balance.csv')
_area    = pd.read_csv(filepath + 'state.csv')

RGIId    = RGIId.set_index('WGMS_ID')
latlon   = latlon.set_index('WGMS_ID')
_AAR     = _AAR.set_index('WGMS_ID')
_smb     = _smb.set_index('WGMS_ID')
_area    = _area.set_index('WGMS_ID')

param = pd.DataFrame()
for i in range(0, len(wgms_id)):
    n = wgms_id['WGMS_ID'][i]
    AAR = _AAR.loc[n]
    if n in _smb.index:
        smb = _smb.loc[n]
        if type(AAR['YEAR']) is not np.int64 and type(smb['YEAR']) is not np.int64:
            result = cal_AAR(AAR=AAR, smb=smb)
            result = pd.DataFrame(result, columns=[n]).T
            
            result.insert(0, 'area_2020', _area.loc[(_area.index==n)&(_area['YEAR']==2020),'AREA'])
            
            result.insert(0, 'lon', [latlon.loc[n]['LONGITUDE']])
            result.insert(0, 'lat', [latlon.loc[n]['LATITUDE']])
            if n in RGIId.index:
                result.insert(0, 'RGIId', [RGIId.loc[n]['RGI60_ID']])
            else:
                result.insert(0, 'RGIId', ['NaN'])
            
        else:
            result = pd.Series(np.zeros(12) * np.nan,
                               index=['RGIId', 'lat', 'lon', 'area_2020',
                                      'n', 'slope', 'intercept', 'r_value', 'p_value', 'mad_err', 'wgms_AAR_mean', 'wgms_SMB_mean'])
            result = pd.DataFrame(result, columns=[n]).T
    else:
        result = pd.Series(np.zeros(12) * np.nan,
                           index=['RGIId', 'lat', 'lon', 'area_2020',
                                  'n', 'slope', 'intercept', 'r_value', 'p_value', 'mad_err', 'wgms_AAR_mean', 'wgms_SMB_mean'])
        result = pd.DataFrame(result, columns=[n]).T
    
    param = pd.concat([param, result])

param.to_csv('/Users/wyan0065/Desktop/AAR-disequilibrium/AABR/data_and_code/data/WGMS_AAR_all.csv')

#%% ===== Compare simulation results with observations =====
import pandas as pd
import numpy as np
import xarray as xr

filepath = '/Users/wyan0065/Desktop/AAR-disequilibrium/AABR/data_and_code/data/';
wgms  = pd.read_csv(filepath + 'WGMS_AAR-ELA-AABR.csv').iloc[:, 1:]
wgms  = wgms.T
wgms.index.name = 'RGIId'
wgms.columns = ['wgms_ELA', 'wgms_THAR', 'wgms_AAR', 'wgms_AABR']
RGIId = wgms.index
n = len(RGIId)

fn = 'results_all_mad_debriscalving.nc';
output_ds_all = xr.open_dataset(filepath + fn)
find_id = np.where(np.isin(output_ds_all['RGIId'].values, RGIId)==True);
wgms['CenLon']       = output_ds_all['CenLon'].values[find_id]
wgms['CenLat']       = output_ds_all['CenLat'].values[find_id]

wgms['compile_AAR']   = (output_ds_all['compile_AAR'].values[find_id,0]).reshape(n)
wgms['intercept_AAR'] = (output_ds_all['intercept_AAR'].values[find_id,0]).reshape(n)
wgms['steady_AAR']    = (output_ds_all['steady_AAR'].values[find_id,0]).reshape(n)

wgms['compile_ELA']   = (output_ds_all['compile_ELA'].values[find_id,0]).reshape(n)
wgms['intercept_ELA'] = (output_ds_all['intercept_ELA'].values[find_id,0]).reshape(n)
wgms['steady_ELA']    = (output_ds_all['steady_ELA'].values[find_id,0]).reshape(n)

wgms['compile_THAR']   = (output_ds_all['compile_THAR'].values[find_id,0]).reshape(n)
wgms['intercept_THAR'] = (output_ds_all['intercept_THAR'].values[find_id,0]).reshape(n)
wgms['steady_THAR']    = (output_ds_all['steady_THAR'].values[find_id,0]).reshape(n)

wgms['compile_AABR']   = (output_ds_all['compile_AABR'].values[find_id,0]).reshape(n)
wgms['intercept_AABR'] = (output_ds_all['intercept_AABR'].values[find_id,0]).reshape(n)
wgms['steady_AABR']    = (output_ds_all['steady_AABR'].values[find_id,0]).reshape(n)

output_ds_all.close()

wgms.to_csv(filepath + '/WGMS_comparison.csv')



