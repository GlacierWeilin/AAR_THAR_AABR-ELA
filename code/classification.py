#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 14 14:37:48 2025
@author: Weilin Yang (weilinyang.yang@monash.edu)
"""

import numpy as np
import xarray as xr
import pandas as pd
import os
from itertools import combinations

# Remove outliers
def remove_outliers_iqr(values):
    q1 = np.nanpercentile(values[:,0], 25)
    q3 = np.nanpercentile(values[:,0], 75)
    iqr = q3 - q1
    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr
    loc = np.where((values[:,0] < lower) | (values[:,0] > upper))[0]
    values[loc,:] = np.nan
    return values

# calculate uncertainty
def cal_uncertainty(x):
    return np.sqrt(np.nansum(x * x))/np.sum(~np.isnan(x))

# Classification tree
def classification_tree(values, label, exclude=[]):
    valid = ~np.isnan(values[:,0])
    
    df = pd.DataFrame({
        'glacier_type': glacier_type[valid],
        'temperature_category': temp_labels[valid],
        'precipitation_category': prcp_labels[valid],
        'area_category': area_labels[valid],
        'slope_category': slope_labels[valid],
        'aspect_category': aspect_category[valid],
        'value': values[valid,0],
        'uncertainty': values[valid,1]
    })

    base_levels = [
        'glacier_type',
        'temperature_category',
        'precipitation_category',
        'area_category',
        'slope_category',
        'aspect_category'
    ]
    included_levels = [l for l in base_levels if l not in exclude]

    levels = []
    for i in range(1, len(included_levels) + 1):
        levels.append(included_levels[:i])

    all_summaries = []
    for i, level in enumerate(levels, 1):
        grouped = df.groupby(level)
        summary = grouped.agg({
            'value': ['count', 'mean', 'std'],
            'uncertainty': cal_uncertainty
        }).reset_index()
        summary.columns = level + [f'{label}_n', f'{label}_mean', f'{label}_std', f'{label}_1σ']
        all_summaries.append((i, level, summary))
    return all_summaries



file_path = '/Users/wyan0065/Desktop/AAR-disequilibrium/AABR/data_and_code/data/'
out_path = file_path + 'classificationtree/'
os.makedirs(out_path, exist_ok=True)

data = xr.open_dataset(file_path + 'results_all_mad_debriscalving.nc')
bins = xr.open_dataset(out_path + 'results_equal_count_bins3.nc')

# Glacier type encoding
is_debris   = data['is_debris'].values
is_icecap   = data['is_icecap'].values
is_tidwater = data['is_tidewater'].values
glacier_type = (
    is_debris.astype(int) * 1 +
    is_icecap.astype(int) * 2 +
    is_tidwater.astype(int) * 4
)

area   = data['Area'].values
slope  = data['Slope'].values
temp   = data['temp'].values
prcp   = data['prcp'].values
aspect = data['Aspect'].values
lat    = data['CenLat'].values
aspect_cos = np.cos(np.deg2rad(aspect))

# Bin labels
area_bins  = bins['area_dim_bin_edges'].values[1:-1]
slope_bins = bins['slope_dim_bin_edges'].values[1:-1]
temp_bins  = bins['temp_dim_bin_edges'].values[1:-1]
prcp_bins  = bins['prcp_dim_bin_edges'].values[1:-1]

area_labels  = np.digitize(area, area_bins, right=False)
slope_labels = np.digitize(slope, slope_bins, right=False)
temp_labels  = np.digitize(temp, temp_bins, right=False)
prcp_labels  = np.digitize(prcp, prcp_bins, right=False)

aspect_category = np.full_like(aspect_cos, fill_value=np.nan, dtype=int)
north_hemisphere = lat >= 0
south_hemisphere = ~north_hemisphere
aspect_category[(north_hemisphere) & (aspect_cos < 0)] = 0
aspect_category[(north_hemisphere) & (aspect_cos >= 0)]  = 1
aspect_category[(south_hemisphere) & (aspect_cos < 0)] = 2
aspect_category[(south_hemisphere) & (aspect_cos >= 0)]  = 3


aar_clean  = remove_outliers_iqr(data['compile_AAR'].values)
thar_clean = remove_outliers_iqr(data['compile_THAR'].values)
aabr_clean = remove_outliers_iqr(data['compile_AABR'].values)


all_vars = [
    'temperature_category',
    'precipitation_category',
    'area_category',
    'slope_category',
    'aspect_category'
]

exclusion_combinations = []
for r in range(0, len(all_vars)):
    exclusion_combinations += list(combinations(all_vars, r))

exclusion_combinations.append(tuple(all_vars))

for exclude in exclusion_combinations:
    exclude_list = list(exclude)
    aar_levels  = classification_tree(aar_clean,  'AAR',  exclude=exclude_list)
    thar_levels = classification_tree(thar_clean, 'THAR', exclude=exclude_list)
    aabr_levels = classification_tree(aabr_clean, 'AABR', exclude=exclude_list)

    for i, level, aar_df in aar_levels:
        _, _, thar_df = thar_levels[i - 1]
        _, _, aabr_df = aabr_levels[i - 1]

        merged = aar_df.merge(thar_df, on=level).merge(aabr_df, on=level)

        exclude_tag = 'exclude_' + '_'.join(exclude_list) if exclude_list else 'all'
        level_tag = '_'.join(level)
        output_file = f'Classification_{exclude_tag}_Level{i}_{level_tag}.csv'
        merged.to_csv(os.path.join(out_path, output_file), encoding='utf-8-sig', index=False)
