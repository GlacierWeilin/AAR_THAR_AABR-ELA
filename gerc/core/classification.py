#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
Created on Tue Jan 20 20:07:10 2026

@author: Weilin Yang (weilinyang.yang@monash.edu)
'''

import os
import pandas as pd

BASE_DIR = os.path.dirname(os.path.dirname(__file__))
CLASS_DIR = os.path.join(BASE_DIR, 'data', 'classificationtree')


def run_classification(is_debris, is_icecap, is_tidewater,
                       temp, prcp, area, slope, aspect):
    glacier_type_code = is_debris + 2 * is_icecap + 4 * is_tidewater

    inputs = {
        'temperature_category': temp,
        'precipitation_category': prcp,
        'area_category': area,
        'slope_category': slope,
        'aspect_category': aspect
    }

    include_vars = [k for k, v in inputs.items() if v != -1]
    exclude_vars = [k for k, v in inputs.items() if v == -1]

    def try_file(level, used_vars):
        if exclude_vars and len(exclude_vars) < 5:
            excl = '_'.join(exclude_vars)
            used = '_'.join(used_vars)
            name = f'Classification_exclude_{excl}_Level{level}_glacier_type'
            if used:
                name += f'_{used}'
        else:
            used = '_'.join(used_vars)
            name = f'Classification_all_Level{level}_glacier_type'
            if used:
                name += f'_{used}'

        path = os.path.join(CLASS_DIR, name + '.csv')
        if not os.path.isfile(path):
            return None, None

        df = pd.read_csv(path, encoding='utf-8-sig')
        df = df[df['glacier_type'] == glacier_type_code]

        for col in used_vars:
            if col in df.columns:
                df = df[df[col] == inputs[col]]

        df = df[df['AAR_n'] >= 5]

        if df.empty:
            return None, None

        return df.iloc[0].copy(), level

    matched_row = None
    matched_level = None

    if include_vars:
        for level in range(len(include_vars) + 1, 0, -1):
            used_vars = include_vars[:max(0, level - 1)]
            matched_row, matched_level = try_file(level, used_vars)
            if matched_row is not None:
                break
    else:
        matched_row, matched_level = try_file(1, [])

    if matched_row is None:
        raise RuntimeError('No matching classification found.')

    label_map = {
        'temperature_category': {0: 'Low', 1: 'Moderate', 2: 'High'},
        'precipitation_category': {0: 'Low', 1: 'Moderate', 2: 'High'},
        'area_category': {0: 'Low', 1: 'Moderate', 2: 'High'},
        'slope_category': {0: 'Low', 1: 'Moderate', 2: 'High'},
        'aspect_category': {0: 'NS', 1: 'NN', 2: 'SS', 3: 'SN'},
        'is_debris': {0: 'No', 1: 'Yes'},
        'is_icecap': {0: 'No', 1: 'Yes'},
        'is_tidewater': {0: 'No', 1: 'Yes'}
    }

    base_cols = ['classification_level', 'is_debris', 'is_icecap', 'is_tidewater'] + include_vars + exclude_vars

    value_map = {
        'classification_level': matched_level,
        'is_debris': label_map['is_debris'][is_debris],
        'is_icecap': label_map['is_icecap'][is_icecap],
        'is_tidewater': label_map['is_tidewater'][is_tidewater]
    }

    for k in inputs:
        value_map[k] = label_map.get(k, {}).get(inputs[k], 'Not Applicable')

    header = pd.DataFrame([base_cols])
    row_df = pd.DataFrame([[value_map[col] for col in base_cols]])

    for prefix in ['AAR_', 'THAR_', 'AABR_']:
        cols = [c for c in matched_row.index if c.startswith(prefix)]
        header = pd.concat([header, pd.DataFrame([cols])], axis=1)
        row_df = pd.concat([row_df, pd.DataFrame([matched_row[cols].values])], axis=1)

    final_df = pd.concat([header, row_df], axis=0)

    return final_df
