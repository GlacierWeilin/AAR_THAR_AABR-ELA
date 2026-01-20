#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 20 20:07:10 2026

@author: wyan0065
"""

import os
import pandas as pd

BASE_DIR = os.path.dirname(os.path.dirname(__file__))
CLASS_DIR = os.path.join(BASE_DIR, "data", "classificationtree")

def run_classification(
    is_debris, is_icecap, is_tidewater,
    temp, prcp, area, slope, aspect
):
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
            excl = "_".join(exclude_vars)
            used = "_".join(used_vars)
            name = f"Classification_exclude_{excl}_Level{level}_glacier_type"
            if used:
                name += f"_{used}"
        else:
            used = "_".join(used_vars)
            name = f"Classification_all_Level{level}_glacier_type"
            if used:
                name += f"_{used}"

        path = os.path.join(CLASS_DIR, name + ".csv")
        if not os.path.isfile(path):
            return None, None

        df = pd.read_csv(path)
        df = df[df["glacier_type"] == glacier_type_code]

        for col in used_vars:
            df = df[df[col] == inputs[col]]

        df = df[df["AAR_n"] >= 5]
        if df.empty:
            return None, None

        return df.iloc[0], level

    matched = None
    level_used = None

    for level in range(len(include_vars) + 1, 0, -1):
        used = include_vars[:max(0, level - 1)]
        matched, level_used = try_file(level, used)
        if matched is not None:
            break

    if matched is None:
        raise RuntimeError("No matching classification found.")

    result = pd.DataFrame([matched])
    return result, level_used
