#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 30 11:23:15 2025

@author: Weilin Yang (weilinyang.yang@monash.edu)
"""

import shutil
import os
import numpy as np
import pandas as pd
import logging

from oggm import cfg, workflow, entity_task, global_task
from oggm.workflow import execute_entity_task

log = logging.getLogger(__name__)
@entity_task(log)
def compute_THAR_AABR(gdir, AAR_all):
    """Compute the glacier ELA, THAR and AABR based on given AAR
    Parameters
    ----------
    gdir: oggm.GlacierDirectory
        the glacier directory to process
    ELA_all: DataFrame
    """
    
    rgi_id = gdir.rgi_id
    loc = np.where(AAR_all['RGIId'].values == rgi_id)[0]
    
    path = gdir.get_filepath('inversion_flowlines')
    # WGMS
    wgms_aar = AAR_all['intercept'].values[loc]
    wgms_aar = wgms_aar[0]
    if os.path.exists(path):
        fls = gdir.read_pickle('inversion_flowlines')
        fl = fls[0]
        nbin_area    = fl.widths_m * fl.dx_meter
        nbin_surface = fl.surface_h
        
        tot_area = np.nansum(nbin_area)
        Aac = tot_area * wgms_aar;
        acc_bin_area = np.cumsum(nbin_area)
        acc_loc = np.argmin(np.abs(acc_bin_area - Aac))
        ELA = nbin_surface[acc_loc]

        THAR = (ELA- nbin_surface[-1])/(nbin_surface[0]-nbin_surface[-1])
            
        # update Aac
        Aac = np.sum(nbin_area[:acc_loc+1])
        AAR = Aac / tot_area
            
        # AABR
        zac = np.nansum(nbin_area[:acc_loc+1] * nbin_surface[:acc_loc+1])
            
        if acc_loc!=np.shape(nbin_surface)[0]-1:
            zab = np.nansum(nbin_area[acc_loc+1:] * nbin_surface[acc_loc+1:])
            AABR = (zac) / (zab)
        else:
            AABR = np.nan
                
    odf = pd.Series(data=[ELA, THAR, AAR, AABR],
                    index=['Loibl_ELA', 'Loibl_THAR', 'Loibl_AAR', 'Loibl_AABR'])
    
    return odf

@global_task(log)
def compile_THAR_AABR(gdirs, filesuffix='', path=True, csv=True, AAR_all=None):
    """Compiles a table of ELA, THAR, and AABR.

    The file is stored in a hdf file (not csv) per default. Use pd.read_hdf
    to open it.

    Parameters
    ----------
    gdirs : list of :py:class:`oggm.GlacierDirectory` objects
        the glacier directories to process
    filesuffix : str
        add suffix to output file
    path : str, bool
        Set to "True" in order  to store the info in the working directory
        Set to a path to store the file to your chosen location (file
        extension matters)
    csv: bool
        Set to store the data in csv instead of hdf.
    AAR_all: DataFrame
    """

    out_df = execute_entity_task(compute_THAR_AABR, gdirs, AAR_all=AAR_all);

    for idx, s in enumerate(out_df):
        if s is None:
            out_df[idx] = pd.Series(np.nan)

    out = pd.concat(out_df, axis=1, keys=[gd.rgi_id for gd in gdirs])
    out = out.dropna(axis=0, how='all')

    if path:
        if path is True:
            fpath = os.path.join(cfg.PATHS['working_dir'],
                                 'Loibl_AAR-ELA-AABR' + filesuffix)
            if csv:
                out.to_csv(fpath + '.csv')
            else:
                out.to_hdf(fpath + '.hdf', key='df')
        else:
            ext = os.path.splitext(path)[-1]
            if ext.lower() == '.csv':
                out.to_csv(path)
            elif ext.lower() == '.hdf':
                out.to_hdf(path, key='df')
    return out

for i in range(1,7):
    cfg.initialize(logging_level='WARNING')
    cfg.PARAMS['use_multiprocessing'] = True
    cfg.PARAMS['continue_on_error'] = True
    cfg.PARAMS['dl_verify'] = False
    cfg.PARAMS['has_internet'] = False
    cfg.PATHS['dl_cache_dir'] = '/g/data/rd53/wy2165/OGGM/download_cache/'
    cfg.PATHS['working_dir'] = '/scratch/rd53/wy2165/AABR_Loibl/'
    
    filepath = '/g/data/rd53/wy2165/ELA_ratios/';
    obs  = pd.read_csv(filepath + 'AAR0_Loibl_Dussaillant.csv')
    gdf_sel = (obs['RGIId'].values).tolist()
    AAR_all = obs[['RGIId','intercept']]
    
    base_url = 'https://cluster.klima.uni-bremen.de/~oggm/gdirs/oggm_v1.6/L3-L5_files/2023.3/elev_bands/W5E5/'
    if i!=6:
        gdirs = workflow.init_glacier_directories(gdf_sel[5000*(i-1):5000*i], from_prepro_level=3, prepro_border=80, prepro_base_url=base_url)
    else:
        gdirs = workflow.init_glacier_directories(gdf_sel[5000*(i-1):], from_prepro_level=3, prepro_border=80, prepro_base_url=base_url)
        
    compile_THAR_AABR(gdirs, filesuffix='_' + str(i), path=True, csv=True, AAR_all=AAR_all);
    
    folder_path = cfg.PATHS['working_dir'] + 'per_glacier'
    shutil.rmtree(folder_path)