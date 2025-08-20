#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 10 12:16:35 2025

@author: Weilin Yang (weilinyang.yang@monash.edu)
"""

import shutil
import os
import csv
import numpy as np
import pandas as pd
import xarray as xr
import logging

import scipy.stats as st

from oggm import cfg, workflow, entity_task, global_task
from oggm.workflow import execute_entity_task

log = logging.getLogger(__name__)
@entity_task(log)
def compute_from_ELA(gdir, ela_all=None, mb_all=None):
    """Compute the glacier AAR and AABR based on annual ELA and mass balance.
    Parameters
    ----------
    gdir: oggm.GlacierDirectory
        the glacier directory to process
    ela_all: DataFrame
        annual ela of each glacier
    mb_all: DataFrame
        annual mb of each glacier
    """
    
    rgi_id = gdir.rgi_id
    _ela = ela_all[rgi_id]
    mb = mb_all[rgi_id]
    
    ELA  = np.nan
    AABR = np.nan
    AAR  = np.nan
    THAR = np.nan
    
    x=[]
    y=[]
    for t in mb.index:
        if np.isnan(_ela[t]) == False and np.isnan(mb[t]) == False:
            x = np.append(x, mb[t])
            y = np.append(y, _ela[t])
            
    if len(x) >= 5:
    
        slope, intercept, r_value, p_value, std_err = st.linregress(x, y);
        
        ELA = intercept;
        path = gdir.get_filepath('inversion_flowlines')
        if os.path.exists(path):
            fls = gdir.read_pickle('inversion_flowlines')
            fl = fls[0]
            nbin_area    = fl.widths_m * fl.dx_meter
            nbin_surface = fl.surface_h
            
            tot_area = np.nansum(nbin_area)
            acc_loc = np.argmin(np.abs(nbin_surface - ELA))
            ELA = nbin_surface[acc_loc]
            THAR = (ELA - nbin_surface[-1]) / (nbin_surface[0] - nbin_surface[-1])
            
            Aac = np.sum(nbin_area[:acc_loc+1])
            AAR = Aac / tot_area
            
            # AABR
            zac = np.nansum(nbin_area[:acc_loc+1] * nbin_surface[:acc_loc+1])
            
            if acc_loc!=np.shape(nbin_surface)[0]-1:
                zab = np.nansum(nbin_area[acc_loc+1:] * nbin_surface[acc_loc+1:])
                AABR = (zac) / (zab)
            else:
                AABR = np.nan
        
            if AAR < 0.05 or AAR> 0.95 or np.isnan(AABR):
                ELA  = np.nan
                AAR  = np.nan
                AABR = np.nan
        else:
            AABR = np.nan
            AAR  = np.nan
    else:
        ELA  = np.nan
        AABR = np.nan
        AAR  = np.nan
                
    odf = pd.Series(data=[ELA, THAR, AAR, AABR], index=['ELA','THAR', 'AAR', 'AABR'])
    
    return odf

@entity_task(log)
def compute_from_AAR(gdir, AAR=None):
    """Compute the glacier ELA based on given AAR.
    Parameters
    ----------
    gdir: oggm.GlacierDirectory
        the glacier directory to process
    AAR: float
        give AAR
    """
    
    path = gdir.get_filepath('inversion_flowlines')
    ELA = np.nan
    if os.path.exists(path):
        fls = gdir.read_pickle('inversion_flowlines')
        fl = fls[0]
        nbin_area    = fl.widths_m * fl.dx_meter
        nbin_surface = fl.surface_h
        
        n = len(nbin_surface);
        AAR_all = np.zeros(n);
        tot_area = np.nansum(nbin_area)
        for i in range(0,n):
            AAR_all[i] = np.sum(nbin_area[:i])/tot_area;
        
        closest_index = np.argmin(np.abs(AAR_all - AAR));
        
        ELA = nbin_surface[closest_index]
                
    odf = pd.Series(data=[ELA], index=['ELA'])
    
    return odf

@entity_task(log)
def compute_from_AABR(gdir, AABR=None):
    """Compute the glacier ELA based on given AABR.
    Parameters
    ----------
    gdir: oggm.GlacierDirectory
        the glacier directory to process
    AABR: float
        give AABR
    """
    
    ELA = np.nan
    path = gdir.get_filepath('inversion_flowlines')
    if os.path.exists(path):
        fls = gdir.read_pickle('inversion_flowlines')
        fl = fls[0]
        nbin_area    = fl.widths_m * fl.dx_meter
        nbin_surface = fl.surface_h
        
        n = len(nbin_surface);
        AABR_all = np.zeros(n);
        for i in range(0,n):
            
            zac = np.sum(nbin_area[:i] * nbin_surface[:i])
            zab = np.sum(nbin_area[i:] * nbin_surface[i:])
            
            AABR_all[i] = (zac) / (zab);
        
        closest_index = np.argmin(np.abs(AABR_all - AABR));
        
        ELA = nbin_surface[closest_index]
                
    odf = pd.Series(data=[ELA], index=['ELA'])
    
    return odf

@entity_task(log)
def compute_from_THAR(gdir, THAR=None):
    """Compute the glacier ELA based on given THAR.
    Parameters
    ----------
    gdir: oggm.GlacierDirectory
        the glacier directory to process
    THAR: float
        give THAR
    """
    
    ELA = np.nan
    path = gdir.get_filepath('inversion_flowlines')
    if os.path.exists(path):
        fls = gdir.read_pickle('inversion_flowlines')
        fl = fls[0]
        nbin_surface = fl.surface_h
        
        n = len(nbin_surface);
        THAR_all = np.zeros(n);
        for i in range(0,n):
            
            THAR_all[i] = (nbin_surface[i] - nbin_surface[-1]) / (nbin_surface[0] - nbin_surface[-1]);
        
        closest_index = np.argmin(np.abs(THAR_all - THAR));
        
        ELA = nbin_surface[closest_index]
                
    odf = pd.Series(data=[ELA], index=['ELA'])
    
    return odf

log = logging.getLogger(__name__)
@entity_task(log)
def compute_AABR_from_AAR(gdir, AAR_all=None, name=None):
    """Compute the glacier ELA and AABR based on given AAR.
    AAR varies from glaciers.
    Parameters
    ----------
    gdir: oggm.GlacierDirectory
        the glacier directory to process
    AAR_all: xarray
             Accumulation-area ratio
    """
    
    # 0-median,1-min,2-max
    ELA  = np.zeros(3) * np.nan
    AAR  = np.zeros(3) * np.nan
    AABR = np.zeros(3) * np.nan
    THAR = np.zeros(3) * np.nan
    
    rgi_id = gdir.rgi_id
    loc = np.where(AAR_all['RGIId'].values == rgi_id)[0]
    _AAR = AAR_all[name].values[loc,:]
    _AAR = _AAR[0]
    # 0-median, 1-mad
    AAR[0] = _AAR[0]
    AAR[1] = AAR[0] - _AAR[1]
    AAR[2] = AAR[0] + _AAR[1]
    
    path = gdir.get_filepath('inversion_flowlines')
    if os.path.exists(path):
        fls = gdir.read_pickle('inversion_flowlines')
        fl = fls[0]
        nbin_area    = fl.widths_m * fl.dx_meter
        nbin_surface = fl.surface_h
        
        tot_area = np.nansum(nbin_area)
        for i in range(3):
            Aac = tot_area * AAR[i];
            acc_bin_area = np.cumsum(nbin_area)
            acc_loc = np.argmin(np.abs(acc_bin_area - Aac))
            
            ELA[i] = nbin_surface[acc_loc]
            THAR[i] = (ELA[i]- nbin_surface[-1])/(nbin_surface[0]-nbin_surface[-1])
            
            # update Aac
            Aac = np.sum(nbin_area[:acc_loc+1])
            AAR[i] = Aac / tot_area
            
            zac = np.nansum(nbin_area[:acc_loc+1] * nbin_surface[:acc_loc+1])
            
            if acc_loc!=np.shape(nbin_surface)[0]-1:
                zab = np.nansum(nbin_area[acc_loc+1:] * nbin_surface[acc_loc+1:])
                AABR[i] = (zac) / (zab)
            else:
                AABR[i] = np.nan
                ELA[i]  = np.nan
                AAR[i]  = np.nan
                THAR[i]  = np.nan
                
    odf = pd.Series(data=[ELA[0], ELA[1], ELA[2], AAR[0], AAR[1], AAR[2], AABR[0], AABR[1], AABR[2], \
                    THAR[0], THAR[1], THAR[2]],
                    index=['ELA_median', 'ELA_min', 'ELA_max', 'AAR_median', 'AAR_min', 'AAR_max', 
                           'AABR_medain', 'AABR_min', 'AABR_max', 'THAR_median', 'THAR_min', 'THAR_max'])
    
    return odf

@entity_task(log)
def compute_temp(gdir, temp_all=None, lapserate_all=None,ele_all=None):
    """Compute the glacier-wide temperature based on the annual average temperature at reference height
        (2014~2023)
    Parameters
    ----------
    gdir: oggm.GlacierDirectory
        the glacier directory to process
    temp_all: xarray
    """
    
    rgi_id = gdir.rgi_id
    temp      = temp_all.sel(RGIId=rgi_id).values
    lapserate = lapserate_all.sel(RGIId=rgi_id).values
    ele       = ele_all.sel(RGIId=rgi_id).values
    path = gdir.get_filepath('inversion_flowlines')
    if os.path.exists(path):
        fls = gdir.read_pickle('inversion_flowlines')
        fl = fls[0]
        nbin_area    = fl.widths_m * fl.dx_meter
        nbin_surface = fl.surface_h
        
        nbin_temp = temp + (nbin_surface - ele) * lapserate
        weighted_temp = np.average(nbin_temp, weights=nbin_area)
    else:
        weighted_temp = np.nan
                
    odf = pd.Series(data=[weighted_temp], index=['glacier_temp'])
    
    return odf

@global_task(log)
def compile_from_ELA(gdirs, filesuffix='', path=True, csv=True, ela_all=None, mb_all=None):
    """Compiles a table of ELA, AAR and AABR.

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
    ela_all: DataFrame
        annual ela of each glacier
    mb_all: DataFrame
        annual mb of each glacier
    """

    out_df = execute_entity_task(compute_from_ELA, gdirs, ela_all=ela_all, mb_all=mb_all)

    for idx, s in enumerate(out_df):
        if s is None:
            out_df[idx] = pd.Series(np.nan)

    out = pd.concat(out_df, axis=1, keys=[gd.rgi_id for gd in gdirs])
    out = out.dropna(axis=0, how='all')

    if path:
        if path is True:
            fpath = os.path.join(cfg.PATHS['working_dir'],
                                 'ELA-AAR-AABR' + filesuffix)
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

@global_task(log)
def compile_from_AAR(gdirs, filesuffix='', path=True, csv=True, AAR=None):
    """Compiles a table of ELA.

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
    AAR: float
        give AAR
    """

    out_df = execute_entity_task(compute_from_AAR, gdirs, AAR=AAR)

    for idx, s in enumerate(out_df):
        if s is None:
            out_df[idx] = pd.Series(np.nan)

    out = pd.concat(out_df, axis=1, keys=[gd.rgi_id for gd in gdirs])
    out = out.dropna(axis=0, how='all')

    if path:
        if path is True:
            fpath = os.path.join(cfg.PATHS['working_dir'],
                                 'AAR-ELA' + str(AAR) + filesuffix)
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

@global_task(log)
def compile_from_AABR(gdirs, filesuffix='', path=True, csv=True, AABR=None):
    """Compiles a table of ELA.

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
    AABR: float
        give AABR
    """

    out_df = execute_entity_task(compute_from_AABR, gdirs, AABR=AABR)

    for idx, s in enumerate(out_df):
        if s is None:
            out_df[idx] = pd.Series(np.nan)

    out = pd.concat(out_df, axis=1, keys=[gd.rgi_id for gd in gdirs])
    out = out.dropna(axis=0, how='all')

    if path:
        if path is True:
            fpath = os.path.join(cfg.PATHS['working_dir'],
                                 'AABR-ELA' + str(AABR) + filesuffix)
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

@global_task(log)
def compile_from_THAR(gdirs, filesuffix='', path=True, csv=True, THAR=None):
    """Compiles a table of ELA.

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
    THAR: float
        give THAR
    """

    out_df = execute_entity_task(compute_from_THAR, gdirs, THAR=THAR)

    for idx, s in enumerate(out_df):
        if s is None:
            out_df[idx] = pd.Series(np.nan)

    out = pd.concat(out_df, axis=1, keys=[gd.rgi_id for gd in gdirs])
    out = out.dropna(axis=0, how='all')

    if path:
        if path is True:
            fpath = os.path.join(cfg.PATHS['working_dir'],
                                 'THAR-ELA' + str(THAR) + filesuffix)
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

@global_task(log)
def compile_AABR(gdirs, filesuffix='', path=True, csv=True, AAR_all=None, name='compile_AAR'):
    """Compiles a table of ELA, AAR, and AABR.

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
    AAB: xarray
        give AAR
    """

    out_df = execute_entity_task(compute_AABR_from_AAR, gdirs, AAR_all=AAR_all, name='compile_AAR')

    for idx, s in enumerate(out_df):
        if s is None:
            out_df[idx] = pd.Series(np.nan)

    out = pd.concat(out_df, axis=1, keys=[gd.rgi_id for gd in gdirs])
    out = out.dropna(axis=0, how='all')

    if path:
        if path is True:
            fpath = os.path.join(cfg.PATHS['working_dir'],
                                 'AAR-ELA-AABR' + filesuffix)
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

@global_task(log)
def compile_temp(gdirs, filesuffix='', path=True, csv=True, temp_all=None, lapserate_all=None,ele_all=None):
    """Compiles a table of glacier-wide temperature

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
    """

    out_df = execute_entity_task(compute_temp, gdirs, temp_all=temp_all, 
                                 lapserate_all=lapserate_all, ele_all=ele_all)

    for idx, s in enumerate(out_df):
        if s is None:
            out_df[idx] = pd.Series(np.nan)

    out = pd.concat(out_df, axis=1, keys=[gd.rgi_id for gd in gdirs])
    out = out.dropna(axis=0, how='all')

    if path:
        if path is True:
            fpath = os.path.join(cfg.PATHS['working_dir'],
                                 'glacier_temp' + filesuffix)
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

# elev_bands
url = '/scratch/rd53/wy2165/OGGM/download_cache/cluster.klima.uni-bremen.de/~oggm/gdirs/oggm_v1.6/L3-L5_files/2023.3/elev_bands/W5E5/RGI62/b_080/L3/summary/'
csv_files = [f for f in os.listdir(url) if f.endswith('.csv')]
csv_files_sorted = sorted(csv_files, key=lambda x: int(x[19:21]))
gdf_sel = []
for i in range(0, 19):
    fpath = url + csv_files_sorted[i]
    with open(fpath, 'r') as csvfile:
        reader = csv.DictReader(csvfile)
        column = [row['rgi_id'] for row in reader]
    gdf_sel = np.append(gdf_sel, column)

for i in range(1,45):
    
    cfg.initialize(logging_level='WARNING')
    cfg.PARAMS['use_multiprocessing'] = True
    cfg.PARAMS['continue_on_error'] = True
    cfg.PARAMS['dl_verify'] = False
    cfg.PARAMS['has_internet'] = False
    cfg.PATHS['dl_cache_dir'] = '/scratch/rd53/wy2165/OGGM/download_cache/'
    cfg.PATHS['working_dir'] = '/scratch/rd53/wy2165/AABR/'
    
    base_url = 'https://cluster.klima.uni-bremen.de/~oggm/gdirs/oggm_v1.6/L3-L5_files/2023.3/elev_bands/W5E5/'
    if i!=44:
        gdirs = workflow.init_glacier_directories(gdf_sel[5000*(i-1):5000*i], from_prepro_level=3, prepro_border=80, prepro_base_url=base_url)
    else:
        gdirs = workflow.init_glacier_directories(gdf_sel[5000*(i-1):], from_prepro_level=3, prepro_border=80, prepro_base_url=base_url)

    # AAR 0.58 global mean
    AAR = 0.58
    compile_from_AAR(gdirs, filesuffix='_oggm_'+str(i), path=True, csv=True, AAR=AAR);

    # AAR 0.5
    AAR = 0.5
    compile_from_AAR(gdirs, filesuffix='_oggm_'+str(i), path=True, csv=True, AAR=AAR);

    # AABR 1.56 global median
    AABR = 1.56
    compile_from_AABR(gdirs, filesuffix='_oggm_'+str(i), path=True, csv=True, AABR=AABR);

    # AABR 1.75 global mean
    AABR = 1.75
    compile_from_AABR(gdirs, filesuffix='_oggm_'+str(i), path=True, csv=True, AABR=AABR);
    
    # THAR 0.5
    THAR = 0.5
    compile_from_THAR(gdirs, filesuffix='_oggm_'+str(i), path=True, csv=True, THAR=THAR);
    
    # AAR: compile, intercept, steady
    AAR_all = xr.open_dataset(cfg.PATHS['working_dir'] + 'results_AAR_debriscalving.nc')
    compile_AABR(gdirs, filesuffix='_compile_'+str(i), path=True, csv=True, AAR_all=AAR_all, name='compile_AAR');
    compile_AABR(gdirs, filesuffix='_intercept_'+str(i), path=True, csv=True, AAR_all=AAR_all, name='intercept_AAR');
    compile_AABR(gdirs, filesuffix='_steady_'+str(i), path=True, csv=True, AAR_all=AAR_all, name='steady_AAR');
    AAR_all.close()

    # get glacier temperature: ERA5
    ds = xr.open_dataset(cfg.PATHS['working_dir']+'glacier_statistics.nc')
    temp_all = ds['temp']
    lapserate_all = ds['lapserate']
    ele_all = ds['ele']
    compile_temp(gdirs, filesuffix='_oggm_'+str(i), path=True, csv=True, 
                 temp_all=temp_all, lapserate_all=lapserate_all, ele_all=ele_all);
    
    ds.close()
    
    folder_path = cfg.PATHS['working_dir'] + 'per_glacier'
    shutil.rmtree(folder_path)
