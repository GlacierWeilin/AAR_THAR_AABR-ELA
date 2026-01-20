#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 13 16:15:36 2024

@author: Weilin Yang (weilinyang.yang@monash.edu)
"""

import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import matplotlib.colors as mcolors
import cartopy.crs as ccrs
import calendar

file_fp = '/Users/wyan0065/Desktop/PyGEM/calving/climate_data/ERA5/';
ds_temp = xr.open_dataset(file_fp+'ERA5_temp_monthly.nc', decode_times=False);
tt = ds_temp['t2m'].values
ds_temp.close()
resolution = 0.25
lat = np.arange(-90, 90+resolution, resolution)
lon = np.arange(0, 360, resolution)
lon = (lon+180) % 360-180
[time,m,n] = np.shape(tt)

lonmin = np.min(lon); lonmax = np.max(lon);
latmin = lat[0]; latmax = lat[-1];
extents = [lonmin, lonmax, latmin, latmax]

tt = np.nanmean(tt.reshape(85,12,721,1440), axis=1)
tt = tt[55:75,:,:] # 1995~2014
tt = np.nanmean(tt,axis=0)-273.15
temp = tt.copy()
temp[:,0:int(n/2)] = tt[:,int(n/2):n]
temp[:,int(n/2):n] = tt[:,0:int(n/2)]

ds_prcp = xr.open_dataset(file_fp+'ERA5_totalprecip_monthly.nc', decode_times=False);
pp = ds_prcp['tp'].values
ds_prcp.close()
[time,m,n] = np.shape(pp)
mday = np.zeros([time,1])
k=0
for year in range(1940, 2024):
    for month in range(1,13):
        monthRange = calendar.monthrange(year, month)
        mday[k] = monthRange[1]
        k += 1

mday = np.tile(mday,m*n)
mday = mday.reshape(time,m,n)
pp = pp*mday
pp = np.nansum(pp.reshape(85,12,721,1440), axis=1)
pp = pp[55:75,:,:]
pp = np.nanmean(pp,axis=0)

prcp = pp.copy()
prcp[:,0:int(n/2)] = pp[:,int(n/2):n]
prcp[:,int(n/2):n] = pp[:,0:int(n/2)]


ds_uv = xr.open_dataset(file_fp+'ERA5_10muv_monthly_1995_2014.nc')
u10 = (ds_uv['u10']).mean(dim='valid_time').values
v10 = (ds_uv['v10']).mean(dim='valid_time').values
ds_uv.close()

u = u10.copy()
u[:,0:int(n/2)] = u10[:,int(n/2):n]
u[:,int(n/2):n] = u10[:,0:int(n/2)]

v = v10.copy()
v[:,0:int(n/2)] = v10[:,int(n/2):n]
v[:,int(n/2):n] = v10[:,0:int(n/2)]
#%%
plt.rcParams.update({'lines.linewidth':0.7})
plt.rcParams.update({'font.size': 8})
plt.rcParams.update({'font.sans-serif': 'Arial'})

plt.rcParams.update({'axes.linewidth':0.7})
plt.rcParams.update({'axes.titlepad':3})
plt.rcParams.update({'axes.titlesize':7})
plt.rcParams.update({'axes.labelpad':2})
plt.rcParams.update({'xtick.major.pad':2})
plt.rcParams.update({'ytick.major.pad':2})
plt.rcParams.update({'xtick.major.width':0.7})
plt.rcParams.update({'ytick.major.width':0.7})
plt.rcParams.update({'xtick.major.size':1.5})
plt.rcParams.update({'ytick.major.size':1.5})

fig = plt.figure(figsize=(4.72, 6.8), dpi=600)
gs = GridSpec(3, 1, figure=fig, hspace=0.2, height_ratios=[1,1,1])
plt.subplots_adjust(left=-0.02, right=1.1, top=0.98, bottom=0.02)

proj = ccrs.Robinson()
## ======================================================================== figure a ========================================================================
ax1 = fig.add_subplot(gs[0, 0], projection=proj)
ax1.set_global()
ax1.coastlines(resolution='10m', lw=0.5)
ax1.set_title('Mean annual temperature during 1995 ~ 2014 ($^\circ$C)', fontweight='bold',loc='center', pad=2)
ax1.text(0, 1.008, 'a', fontsize=8, fontweight='bold', transform=ax1.transAxes);
ax1.tick_params(axis='both', which='major', length=2, width=0.5, color='black', labelcolor='black', pad=1)
ax1.tick_params(axis='both', which='minor', length=1, width=0.5, color='black')

gl1 = ax1.gridlines(
    crs=ccrs.PlateCarree(),
    draw_labels=True,
    linewidth=0.5,
    color='gray',
    alpha=0.5,
    linestyle='--'
)

gl1.top_labels = False
gl1.xlabel_style = {'size': 7}
gl1.ylabel_style = {'size': 7}
gl1.ypadding = 3
gl1.xpadding = 3

norm = mcolors.TwoSlopeNorm(vmin=-50, vcenter=0, vmax=30)
im1  = ax1.imshow(temp, extent=extents, transform=ccrs.PlateCarree(), alpha=0.8,
                  norm=norm, cmap='RdYlBu_r')

char1 = fig.colorbar(im1, ax=ax1, ticks=np.array([-50,-30,-10,0,10,20,30]), extend='both',
                     shrink=0.7, aspect=20, pad=0.06, orientation='vertical')
char1.ax.tick_params(direction='in', size=2, width=0.5, labelsize=7)

## ======================================================================== figure b ========================================================================
ax2 = fig.add_subplot(gs[1, 0], projection=proj)
ax2.set_global()
ax2.coastlines(resolution='10m', lw=0.5)
ax2.set_title('Mean annual precipitation during 1995 ~ 2014 (m)', fontweight='bold',loc='center', pad=2)
ax2.text(0, 1.008, 'b', fontsize=8, fontweight='bold', transform=ax2.transAxes);
ax2.tick_params(axis='both', which='major', length=2, width=0.5, color='black', labelcolor='black', pad=1)
ax2.tick_params(axis='both', which='minor', length=1, width=0.5, color='black')

gl2 = ax2.gridlines(
    crs=ccrs.PlateCarree(),
    draw_labels=True,
    linewidth=0.5,
    color='gray',
    alpha=0.5,
    linestyle='--'
)

gl2.top_labels = False
gl2.xlabel_style = {'size': 7}
gl2.ylabel_style = {'size': 7}
gl2.ypadding = 3
gl2.xpadding = 3

norm = mcolors.Normalize(vmin=0, vmax=3)
im2  = ax2.imshow(prcp, extent=extents, transform=ccrs.PlateCarree(), alpha=0.8,
                  norm=norm, cmap='RdYlBu_r')

char2 = fig.colorbar(im2, ax=ax2, ticks=np.linspace(0, 3, 6), extend='both',
                     shrink=0.7, aspect=20, pad=0.06, orientation='vertical')
char2.ax.tick_params(direction='in', size=2, width=0.5, labelsize=7)

## ======================================================================== figure c ========================================================================
ax3 = fig.add_subplot(gs[2, 0], projection=proj)
ax3.set_global()
ax3.coastlines(resolution='10m', lw=0.5)
ax3.set_title('Mean annual 10m wind field during 1995 ~ 2014 (m/s)', fontweight='bold',loc='center', pad=2)
ax3.text(0, 1.008, 'c', fontsize=8, fontweight='bold', transform=ax3.transAxes);

gl3 = ax3.gridlines(
    crs=ccrs.PlateCarree(),
    draw_labels=True,
    linewidth=0.5,
    color='gray',
    alpha=0.5,
    linestyle='--'
)

gl3.top_labels = False
gl3.xlabel_style = {'size': 7}
gl3.ylabel_style = {'size': 7}
gl3.ypadding = 3
gl3.xpadding = 3

wind_speed = np.sqrt(u**2 + v**2)

norm = mcolors.Normalize(vmin=0, vmax=12)
im3  = ax3.imshow(wind_speed, extent=extents, transform=ccrs.PlateCarree(), alpha=0.8,
                  norm=norm, cmap='RdYlBu_r')
char3 = fig.colorbar(im3, ax=ax3, ticks=np.linspace(0, 12, 4), extend='both',
                     shrink=0.7, aspect=20, pad=0.06, orientation='vertical')
char3.ax.tick_params(direction='in', size=2, width=0.5, labelsize=7)

step = 1
lon_sub = lon[::step]
lat_sub = lat[::step]
u_sub = u[::step, ::step]
v_sub = v[::step, ::step]

Lon, Lat = np.meshgrid(lon_sub, lat_sub)

ax3.streamplot(
    Lon, Lat, u_sub, v_sub,
    density=2, 
    linewidth=0.5,
    arrowsize=0.5,
    color='white',
    transform=ccrs.PlateCarree()
)

out_pdf = '/Users/wyan0065/Desktop/AAR-disequilibrium/AABR/manuscript/' + 'figure_S6.png'
plt.savefig(out_pdf, dpi=300)

plt.show()