#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 27 12:16:24 2025

@author: Romain Hugonnet(romain.hugonnet@gmail.com) "Accelerated global glacier mass loss in the early twenty-first century"
@revised: Weilin Yang (weilinyang.yang@monash.edu)
"""

import numpy as np
import xarray as xr

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.path as mpath
from matplotlib.cm import ScalarMappable
import cartopy.crs as ccrs
import cartopy.feature as cfeature

# functions

def coordXform(orig_crs, target_crs, x, y):
    return target_crs.transform_points(orig_crs, x, y )

def poly_from_extent(ext):

    poly = np.array([(ext[0],ext[2]),(ext[1],ext[2]),(ext[1],ext[3]),(ext[0],ext[3]),(ext[0],ext[2])])

    return poly

def latlon_extent_to_robinson_axes_verts(polygon_coords):

    robin = np.transpose(np.array(list(zip(*polygon_coords)),dtype=float))


    limits_robin = coordXform(ccrs.PlateCarree(),ccrs.Robinson(),np.array([-179.99,179.99,0,0]),np.array([0,0,-89.99,89.99]))

    ext_robin_x = limits_robin[1][0] - limits_robin[0][0]
    ext_robin_y = limits_robin[3][1] - limits_robin[2][1]

    verts = robin.copy()
    verts[:,0] = (verts[:,0] + limits_robin[1][0])/ext_robin_x
    verts[:,1] = (verts[:,1] + limits_robin[3][1])/ext_robin_y

    return verts[:,0:2]

def add_inset(fig,extent,pos,bounds,markup_sub=None,polygon=None,sub_pos=None,sub_adj=None, 
              data=None,norm=None,cmap_cus=None,label=None):

    sub_ax = fig.add_axes(pos,projection=ccrs.Robinson())
    sub_ax.set_extent(extent, ccrs.Geodetic())

    sub_ax.add_feature(cfeature.NaturalEarthFeature('physical', 'land', '50m', facecolor='grey'))
    sub_ax.add_feature(cfeature.NaturalEarthFeature('physical', 'ocean', '50m', facecolor='lightgrey'))
    
    sub_ax.imshow(data, transform=ccrs.PlateCarree(), alpha=1,
                  norm=norm, cmap=cmap_cus, zorder=2);
    
    if polygon is None and bounds is not None:
        polygon = poly_from_extent(bounds)

    if bounds is not None:
        verts = mpath.Path(latlon_extent_to_robinson_axes_verts(polygon))
        sub_ax.set_boundary(verts, transform=sub_ax.transAxes)
        
    sub_ax.spines['geo'].set_edgecolor('white')
    
    if markup_sub is not None and label[1] == 'a':

        lon_min = np.min(list(zip(*polygon))[0])
        lon_max = np.max(list(zip(*polygon))[0])
        lon_mid = 0.5*(lon_min+lon_max)

        lat_min = np.min(list(zip(*polygon))[1])
        lat_max = np.max(list(zip(*polygon))[1])
        lat_mid = 0.5*(lat_min+lat_max)

        robin = np.array(list(zip([lon_min,lon_min,lon_min,lon_mid,lon_mid,lon_max,lon_max,lon_max],[lat_min,lat_mid,lat_max,lat_min,lat_max,lat_min,lat_mid,lat_max])))

        if sub_pos=='lb':
            rob_x = robin[0][0]
            rob_y = robin[0][1]
            ha='left'
            va='bottom'
        elif sub_pos=='lm':
            rob_x = robin[1][0]
            rob_y = robin[1][1]
            ha='left'
            va='center'
        elif sub_pos=='lt':
            rob_x = robin[2][0]
            rob_y = robin[2][1]
            ha='left'
            va='top'
        elif sub_pos=='mb':
            rob_x = robin[3][0]
            rob_y = robin[3][1]
            ha='center'
            va='bottom'
        elif sub_pos=='mt':
            rob_x = robin[4][0]
            rob_y = robin[4][1]
            ha='center'
            va='top'
        elif sub_pos=='rb':
            rob_x = robin[5][0]
            rob_y = robin[5][1]
            ha='right'
            va='bottom'
        elif sub_pos=='rm':
            rob_x = robin[6][0]
            rob_y = robin[6][1]
            ha='right'
            va='center'
        elif sub_pos=='rt':
            rob_x = robin[7][0]
            rob_y = robin[7][1]
            ha='right'
            va='top'

        if sub_pos[0] == 'r':
            rob_x = rob_x - 100000
        elif sub_pos[0] == 'l':
            rob_x = rob_x + 100000

        if sub_pos[1] == 'b':
            rob_y = rob_y + 100000
        elif sub_pos[1] == 't':
            rob_y = rob_y - 100000

        if sub_adj is not None:
            rob_x += sub_adj[0]
            rob_y += sub_adj[1]

        sub_ax.text(rob_x,rob_y,markup_sub,
                 horizontalalignment=ha, verticalalignment=va,
                 transform=ccrs.Robinson(), color='black',fontsize=4,bbox=dict(facecolor='white', alpha=1,linewidth=0.35,pad=1.5),fontweight='bold',zorder=25)

def add_compartment_world_map(fig,pos,label, data, norm, cmap_cus,ticks):

    yr = pos[3]
    yb = pos[1]

    xr = pos[2]
    xb = pos[0]

    #Antarctic Peninsula
    bounds_ap = [-5500000,-3400000,-8000000,-5900000]
    add_inset(fig, [-179.99,179.99,-89.99,89.99], [xb-0.735*xr,yb -0.06*yr, 2.7*xr, 2.7*yr],
              bounds=bounds_ap,markup_sub='19A',sub_pos='lt',sub_adj=(30000,-170000), data=data, norm=norm, cmap_cus=cmap_cus, label=label)

    #Antarctic West
    bounds_aw=[-9500000,-5600000,-7930000,-7320000]
    add_inset(fig, [-179.99,179.99,-89.99,89.99], [xb-0.4205*xr, yb-0.1513*yr, 2.7*xr, 2.7*yr],
              bounds=bounds_aw,markup_sub='19A',sub_pos='lt',sub_adj=(70000,-15000), data=data, norm=norm, cmap_cus=cmap_cus, label=label)

    #Antarctic East
    bounds_ae1 = [-1960000,2250000,-7700000,-7080000]
    add_inset(fig, [-179.99,179.99,-89.99,89.99], [xb-0.7087*xr, yb-0.1872*yr, 2.7*xr, 2.7*yr],
              bounds=bounds_ae1,markup_sub='19B',sub_pos='lt',sub_adj=(0,-25000), data=data, norm=norm, cmap_cus=cmap_cus, label=label)

    #Antartic East 2
    bounds_ae2 = [2450000,7500000,-7570000,-6720000]
    add_inset(fig, [-179.99,179.99,-89.99,89.99], [xb-1.0585*xr, yb-0.113*yr, 2.7*xr, 2.7*yr],
              bounds=bounds_ae2,markup_sub='19C',sub_pos='rt',sub_adj=(-1075000,-45000), data=data, norm=norm, cmap_cus=cmap_cus, label=label)

    #Antartic East 3
    bounds_ae3=[9430000,11900000,-8200000,-6770000]
    add_inset(fig, [-179.99,179.99,-89.99,89.99], [xb-1.2960*xr, yb-0.109*yr, 2.7*xr, 2.7*yr],
              bounds=bounds_ae3,markup_sub='19C',sub_pos='lm',sub_adj=(15000,0), data=data, norm=norm, cmap_cus=cmap_cus, label=label)

    #South America
    bounds_sa = [-7340000,-5100000,-5900000,0]
    add_inset(fig, [-179.99,179.99,-89.99,89.99], [xb-0.765*xr, yb-0.4694*yr, 2.7*xr, 2.7*yr],
              bounds=bounds_sa,markup_sub='16-17',sub_pos='lm',sub_adj=(20000,-2000000), data=data, norm=norm, cmap_cus=cmap_cus, label=label)

    #Europe
    bounds_eu = [0,1500000,4500000,5400000]
    add_inset(fig, [-179.99,179.99,-89.99,89.99], [xb-0.865*xr, yb-1.645*yr, 2.7*xr, 2.7*yr],
              bounds=bounds_eu,markup_sub='11',sub_pos='lt',sub_adj=(20000,-135000), data=data, norm=norm, cmap_cus=cmap_cus, label=label)

    #Caucasus
    bounds_cau = [3200000,4800000,3300000,4800000]
    add_inset(fig, [-179.99,179.99,-89.99,89.99], [xb-1.122*xr,yb -1.685*yr, 2.7*xr, 2.7*yr],
              bounds=bounds_cau,markup_sub='12',sub_pos='lb',sub_adj=(60000,20000), data=data, norm=norm, cmap_cus=cmap_cus, label=label)

    #New Zealand
    bounds_nz=[13750000,15225000,-5400000,-3800000]
    add_inset(fig, [-179.99,179.99,-89.99,89.99], [xb-1.56*xr, yb-0.3235*yr, 2.7*xr, 2.7*yr],
              bounds=bounds_nz,markup_sub='18',sub_pos='lt',sub_adj=(135000,-100000), data=data, norm=norm, cmap_cus=cmap_cus, label=label)

    #Kamchatka Krai
    bounds_kam = [11500000,13200000,5100000,6700000]
    add_inset(fig, [-179.99,179.99,-89.99,89.99], [xb-1.3991*xr, yb-1.73*yr, 2.7*xr, 2.7*yr],
              bounds=bounds_kam,markup_sub='10C',sub_pos='lb',sub_adj=(365000,15000), data=data, norm=norm, cmap_cus=cmap_cus, label=label)

    #HMA and North Asia 1
    bounds_hma = [5750000,9550000,2650000,5850000]
    add_inset(fig, [-179.99,179.99,-89.99,89.99], [xb-1.216*xr,yb -1.5835*yr, 2.7*xr, 2.7*yr],
              bounds=bounds_hma,markup_sub='13-15\n& 10A',sub_pos='rt',sub_adj=(-20000,-100000), data=data, norm=norm, cmap_cus=cmap_cus, label=label)

    #Arctic
    bounds_arctic = [-6060000,6420000,6100000,8400000]
    poly_arctic = np.array([(-6050000,7650000),(-5400000,6800000),(-4950000,6400000),(-3870000,5710000),(-2500000,5710000),(-2000000,5720000),(1350000,5720000),(2300000,6600000),(6500000,6600000),(6500000,8400000),(-6050000,8400000),(-6050000,7650000)])
    add_inset(fig, [-179.99,179.99,-89.99,89.99], [xb-0.8675*xr, yb-1.715*yr, 2.7*xr, 2.7*yr],
              bounds=bounds_arctic,polygon=poly_arctic,markup_sub='03-09\n& 10B',sub_pos='mt',sub_adj=(-200000,-820000),
              data=data, norm=norm, cmap_cus=cmap_cus, label=label)

    #North America
    bounds_na = [-13600000,-9000000,3700000,7350000]
    poly_na = np.array([(-13600000,5600000),(-13600000,6000000),(-12900000,6000000),(-12900000,6800000),(-12500000,6800000),(-11500000,7420000),(-9000000,7420000),(-9000000,3750000),(-11000000,3750000),(-11000000,5600000),(-13600000,5600000)])
    add_inset(fig, [-179.99,179.99,-89.99,89.99], [xb-0.15*xr, yb-1.885*yr, 2.7*xr, 2.7*yr],
              bounds=bounds_na,polygon=poly_na,markup_sub='01-02',sub_pos='rm',sub_adj=(-1970000,230000),
              data=data, norm=norm, cmap_cus=cmap_cus, label=label)
    
    out_axes = fig.add_axes(pos)
    out_axes.set_xticks([])
    out_axes.set_yticks([])
    out_axes.spines['top'].set_visible(False)
    out_axes.spines['left'].set_visible(False)
    out_axes.spines['right'].set_visible(False)
    out_axes.spines['bottom'].set_visible(False)
    out_axes.set_facecolor('none')
    if label is not None:
        out_axes.text(0.5, 0.98, label[0], transform=out_axes.transAxes,fontsize=6, fontweight='bold', ha='center', va='center',zorder=30)
        out_axes.text(0.01, 0.98, label[1], transform=out_axes.transAxes,fontsize=6, fontweight='bold', ha='center', va='center',zorder=30)
    
    legend_pos = pos
    legend_pos[1] = legend_pos[1] - 0.06
    ax_legend = fig.add_axes(pos)
    ax_legend.set_xticks([])
    ax_legend.set_yticks([])
    ax_legend.spines['top'].set_visible(False)
    ax_legend.spines['left'].set_visible(False)
    ax_legend.spines['right'].set_visible(False)
    ax_legend.spines['bottom'].set_visible(False)
    ax_legend.set_facecolor('none')
    sm = ScalarMappable(norm=norm, cmap=cmap_cus)
    
    cbar = fig.colorbar(sm,ax=ax_legend,ticks=ticks,extend='both',orientation='horizontal',
                        shrink=0.5,pad=0,aspect=27)

    cbar.ax.tick_params(direction='in', size=2, width=0.5, labelsize=5.2, pad=1.5, labelcolor='black')
    for label in cbar.ax.get_yticklabels():
        label.set_fontweight('bold')
#%%        
# add data
plt.rcParams.update({'font.size': 5.2})
plt.rcParams.update({'lines.linewidth':0.35})
plt.rcParams.update({'axes.linewidth':0.35})
plt.rcParams.update({'lines.markersize':2.5})
plt.rcParams.update({'axes.labelpad':1.5})
plt.rcParams.update({'font.sans-serif': 'Arial'})

fig_width_inch = 3.1
fig = plt.figure(figsize=(fig_width_inch, 2.35 * fig_width_inch / 1.9716 * 1.5))

data_path = '/Users/wyan0065/Desktop/AAR-disequilibrium/AABR/data_and_code/data/results_0.5.nc'
ds = xr.open_dataset(data_path)

names = ['AAR_compile', 'THAR_compile', 'AABR_compile']
all_ticks = np.array([[0.439, 0.492, 0.544, 0.607, 0.670],
                       [0.332,0.422, 0.512, 0.602, 0.692],[0.83, 1.18, 1.53, 2.02, 2.51]])

all_pos = np.array([[0.006, 0.71666666666667, 0.98, 0.27801418439716313],  #0.5, 0.425531914893617
                    [0.006, 0.38333333333333, 0.98, 0.27801418439716313], 
                    [0.006, 0.05, 0.98, 0.27801418439716313]])

all_label = np.array([['Steady-state AAR (AAR$_0$)','a'], 
                      ['Steady-state THAR (THAR$_0$)','b'], ['Steady-state AABR (AABR$_0$)','c']])

for i,name in enumerate(names):
    
    ticks = all_ticks[i,:]
    pos   = all_pos[i,:]
    label = all_label[i,:]
    
    data = ds[name].values[:,:,0]
    data = np.flipud(data)

    col_bounds = np.linspace(ticks[0],ticks[2],10)
    col_bounds = np.append(col_bounds, np.linspace(ticks[2],ticks[4],10))
    cb = []
    cb_val = np.linspace(1, 0, len(col_bounds))
    for j in range(len(cb_val)):
        cb.append(mpl.cm.RdBu(cb_val[j]))
        cmap_cus = mpl.colors.LinearSegmentedColormap.from_list('my_cb', list(zip((col_bounds - min(col_bounds)) / (max(col_bounds - min(col_bounds))), 
                                                                                  cb)), N=1000)
        norm = mpl.colors.Normalize(vmin=min(col_bounds), vmax=max(col_bounds))

    add_compartment_world_map(fig,pos,label,data,norm,cmap_cus,ticks)

out_pdf = '/Users/wyan0065/Desktop/AAR-disequilibrium/AABR/manuscript/' + 'figure_2.png'
plt.savefig(out_pdf, dpi=600)
plt.show()


