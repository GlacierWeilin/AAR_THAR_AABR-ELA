#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 26 20:44:36 2025

@author: Weilin Yang (weilinyang.yang@monash.edu)
"""

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.feature import ShapelyFeature
from cartopy.io.shapereader import Reader

regions_shp = '/Users/wyan0065/Desktop/AAR-disequilibrium/AABR/data_and_code/data/00_rgi60_O1Regions.shp'
rgi_shp     = '/Users/wyan0065/Desktop/AAR-disequilibrium/AABR/data_and_code/data/rgi60_all.shp'
csv_path = '/Users/wyan0065/Desktop/AAR-disequilibrium/AABR/data_and_code/data/WGMS_AAR_RGI.csv'
df = pd.read_csv(csv_path)
RGI = df[['RGIId', 'lon', 'lat']]

# background
plt.rcParams.update({'lines.linewidth':0.5})
plt.rcParams.update({'font.size': 6})
plt.rcParams.update({'font.sans-serif': 'Arial'})

plt.rcParams.update({'axes.linewidth':0.5})
plt.rcParams.update({'axes.titlepad':3})
plt.rcParams.update({'axes.titlesize':6})
plt.rcParams.update({'axes.labelpad':2})
plt.rcParams.update({'xtick.major.pad':2})
plt.rcParams.update({'ytick.major.pad':2})
plt.rcParams.update({'xtick.major.width':0.5})
plt.rcParams.update({'ytick.major.width':0.5})
plt.rcParams.update({'xtick.major.size':1.5})
plt.rcParams.update({'ytick.major.size':1.5})

fig = plt.figure(figsize=(4.72, 2.57), dpi=600)
ax_back = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree(),facecolor='None')
box_fig = fig.get_window_extent()

ax_back.set_global()
ax_back.spines['geo'].set_linewidth(0)

# add data
ax = fig.add_axes([0.05,0.05,0.9,0.9],projection=ccrs.Robinson())

ax.set_global()
ax.add_feature(cfeature.LAND, facecolor='gray')
ax.add_feature(cfeature.OCEAN, facecolor='lightgray')

shape_feature = ShapelyFeature(Reader(regions_shp).geometries(), ccrs.PlateCarree(), 
                               edgecolor='white', facecolor='none', linewidth=1)
ax.add_feature(shape_feature)

rgi_feature = ShapelyFeature(Reader(rgi_shp).geometries(),ccrs.PlateCarree(),
                             edgecolor='#489FE3',facecolor='#489FE3',linewidth=1)
ax.add_feature(rgi_feature)

ax.scatter(df['lon'],df['lat'],transform=ccrs.PlateCarree(),color='orange',edgecolor='orange',s=3,zorder=10)
#ax.scatter(10.77000046,46.79999924,transform=ccrs.PlateCarree(),color='red',edgecolor='orange',s=15,zorder=10)

# add labels
ax_label = fig.add_axes([0,0,1,1], projection=ccrs.PlateCarree(),facecolor='None')
ax_label.spines['geo'].set_linewidth(0)

text_label = ['(1) Alaska', '(2) W Canada & US', '(3) Arctic Canada\n North', '(4) Arctic Canada\n South', '(5) Greenland\n Periphery',\
              '(6) Iceland', '(7) Svalbard', '(8) Scandinavia', '(9) Russian Arctic', '(10) North Asia', \
                  '(11) Central\n Europe', '(12) Caucasus\n & Middle East', '(13) Central\n Asia', '(14) South\n Asia West',\
                      '(15) South\n Asia East', '(16) Low Latitudes', '(17) Southern Andes',\
                          '(18) New Zealand','(19) Antarctic & Subantarctic']
#                      1,     2,   3,   4,   5,   6,     7,  8,    9, 10, 11, 12,  13, 14, 15, 16,  17,  18,   19
text_lon = np.array([-129, -100, -65, -55, -25, -15,    10, 30,   50, 90,  2, 36, 104, 60, 88, 20, -32, 130,  10])
text_lat = np.array([  49,   34,  95,  56,  95,  58,  89.5, 62, 89.5, 82, 39, 30,  46, 15, 24, -7, -35, -30, -48])

for i in range(0,19):
    ax_label.text(text_lon[i], text_lat[i], text_label[i], fontsize=5, alpha=1, 
                  color='black', ha='center', va='top', transform=ccrs.PlateCarree(), 
                  bbox={'facecolor':'white', 'pad': 1, 'linewidth': 0.2});

text_label =['A.West and Peninsula', 'B.East 1', 'C.East 2', 'A.Altay and Sayan', 'B.Ural', 'C.Kamchatka Krai']
text_lon = np.array([-85,  20, 100, 73, 53, 120])
text_lat = np.array([-65, -60, -60, 58, 68, 52])

for i in range(0,6):
    ax_label.text(text_lon[i], text_lat[i], text_label[i], fontsize=5, alpha=1, 
                  color='black', ha='center', va='top', transform=ccrs.PlateCarree());

anno_lon = np.array([[-65, -21, 10, 45, 89, 60, 92],[-60, -27,  3, 38, 78, 62, 86]])
anno_lat = np.array([[ 83,  83, 83, 83, 42, 16, 25],[ 76,  76, 76, 76, 45, 32, 30]])

for i in range(0,7):
    ax_label.annotate('',xy=(anno_lon[0,i], anno_lat[0,i]),xytext=(anno_lon[1,i], anno_lat[1,i]), 
                      xycoords=ccrs.PlateCarree()._as_mpl_transform(ax_label),
                      textcoords=ccrs.PlateCarree()._as_mpl_transform(ax_label),
                      arrowprops=dict(arrowstyle='-',color='black',linewidth=0.3,shrinkA=0,shrinkB=0),
                      transform=ccrs.PlateCarree())

#ax_label.text(3, 48, 'Hintereisferner', fontsize=5, alpha=1, 
#              color='red', ha='right', va='center', transform=ccrs.PlateCarree());
#ax_label.annotate('',xy=(9.5, 48),xytext=(3, 48), 
#                  xycoords=ccrs.PlateCarree()._as_mpl_transform(ax_label),
#                  textcoords=ccrs.PlateCarree()._as_mpl_transform(ax_label),
#                  arrowprops=dict(arrowstyle='<-',color='red',linewidth=0.3,shrinkA=0,shrinkB=0),
#                  transform=ccrs.PlateCarree(), color='red', fontsize=5)

# spine and legend
for spine in ax.spines.values():
    spine.set_edgecolor('lightgrey')
    spine.set_linewidth(0.5)
    
#gl = ax_label.gridlines(
#    crs=ccrs.PlateCarree(),
#    draw_labels=True,
#    linewidth=0.5,
#    color='gray',
#    alpha=0.5,
#    linestyle='--'
#)

#gl.xlocator = FixedLocator(range(-180, 181, 10))
#gl.ylocator = FixedLocator(range(-90, 91, 10))

#gl.xlabel_style = {'size': 6}
#gl.ylabel_style = {'size': 6}
    
gl = ax.gridlines(
    crs=ccrs.PlateCarree(),
    draw_labels=True,
    linewidth=0.5,
    color='gray',
    alpha=0.5,
    linestyle='--'
)

gl.top_labels = False
gl.xlabel_style = {'size': 6}
gl.ylabel_style = {'size': 6}
gl.ypadding = 3
gl.xpadding = 3

legend_elements = [
    #Line2D([0], [0], marker='o', color='red', label='Hintereisferner',
    #      markersize=2, linestyle='None', markeredgecolor='red'),
    Line2D([0], [0], marker='o', color='orange', label='WGMS Glaciers',
          markersize=1.8, linestyle='None', markeredgecolor='orange'),
    Line2D([0], [0], color='white', linewidth=1, label='RGI Regions'),
    Patch(facecolor='#489FE3', edgecolor='#489FE3', label='Glaciers')
]

legend = ax.legend(handles=legend_elements,
                   loc='lower left',
                   bbox_to_anchor=(0, 0.385),
                   frameon=False,
                   framealpha=0.8,
                   fontsize=6,
                   title_fontsize=6,
                   borderpad=0.5,
                   handlelength=1.5)

#for text in legend.get_texts():
#    if text.get_text() == 'Hintereisferner':
#        text.set_color('red')
#    else:
#        text.set_color('black')

out_pdf = '/Users/wyan0065/Desktop/AAR-disequilibrium/AABR/manuscript/' + 'figure_S1.png'
plt.savefig(out_pdf, dpi=600)
plt.show()