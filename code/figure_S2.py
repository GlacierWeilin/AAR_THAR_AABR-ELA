#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 10 15:34:31 2024

@author: Weilin Yang (weilinyang.yang@monash.edu)
"""

import numpy as np
import pandas as pd
import scipy.stats as st
import matplotlib.pyplot as plt

data = pd.read_csv('/Users/wyan0065/Desktop/AAR-disequilibrium/AABR/data_and_code/data/Hintereisferner_glacier_AAR.csv')
data = data.set_index('Year')
data = data.loc['1995':'2014']

x = data['SMB'].values;
y = data['AAR'].values/100;
loc = np.where(y < 0.05)[0]
outlier_x = x[loc]
outlier_y = y[loc]
loc = np.where(y >= 0.05)[0]
x = x[loc]
y = y[loc]
slope, intercept, r_value, p_value, std_err = st.linregress(x, y);

new_x =np.linspace(-1900, 200, 100)
#%%
plt.rcParams.update({'lines.linewidth':0.7})
plt.rcParams.update({'font.size': 7})
plt.rcParams.update({'font.sans-serif': 'Arial'})

plt.rcParams.update({'axes.linewidth':0.7})
plt.rcParams.update({'axes.titlepad':3})
plt.rcParams.update({'axes.titlesize':8})
plt.rcParams.update({'axes.labelpad':2})
plt.rcParams.update({'xtick.major.pad':2})
plt.rcParams.update({'ytick.major.pad':2})
plt.rcParams.update({'xtick.major.width':0.5})
plt.rcParams.update({'ytick.major.width':0.5})
plt.rcParams.update({'xtick.major.size':1.5})
plt.rcParams.update({'ytick.major.size':1.5})

fig = plt.figure(figsize=(3.15, 1.83), dpi=600)

ax = plt.axes([0.05,0.15,0.94,0.78], xlim=(-2000,200), ylim=(0,0.9))
ax.set_title('RGI60-11.00897: Hintereisferner glacier')
ax.set(xlabel='MB (mm w.e.)')
ax.set(ylabel='AAR')

ax.plot(x, y, 'o', markersize=2, color='#489FE3', label='Annual AARs')
ax.plot(outlier_x, outlier_y, 'o', markersize=1.5, color='black', label='Outliers')
point_x = 0
point_y = intercept
ax.scatter(point_x, point_y, color='orange', s=10, zorder=10, label=f'AAR$_0$: {intercept:.3f}')

ax.plot(new_x, new_x*slope+intercept, '--', linewidth=1, color='#489FE3', label='AAR = AAR$_{0}$+k×MB')

ax.set_yticks([0.1,0.3,0.5,0.6,0.8])
ax.spines['left'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_position(('data', 0))

ax.yaxis.set_label_position("right")
ax.yaxis.tick_right()

ax.legend(loc='upper left', ncols=1, fontsize=6, markerscale=0.7, frameon=False, borderpad=0.3, 
          labelspacing=0.3, handlelength=1.5, handletextpad=0.4, alignment='left');

out_pdf = '/Users/wyan0065/Desktop/AAR-disequilibrium/AABR/manuscript/' + 'figure_S2.png'
plt.savefig(out_pdf, dpi=600)

plt.show()