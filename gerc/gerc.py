#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
Created on Tue Jan 20 20:06:17 2026

@author: Weilin Yang (weilinyang.yang@monash.edu)
'''

import streamlit as st
from core.classification import run_classification

st.set_page_config(page_title='GERC', layout='centered')

st.title('Glacier ELA Ratio Calculator (GERC)')
st.caption('@author: Weilin Yang (weilinyang.yang@monash.edu)')

st.markdown('### Glacier Type')

is_debris = st.radio('Is the glacier debris-covered?', [0, 1], format_func=lambda x: 'No' if x == 0 else 'Yes')
is_icecap = st.radio('Is the glacier an icecap?', [0, 1], format_func=lambda x: 'No' if x == 0 else 'Yes')
is_tidewater = st.radio('Is the glacier marine-terminating?', [0, 1], format_func=lambda x: 'No' if x == 0 else 'Yes')

st.markdown('### Glacier Attributes')

temp = st.radio(
    'Glacier temperature category (°C/year)',
    options=[-1, 0, 1, 2],
    format_func=lambda x: {
        -1: 'Not Applicable',
         0: 'Low (< -11.1)',
         1: 'Moderate (-11.1 ~ -5.7)',
         2: 'High (≥ -5.7)'
    }[x]
)

prcp = st.radio(
    'Glacier precipitation category (mm/year)',
    options=[-1, 0, 1, 2],
    format_func=lambda x: {
        -1: 'Not Applicable',
         0: 'Low (< 630)',
         1: 'Moderate (630 ~ 1300)',
         2: 'High (≥ 1300)'
    }[x]
)

area = st.radio(
    'Glacier area category (km²)',
    options=[-1, 0, 1, 2],
    format_func=lambda x: {
        -1: 'Not Applicable',
         0: 'Low (< 0.13)',
         1: 'Moderate (0.13 ~ 0.51)',
         2: 'High (≥ 0.51)'
    }[x]
)

slope = st.radio(
    'Glacier slope category (°)',
    options=[-1, 0, 1, 2],
    format_func=lambda x: {
        -1: 'Not Applicable',
         0: 'Low (< 20)',
         1: 'Moderate (20 ~ 27)',
         2: 'High (≥ 27)'
    }[x]
)

aspect = st.radio(
    'Glacier aspect category',
    options=[-1, 0, 1, 2, 3],
    format_func=lambda x: {
        -1: 'Not Applicable',
         0: 'Northern Hemisphere – South-facing',
         1: 'Northern Hemisphere – North-facing',
         2: 'Southern Hemisphere – South-facing',
         3: 'Southern Hemisphere – North-facing'
    }[x]
)


if st.button('Run classification'):
    df = run_classification(
        is_debris, is_icecap, is_tidewater,
        temp, prcp, area, slope, aspect
    )

    st.download_button(
        '📥 Download Results CSV',
        df.to_csv(index=False, header=False).encode('utf-8-sig'),
        'Results.csv',
        'text/csv'
    )
