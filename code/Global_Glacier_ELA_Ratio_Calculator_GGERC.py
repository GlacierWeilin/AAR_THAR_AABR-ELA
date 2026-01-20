#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
Created on June 27, 2025

@author: Weilin Yang (weilinyang.yang@monash.edu)

Instruction:

Purpose:
    This tool is used to derive the ELA ratios (AAR, THAR, and AABR) for a group of glaciers 
    that share similar glacier characteristics.

Prerequisites:
1. Python Environment:
    - Recommended Python 3.10.16 (this version has been used for testing)

2. Required Python Libraries:
    - pandas (tested using v2.2.3)
    - tkinter (tested using v8.6)

3. Data Preparation:
    - Place your classification tree CSV files in a designated folder
    - Modify the variable `classification_dir` below to point to your folder
      Example:
          classification_dir = '/Users/username/Desktop/AAR-disequilibrium/AABR/data_and_code/data/classificationtree/'

4. Running the Tool:
    - Run the script directly in a terminal or command line:
        python Global_Glacier_ELA_Ratio_Calculator_GGERC.py

5. Library Installation Example:
    pip install pandas


'''

import os
import tkinter as tk
from tkinter import messagebox, filedialog
import pandas as pd


def main():
    root = tk.Tk()
    root.title('Global Glacier ELA Ratio Calculator (GGERC)')

    # =========================
    # Header: title + subtitle
    # =========================
    header_frame = tk.Frame(root)
    header_frame.pack(fill='x')

    tk.Label(
        header_frame,
        text='Derive Glacier AAR, THAR, and AABR based on the classification tree',
        font=('Arial', 16, 'bold')
    ).pack(pady=(15, 5))

    tk.Label(
        header_frame,
        text='@author: Weilin Yang (weilinyang.yang@monash.edu)',
        font=('Arial', 12),
        fg='gray'
    ).pack(pady=(0, 15))

    # =========================
    # Main content
    # =========================
    main_frame = tk.Frame(root)
    main_frame.pack(fill='both', expand=True)

    padx = 6

    debris_var = tk.IntVar(value=0)
    tk.Label(main_frame, text='Is the glacier debris-covered?').grid(row=0, column=0, sticky='w', padx=padx)
    tk.Radiobutton(main_frame, text='No', variable=debris_var, value=0).grid(row=0, column=1, sticky='w')
    tk.Radiobutton(main_frame, text='Yes', variable=debris_var, value=1).grid(row=0, column=2, sticky='w')

    icecap_var = tk.IntVar(value=0)
    tk.Label(main_frame, text='Is the glacier an icecap?').grid(row=1, column=0, sticky='w', padx=padx)
    tk.Radiobutton(main_frame, text='No', variable=icecap_var, value=0).grid(row=1, column=1, sticky='w')
    tk.Radiobutton(main_frame, text='Yes', variable=icecap_var, value=1).grid(row=1, column=2, sticky='w')

    tidewater_var = tk.IntVar(value=0)
    tk.Label(main_frame, text='Is the glacier marine-terminating?').grid(row=2, column=0, sticky='w', padx=padx)
    tk.Radiobutton(main_frame, text='No', variable=tidewater_var, value=0).grid(row=2, column=1, sticky='w')
    tk.Radiobutton(main_frame, text='Yes', variable=tidewater_var, value=1).grid(row=2, column=2, sticky='w')

    def create_radiobutton_group(label_text, var, options, row):
        tk.Label(main_frame, text=label_text).grid(row=row, column=0, sticky='w', padx=padx)
        for i, (val, text) in enumerate(options):
            tk.Radiobutton(
                main_frame,
                text=text,
                variable=var,
                value=val
            ).grid(row=row, column=i + 1, sticky='w')

    temp_var = tk.IntVar(value=-1)
    temp_options = [
        (-1, 'Not Applicable'),
        (0, 'Low (< -11.1)'),
        (1, 'Moderate (-11.1 ~ -5.7)'),
        (2, 'High (≥ -5.7)')
    ]
    create_radiobutton_group('Glacier temperature category (°C/year):', temp_var, temp_options, 3)

    prcp_var = tk.IntVar(value=-1)
    prcp_options = [
        (-1, 'Not Applicable'),
        (0, 'Low (< 630)'),
        (1, 'Moderate (630 ~ 1300)'),
        (2, 'High (≥ 1300)')
    ]
    create_radiobutton_group('Glacier precipitation category (mm/year):', prcp_var, prcp_options, 4)

    area_var = tk.IntVar(value=-1)
    area_options = [
        (-1, 'Not Applicable'),
        (0, 'Low (< 0.13)'),
        (1, 'Moderate (0.13 ~ 0.51)'),
        (2, 'High (≥ 0.51)')
    ]
    create_radiobutton_group('Glacier area category (km²):', area_var, area_options, 5)

    slope_var = tk.IntVar(value=-1)
    slope_options = [
        (-1, 'Not Applicable'),
        (0, 'Low (< 20)'),
        (1, 'Moderate (20 ~ 27)'),
        (2, 'High (≥ 27)')
    ]
    create_radiobutton_group('Glacier slope category (°):', slope_var, slope_options, 6)

    aspect_var = tk.IntVar(value=-1)
    aspect_options = [
        (-1, 'Not Applicable'),
        (0, 'Northern Hemisphere\nSouth-facing slopes'),
        (1, 'Northern Hemisphere\nNorth-facing slopes'),
        (2, 'Southern Hemisphere\nSouth-facing slopes'),
        (3, 'Southern Hemisphere\nNorth-facing slopes')
    ]
    create_radiobutton_group('Glacier aspect category:', aspect_var, aspect_options, 7)

    # =========================
    # Classification Tree
    # =========================
    def export(matched_row, base_cols, file_path, value_map):
        header = pd.DataFrame([base_cols])
        output_row = []
        for col in base_cols:
            if col in value_map:
                output_row.append(value_map[col])
            else:
                output_row.append(matched_row[col])
        row_df = pd.DataFrame([output_row])

        for prefix in ['AAR_', 'THAR_', 'AABR_']:
            cols = [col for col in matched_row.index if col.startswith(prefix)]
            header = pd.concat([header, pd.DataFrame([cols])], axis=1)
            row_df = pd.concat([row_df, pd.DataFrame([matched_row[cols].values])], axis=1)

        header.to_csv(file_path, header=False, index=False, encoding='utf-8-sig')
        row_df.to_csv(file_path, mode='a', header=False, index=False, encoding='utf-8-sig')

    def filter_and_save():
        is_debris = debris_var.get()
        is_icecap = icecap_var.get()
        is_tidewater = tidewater_var.get()

        glacier_type_code = is_debris * 1 + is_icecap * 2 + is_tidewater * 4

        inputs = {
            'temperature_category': temp_var.get(),
            'precipitation_category': prcp_var.get(),
            'area_category': area_var.get(),
            'slope_category': slope_var.get(),
            'aspect_category': aspect_var.get()
        }

        all_vars = [
            'temperature_category',
            'precipitation_category',
            'area_category',
            'slope_category',
            'aspect_category'
        ]

        exclude_vars = [v for v in all_vars if inputs[v] == -1]
        include_vars = [v for v in all_vars if inputs[v] != -1]

        matched_row = None
        matched_level = None

        classification_dir = '/Users/wyan0065/Desktop/AAR-disequilibrium/AABR/data_and_code/data/classificationtree/'

        def try_file(level, exclude_vars, used_vars):

            if (len(exclude_vars) !=0) & (len(exclude_vars) != 5):
                exclude_str = '_'.join(exclude_vars)
                if len(used_vars) == 0:
                    filename = f'Classification_exclude_{exclude_str}_Level{level}_glacier_type.csv'
                else:
                    used_str    = '_'.join(used_vars)
                    filename = f'Classification_exclude_{exclude_str}_Level{level}_glacier_type_{used_str}.csv'
            else:
                if len(used_vars) == 0:
                    filename = f'Classification_all_Level{level}_glacier_type.csv'
                else:
                    used_str = '_'.join(used_vars)
                    filename = f'Classification_all_Level{level}_glacier_type_{used_str}.csv'

            filepath = os.path.join(classification_dir, filename)
            print(f'Trying file: {filename}')

            if not os.path.isfile(filepath):
                return None, None

            df = pd.read_csv(filepath, encoding='utf-8-sig')
            df_match = df[df['glacier_type'] == glacier_type_code]

            for col in used_vars:
                if col in df_match.columns:
                    df_match = df_match[df_match[col] == inputs[col]]

            df_match = df_match[df_match['AAR_n'] >= 5]
            if not df_match.empty:
                return df_match.iloc[0].copy(), level
            return None, None

        if len(include_vars) != 0:
            for level in range(len(include_vars)+1, 0, -1):
                used_vars = include_vars[:level-1]
                matched_row, matched_level = try_file(level, exclude_vars, used_vars)
                if matched_row is not None:
                    break
        else:
            used_vars = include_vars
            level = 1
            matched_row, matched_level = try_file(level, exclude_vars, used_vars)

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

        file_path = filedialog.asksaveasfilename(
            initialfile='Results.csv',
            defaultextension='.csv',
            filetypes=[('CSV files', '*.csv')]
        )
        if not file_path:
            return

        export(matched_row, base_cols, file_path, value_map)
        messagebox.showinfo('Success', f'✅ Used Level {matched_level} Saved to:\n{file_path}')

    tk.Button(
        main_frame,
        text='Export results as CSV',
        command=filter_and_save
    ).grid(row=8, column=2, columnspan=2, pady=12)

    root.mainloop()


if __name__ == '__main__':
    main()
