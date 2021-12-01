#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 21 11:38:47 2021

@author: mariaolaru
"""

import pandas as pd
import numpy as np
import glob
import os
from matplotlib import pyplot as plt
pd.options.display.float_format = '{:.2f}'.format
#pd.reset_option('display.float_format')


def import_pkg_files(pkg_dir):
    paths = glob.glob(pkg_dir + 'scores*')
    gp = os.path.abspath(os.path.join(paths[0], "../..")) 
    subj_id = os.path.basename(gp)
    
    md_fp = pkg_dir + subj_id + '_pkg_data_combined.csv'
    md = pd.DataFrame([])
    
    if (os.path.exists(md_fp)):
        md = pd.read_csv(md_fp, header=0)
    else:    
        for i in range(0, len(paths)):
            p = paths[i]
            df = pd.read_csv(p)
            df['timestamp'] = pd.to_datetime(df['Date_Time']).astype(int) / 10**9
            md = pd.concat([md, df])
    
        md = md.sort_values('timestamp')
        md = md.reset_index(drop=True)
        md.to_csv(md_fp, index = False, header=True)
        md['timestamp'] = md['timestamp'].astype(int)
        md['timestamp'] = md['timestamp']*1000
    return md

def normalize_data(x, min_x, max_x):
    x_norm = (x-min_x)/(max_x-min_x)
    return x_norm

def remove_outliers(col):
    x = col.copy()
    y = x.median() + 5*x.std()
    indx = x[x > y].index
    x[indx] = np.nan
    return x

def preproc_pkg(pkg_dir):
    pkg_df = import_pkg_files(pkg_dir)
    
    pkg_df['timestamp'] = pkg_df['timestamp'].astype(int)*1000
    pkg_df['timestamp'] = pkg_df['timestamp'] +  60000 #add 1min to end time
    pkg_df['timestamp'] = pkg_df['timestamp'] + 25200000 #add 7hr to end time (b/c Datetime in PT, not GMT)
       
    #Include data where watch is on wrist, from even indices
    pkg_df = pkg_df[(pkg_df['Off_Wrist'] == 0)] 
    
    start_time = pkg_df['timestamp'].head(1).values[0]
    stop_time = pkg_df['timestamp'].tail(1).values[0]

    return [pkg_df, start_time, stop_time]

def preproc_phs(phs_fp, start_time, stop_time):
    phs_df = pd.read_csv(phs_fp)
    phs_df = phs_df.rename({'timestamp_end': 'timestamp'}, axis = 1)
    phs_df['timestamp'] = phs_df['timestamp'].astype(int)
    #subset data
    phs_df = phs_df[(phs_df['timestamp'] > start_time) & (phs_df['timestamp'] < stop_time)]
    
    #create wide phs df
    phs_dfw = phs_df.pivot_table(index = ['timestamp', 'contacts'], values = ['max_amp'], columns = ['band'])
    phs_dfw.columns = [f'{x}_{y}' for x,y in phs_dfw.columns]
    phs_dfw = phs_dfw.reset_index(level=['timestamp', 'contacts'])
    phs_dfw = phs_dfw.sort_values('timestamp')
    
    phs_dfwb = phs_dfw[phs_dfw['contacts'] == '+2-0']
    phs_dfwb = pd.concat([phs_dfwb['timestamp'], phs_dfwb['max_amp_beta']], axis = 1)
    
    phs_dfwg = phs_dfw[phs_dfw['contacts'] == '+2-0']
    phs_dfwg = pd.concat([phs_dfwg['timestamp'], phs_dfwg['max_amp_gamma']], axis = 1)

    #Merge phs tables
    phs_merged = pd.merge(phs_dfwb, phs_dfwg, how = 'inner', on = 'timestamp')
    return phs_merged

def merge_pkg_phs(pkg_df, phs_df):
    df_merged = pd.merge(pkg_df, phs_df, how = 'inner', on = 'timestamp')
    df_merged = df_merged.sort_values('timestamp')
    df_merged = df_merged.reset_index(drop=True)
    return df_merged

def find_nan_chunks(col, num_nans):
    #find nan values
    x = col.isnull().astype(int).groupby(col.notnull().astype(int).cumsum()).cumsum()
    #find indices of nan vals >= num_nan
    indx_remove = x[x >= num_nans].index
    return indx_remove    

pkg_dir = '/Users/mariaolaru/Documents/temp/RCS02/RCS02L/RCS02L_ge_onmed/RCS02R_pkg_data/'
[pkg_df, start_time, stop_time] = preproc_pkg(pkg_dir)

phs_fp = '/Users/mariaolaru/Documents/temp/RCS02/RCS02L/RCS02L_pre-stim/RCS02L_pre-stim_phs.csv'
phs_df = preproc_phs(phs_fp, start_time, stop_time)

phs_df['timestamp'] = round(phs_df['timestamp'], -3)/1000
phs_df['timestamp'] = phs_df['timestamp'] * 1000
phs_df['timestamp'] = phs_df['timestamp'].astype(int)

df_merged = merge_pkg_phs(pkg_df, phs_df)

#Processing
df_merged['BK_rev'] = df_merged['BK']*-1
df_merged['BK_rev'] = df_merged['BK_rev'] + (df_merged['BK_rev'].min()*-1)

#Remove outliers from each max_amp column separately
df_merged['DK'] = remove_outliers(df_merged['DK'])
df_merged['BK_rev'] = remove_outliers(df_merged['BK_rev'])

#Normalize data from 0-1
df_merged['BK_norm'] = normalize_data(df_merged['BK_rev'], 0, np.nanmax(df_merged['BK_rev']))
df_merged['DK_norm'] = normalize_data(df_merged['DK'], 0, np.nanmax(df_merged['DK']))
df_merged['max_amp_beta_norm'] = normalize_data(df_merged['max_amp_beta'], np.nanmin(df_merged['max_amp_beta']), np.nanmax(df_merged['max_amp_beta']))
df_merged['max_amp_gamma_norm'] = normalize_data(df_merged['max_amp_gamma'], np.nanmin(df_merged['max_amp_gamma']), np.nanmax(df_merged['max_amp_gamma']))
#df_merged['max_amp_diff_norm'] = df_merged['max_amp_beta_norm'] - df_merged['max_amp_gamma_norm']
#df_merged['max_amp_diff_norm'] = normalize_data(df_merged['max_amp_diff'], 0, np.nanmax(df_merged['max_amp_diff']))

#Remove data with large nan blocks
indx_nan_chunks = find_nan_chunks(df_merged.max_amp_gamma, 3)
df_merged_trim = df_merged.drop(indx_nan_chunks, axis = 0)

indx_nan_chunks_2 = find_nan_chunks(df_merged_trim.DK, 6)
df_merged_trim = df_merged_trim.drop(indx_nan_chunks_2, axis = 0)

#Fill remaining nan values
df_mergedf = df_merged_trim.ffill().reset_index(drop=True)
df_mergedfb = df_mergedf.bfill().reset_index(drop=True)

#Plotting
plt.rcParams["figure.figsize"] = (30,3.5)
plt.plot(np.arange(1, len(df_mergedfb)+1, 1), df_mergedfb['DK_norm'], alpha = 0.7, label = 'PKG-DK', markersize = 1, color = 'steelblue')
#plt.plot(np.arange(1, len(df_mergedfb)+1, 1), df_mergedfb['BK_norm'], alpha = 0.7, label = 'PKG-BK', markersize = 1, color = 'indianred')
plt.plot(np.arange(1, len(df_mergedfb)+1, 1), df_mergedfb['max_amp_gamma_norm'], alpha = 0.7, label = 'RCS-gamma', markersize = 1, color = 'darkorange')
#plt.plot(np.arange(1, len(df_mergedfb)+1, 1), df_mergedfb['max_amp_beta_norm'], alpha = 0.7, label = 'RCS-beta', color = 'mediumpurple', markersize = 1)
#plt.plot(df_mergedfb['timestamp'], df_mergedfb['max_amp_diff_norm'], alpha = 0.7, markersize = 1, label = 'RCS (beta-gamma)')
plt.legend(ncol = 4, loc = 'upper right')
plt.ylabel('scores (normalized)')
plt.xlabel('time (samples)')

df_vars = df_mergedfb.loc[:, ['BK_norm', 'DK_norm', 'max_amp_beta_norm', 'max_amp_gamma_norm']]
df_corr = df_vars.corr()
