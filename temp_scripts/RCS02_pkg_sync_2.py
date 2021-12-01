#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 21 11:38:47 2021

@author: mariaolaru

Individual freq correlations
"""

import proc.rcs_pkg_sync_funcs as sync
import numpy as np
import pandas as pd
    
pkg_dir = '/Users/mariaolaru/Documents/temp/RCS02/RCS02L/RCS02L_ge_onmed/RCS02R_pkg_data/'
fp_phs = '/Users/mariaolaru/Documents/temp/RCS02/RCS02L/RCS02L_pre-stim/RCS02L_pre-stim_phs.csv'
fp_psd = '/Users/mariaolaru/Documents/temp/RCS02/RCS02L/RCS02L_pre-stim/RCS02L_pre-stim_psd_partial_2m_wholemin.csv'
fp_notes = '/Users/mariaolaru/Documents/temp/RCS02/RCS02L/RCS02L_pre-stim/RCS02L_pre-stim_meta_session_notes.csv'
contacts = ['+2-0', '+3-1', '+10-8', '+11-9']

[df_pkg, start_time, stop_time] = sync.preproc_pkg(pkg_dir)
df_phs = sync.preproc_phs(fp_phs, start_time, stop_time)

df_psd = sync.preproc_psd(fp_psd, start_time, stop_time)

df_notes = sync.preproc_notes(fp_notes, start_time, stop_time)
df_dys = sync.find_dyskinesia(df_notes)
df_meds = sync.get_med_times()

#Processing
freq_bands = np.array(df_psd.columns.levels[0])
df_corr = pd.DataFrame([])

df_merged = sync.process_dfs(df_pkg, df_phs, df_psd, df_meds, df_dys)
df_merged = sync.add_sleep_col(df_merged)

for i in range(len(freq_bands)):
    freq_band = freq_bands[i]
    print("freq band: " + str(freq_band) + "hz")



    sync.plot_pkg_sync(df_merged, freq_band)
    
    
    #get corr coefs
    df_vars = df_merged.loc[:, ['BK_norm', 'DK_norm', (contacts[3]), (contacts[2]), (contacts[1]), (contacts[0])]]
    
    #df_vars = df_mergedfb.loc[:, ['BK_norm', 'DK_norm', 'max_amp_beta', 'max_amp_gamma']]
    df_corr_ind = df_vars.corr()
    
    
    df_corr_ind = df_corr_ind.round(3)
    df_corr_ind['freq_band'] = freq_band
    df_corr = pd.concat([df_corr, df_corr_ind])

out_fp = '/Users/mariaolaru/Documents/temp/RCS02/RCS02L_pkg_rcs' + '/RCS02_corrs' + '.csv'
df_corr.to_csv(out_fp)
print(df_corr)
print('/n')

#get correlation plots
sync.plot_corrs(df_corr, 'DK_norm')
sync.plot_corrs(df_corr, 'BK_norm')

