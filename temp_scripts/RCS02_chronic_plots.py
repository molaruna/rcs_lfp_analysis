#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul  5 11:35:21 2021

@author: mariaolaru
"""
import numpy as np
from preprocess_script import *
from plot_script import *


path_list = "/Users/mariaolaru/google_drive/UCSF_Neuroscience/starr_lab_MO/studies/gamma_entrainment/meta_data/RCS02L_athome_session_list.txt"

#modify funcs to get and set
[msc, gp] = preprocess_settings(path_list)
md = preprocess_data(path_list, msc) #separate fn b/c can take much longer time to process data
#md = pd.read_csv(gp + '/' + 'RCS02_meta_data.csv')

#Find timestamps of interest for analysis
sesh_of_int = msc['session_id'].unique()[9]
ts_min_msc = msc[msc['session_id']==sesh_of_int]['timestamp_unix'].iloc[0]
ts_min_md = md[md['session_id'] == sesh_of_int]['timestamp_unix'].min()
ts_min = np.array([ts_min_msc, ts_min_md]).min()
ts_max = md[md['session_id'] == sesh_of_int]['timestamp_unix'].max()

ts_min = 1558206764000 #start in off state
ts_max = 1558207244000 #add 8min 

ts_range = [ts_min, ts_max]

#Check that settings do not change within these timestamps
qc_msc(msc, ts_range)

#Subset tables based on timestamps of interest
mscs = subset_msc(msc, ts_range)
mscs = mscs.head(1)

fs = int(mscs['ch1_sr'].iloc[0])
[mds, tt] = subset_md(md, mscs, fs, ts_range)

#Create PSD dfs in specified time intervals
step_size = 30
mdsm = melt_mds(mds, fs, step_size)
df_psd = convert_psd(mdsm, fs)

#Plot PSDs  
sesh_id = mscs['session_id'].iloc[0]
out_name = 'psd ' + str(sesh_id)
plot_title = make_plot_title(out_name, step_size, mscs, tt)
plot_PSD_long_chronic(md, msc, gp, indx_int, step_size, out_name, plot_title, df_psd)

#Create PHS dfs for each PSD
freq_comp = np.array([5, 5])
freq_thresh = np.array([60, 90])
df_phs = compute_phs(df_psd, fs, freq_comp, freq_thresh)
  
#Plot PHS of PSDs
sesh_id = mscs['session_id'].iloc[0]
out_name = 'phs_' + str(sesh_id)
plot_title = make_plot_title(out_name, step_size, mscs, tt)
plot_phs_psd(mscs, df_psd, freq_thresh, freq_comp, fs, out_name, plot_title, gp) #BUILD THIS OUT
