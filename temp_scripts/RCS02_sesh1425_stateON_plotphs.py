#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul  5 11:35:21 2021

@author: mariaolaru
"""
import numpy as np
from preproc.preprocess_funcs import *
from proc.process_funcs import *
from plts.plot_funcs import *

dir_name = "/Users/mariaolaru/RCS02 Un-Synced Data/SummitData/SummitContinuousBilateralStreaming/RCS02L/"
#modify funcs to get and set
[msc, gp] = preprocess_settings(dir_name)
md = preprocess_data(dir_name, msc, gp) #separate fn b/c can take much longer time to process data
#md = pd.read_csv(gp + '/' + 'RCS02_meta_data.csv')

#Find timestamps of interest for analysis
sesh_of_int = 1558130697111
ts_min_msc = msc[msc['session_id']==sesh_of_int]['timestamp_unix'].iloc[0]
ts_min_md = md[md['session_id'] == sesh_of_int]['timestamp_unix'].min()
ts_min = np.array([ts_min_msc, ts_min_md]).min()
ts_max = md[md['session_id'] == sesh_of_int]['timestamp_unix'].max()

ts_range = [ts_min, ts_max]

#Check that settings do not change within these timestamps
qc_msc(msc, ts_range)

#Subset tables based on timestamps of interest
mscs = subset_msc(msc, ts_range)

fs = int(mscs['ch1_sr'].iloc[0])
[mds, tt] = subset_md(md, mscs, fs, ts_range)

#Make file name to save outputs
out_name = name_file(mscs, gp) + '_ON'

#Create PSD dfs in specified time intervals
step_size = 30
mdsm = melt_mds(mds, fs, out_name, step_size)
df_psd = convert_psd(mdsm, fs, out_name)

#Create PHS dfs
freq_thresh = np.array([60, 90])
df_phs = compute_phs(df_psd, fs, freq_thresh, out_name)
  
#Plot PHS
#sesh_id = mscs['session_id'].iloc[0]
#out_name = 'phs_' + str(sesh_id)
#plot_title = make_plot_title(out_name, step_size, mscs, tt)
#plot_phs_psd(mscs, df_psd, freq_thresh, freq_comp, fs, out_name, plot_title, gp) #BUILD THIS OUT

#Create msc dfs 
df_msc = compute_msc(mdsm, fs, out_name) #pick sub-cort and cort channels w/ highest avg phs

#temp make plots
ss = df_msc[(df_msc['ch_cort'] == 4) & (df_msc['ch_subcort'] == 2) & (df_msc['step'] == 2)]
ss = ss.reset_index(drop=True)

plt.plot(ss.loc[:, 'Cxy'])

out_fp = '/Users/mariaolaru/google_drive/UCSF_Neuroscience/starr_lab_MO/studies/gamma_entrainment/figures/aim1_plots'            
plt.savefig(out_fp + '/ON_msc_example.svg')


