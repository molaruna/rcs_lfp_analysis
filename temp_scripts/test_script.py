#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 25 13:48:10 2021

@author: mariaolaru
"""

import numpy as np
import pandas as pd
import math
from preproc.preprocess_funcs import *
from plts.plot_funcs import *
from proc.process_funcs import *

parent_dir = "/Users/mariaolaru/Box/RC-S_Studies_Regulatory_and_Data/Patient In-Clinic Data/RCS10/study_visits/v07_gamma_entrainment/SCBS/RCS10L/"

#modify funcs to get and set
[msc, gp] = preprocess_settings(parent_dir)
md = preprocess_data(parent_dir, msc, gp) #separate fn b/c can take much longer time to process data

buffer = 15 #add buffer (in seconds) to beginning
min_length = 5 #minimal time (in seconds) for power spectra

#Make file name to save outputs
out_name = name_file(mscs, gp) + '_testing_2x'

#Find ts ranges
md_ts_tail = md['timestamp_unix'][len(md)-1]
ts_range = find_ts_range(msc, buffer, min_length, md_ts_tail)

for i in range(0, len(ts_range)):   
#Process each subset range of ts
    mscs = subset_msc(msc, ts_range.iloc[i, :])
    mscs = mscs.head(1).reset_index()
    qc_msc(mscs)
    fs = int(mscs['ch1_sr'].iloc[0])
    [mds, tt] = subset_md(md, mscs, fs, ts_range.iloc[i, :])
    
    if (len(mds) < min_length * 1000):
        continue 
    #Create PSD dfs in specified time intervals
    alpha = 0.1
    if (step_size == math.nan):        
        step_size = len(mds)/fs
        alpha = 1
        
    [mdsm, step_size] = melt_mds(mds, fs, out_name, step_size)
    df_psd = convert_psd(mdsm, fs, out_name)
    
    #Plot psd
    out_name = 'RCS10L_psd_' + str(i)
    plot_title = make_plot_title(out_name, step_size, mscs, tt)
    
    plot_PSD(mscs, df_psd, fs, out_name, plot_title, gp, alpha)

