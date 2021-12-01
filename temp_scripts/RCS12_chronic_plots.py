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

dir_name = "/Users/mariaolaru/RCS12 Un-Synced Data/SummitData/SummitContinuousBilateralStreaming/RCS12L"

[msc, gp] = preprocess_settings(dir_name)
md = preprocess_data(dir_name, msc, gp) 

for i in range(len(msc['session_id'].unique())):
    sesh_id = msc['session_id'].unique()[i]
    print("Processing session: " + str(sesh_id))
    
    ts_range = get_tsofint(sesh_id, msc, md) 

    mscs = subset_msc(msc, ts_range)
    qc_msc(mscs)
    #mscs = mscs.reset_index(drop=True)[0]

    fs = int(mscs['ch1_sr'].iloc[0])
    [mds, tt] = subset_md(md, mscs, fs, ts_range)

    #Make file name to save outputs
    out_name = name_file(mscs, gp)

    #Create PSD dfs in specified time intervals
    step_size = 30 #in seconds
    [mdsm, step_size] = melt_mds(mds, fs, out_name, step_size)
    df_psd = convert_psd(mdsm, fs, out_name)

    subj_id = mscs['subj_id'].iloc[0] + " chronically streamed data"
    plot_title = make_plot_title(subj_id, step_size, mscs, tt)

    plot_PSD_long(mscs, df_psd, fs, plot_title, gp)


