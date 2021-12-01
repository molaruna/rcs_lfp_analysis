#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Process the outputs of Matlab's ProcessRCS() function
Outputs: power spectra plots

@author: mariaolaru
"""
import math
from preproc.preprocess_funcs import *
from plts.plot_funcs import *
from proc.process_funcs import *

def plot_rcs(dir_name, out_name, step_size_init = math.nan, alpha = 1):
    """
    Parameters
    ----------
    dir_name : parent directory of "Sessions" Folders   
    
    out_name : title of plot

    step_size_init : amount of time, in seconds, of each power spectra, optional (for superimposing)        

    alpha : opacity of PSD line, floating point between [0,1], optional

    Returns
    -------
    power spectra in parent directory -> plots

    """
    #modify funcs to get and set
    [msc, gp] = preprocess_settings(dir_name)
    md = preprocess_data(dir_name, msc, gp) #separate fn b/c can take much longer time to process data
    
    buffer = 5 #add buffer (in seconds) to beginning
    min_length = 5 #minimal time (in seconds) for power spectra
    
    #Find ts ranges
    md_ts_tail = md['timestamp_unix'][len(md)-1]
    ts_range = find_ts_range(msc, buffer, min_length, md_ts_tail)
    
    for i in range(0, len(ts_range)):   
    #Process each subset range of ts
        mscs = subset_msc(msc, ts_range.iloc[i, :])
        qc_msc(mscs)
        mscs = mscs.head(1).reset_index()
        
        fs = int(mscs['ch1_sr'].iloc[0])
        [mds, tt] = subset_md(md, mscs, fs, ts_range.iloc[i, :])
        
        if (len(mds) < min_length * 1000):
            continue 
        
        if math.isnan(step_size_init):
            step_size = len(mds)/fs         
            alpha = 1
        else: 
            step_size = step_size_init
            
        [mdsm, step_size] = melt_mds(mds, fs, out_name, step_size)
        df_psd = convert_psd(mdsm, fs, out_name)

        
        #Plot psd
        plot_title = make_plot_title(out_name, step_size, mscs, tt)        
        plot_PSD(mscs, df_psd, fs, out_name, plot_title, gp, alpha)

