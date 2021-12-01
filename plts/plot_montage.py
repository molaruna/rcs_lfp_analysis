#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import preproc.preprocess_funcs as preproc
import proc.process_funcs as proc
import plts.plot_funcs as plot

def plot_montage(dir_name, labels):

    [msc, df_notes, gp] = preproc.preprocess_settings(dir_name)
    md = preproc.preprocess_data(dir_name, msc, gp) #separate fn b/c can take much longer time to process data
    msc = preproc.label_montage(msc, labels) #requires 1 label/montage session file
    df =  preproc.link_data(msc, md) #create combined dataframe
    
    [psds_500, psds_1000] = proc.convert_psd_montage(df)
    
    plot.plot_PSD_montage_channels(psds_500, msc, gp, 500) 
    plot.plot_PSD_montage_channels(psds_1000, msc, gp, 1000) 
    
    plot.plot_PSD_montage_conditions(psds_500, msc, gp, 500)
    plot.plot_PSD_montage_conditions(psds_1000, msc, gp, 1000)
