#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct  6 13:20:19 2021

@author: mariaolaru
"""

from preproc.preprocess_funcs import *
from plts.plot_funcs import *
from proc.process_funcs import *
from fooof import FOOOF

dir_name = '/Users/mariaolaru/Documents/temp/RCS02/RCS02L/RCS02L_pre-stim/'

[msc, df_notes, gp] = preprocess_settings(dir_name)
md = preprocess_data(dir_name, msc, gp) #separate fn b/c can take much longer time to process data

subj_id = msc['subj_id'][0] + " " + msc['implant_side'][0]
contacts = [msc['ch0_sense_contacts'].unique()[0], msc['ch1_sense_contacts'].unique()[0], msc['ch2_sense_contacts'].unique()[0], msc['ch3_sense_contacts'].unique()[0]]

[msc_ds, md_ds] = downsample_data(msc, md, 250) #downsamples separately for each msc label of sr

#pkg start-stop times
start_time = 1557928800000
stop_time = 1559174280000

df_temp = md_ds[(md_ds['timestamp'] >= start_time) & (md_ds['timestamp'] <= stop_time)]
#df_linked = link_data(msc_ds, md_ds, gp) #create combined dataframe
#df_linked = link_data_wide(msc_ds, md_ds, gp)
#df_linked = link_data(msc_ds, md_ds, gp) #create combined dataframe
# sr = int(df_linked['sr'][0])

#create psds for 10min duration, 9.5min overlap 
df_psd = convert_psd_long_old(df_temp, gp, contacts, 30, 0, 250)

#df_phs = make_phs_old(md_ds, contacts, sr, [4, 100], 30, 0, gp)

#df_phs = make_phs(df_linked, sr, [4, 100], 30, 0, gp)

#plot_phs(df_phs, gp, subj_id)
