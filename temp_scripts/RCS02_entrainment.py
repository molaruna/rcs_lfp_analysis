#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 21 13:12:25 2021

@author: mariaolaru
"""
import pandas as pd
pd.options.display.float_format = '{:.2f}'.format
#pd.reset_option('display.float_format')
from preproc.preprocess_funcs import *
from plts.plot_funcs import *
from proc.process_funcs import *

dir_name = '/Users/mariaolaru/Documents/temp/RCS02/RCS02L/RCS02L_ge_onmed/'
#modify funcs to get and set
[msc, df_notes, gp] = preprocess_settings(dir_name)
md = preprocess_data(dir_name, msc, gp) #separate fn b/c can take much longer time to process data

mds = remove_stim_artifact(md, msc)

min_dur = 15 #minimum duration for each recording
mdb = remove_short_recordings(mds, msc, min_dur)

df_linked = link_data(msc, mdb, gp) #create combined dataframe
sr = 250
df_psds = convert_psd_wide(df_linked, sr) #psds for all combinations of freq & amp


"""
for i in range(len(df_psds)):
    print(i)
    names = np.array([*df_psds.keys()])    
    df_out = df_psds[names[i]]
    for j in range(len(df_out)):
        print(j)
        names_2 = np.array([*df_out.keys()])  
        df_out_2 = df_out[names_2[j]]
        out_dir = '/Users/mariaolaru/Documents/temp/RCS02/RCS02L/RCS02L_ge_onmed/psds/'
        out_name = 'psd_' + 'stimfreq' + str(names[i]) + '_stimamp' + str(names_2[j]) + '.csv'
        df_out_2.to_csv(out_dir + out_name)
"""

plot_PSD_wide(df_psds, gp)

df_phs = make_phs_wide(df_psds, gp)
df_entrain = get_entrainment_score(df_phs)

df_entraint = df_entrain[(df_entrain['stim_freq'] == 130.2) | (df_entrain['stim_freq'] == 140.1) | (df_entrain['stim_freq'] == 150.6)| (df_entrain['stim_freq'] == 158.7) | (df_entrain['stim_freq'] == 169.5)] 
df_entraint = df_entraint[(df_entraint['stim_amp'] == 0.7) | (df_entraint['stim_amp'] == 1.0) | (df_entraint['stim_amp'] == 1.3) | (df_entraint['stim_amp'] == 1.6) | (df_entraint['stim_amp'] == 1.9) | (df_entraint['stim_amp'] == 2.2) | (df_entraint['stim_amp'] == 2.5) | (df_entraint['stim_amp'] == 2.8) | (df_entraint['stim_amp'] == 3.1)]

plot_entrainment(df_entraint, gp)
