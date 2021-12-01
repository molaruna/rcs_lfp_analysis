#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul  5 11:35:21 2021

@author: mariaolaru
"""
import numpy as np
import pandas as pd
from preproc.preprocess_funcs import *
from plts.plot_funcs import *
from proc.process_funcs import *

parent_dir = "/Users/mariaolaru/Box/RC-S_Studies_Regulatory_and_Data/Patient In-Clinic Data/RCS10/study_visits/v07_gamma_entrainment/SCBS/RCS10L/"

#modify funcs to get and set
[msc, gp] = preprocess_settings(parent_dir)
md = preprocess_data(parent_dir, msc, gp) #separate fn b/c can take much longer time to process data

#Step 1: manually subset data into timechunks > 45s w/ X amp X freq
phs_final = pd.DataFrame()
psd_final = pd.DataFrame()
buffer = 15*1000 #add 15s buffer time to beginning

def proc_tsofint(ts_min, ts_max, buffer, md, msc, gp, phs_final, psd_final, out_name_custom):
    ts_min = ts_min + buffer
    ts_max = ts_max - 1
    ts_range = [ts_min, ts_max]
    mscs = subset_msc(msc, ts_range)
    mscs = mscs.head(1).reset_index()
    fs = int(mscs['ch1_sr'].iloc[0])
    [mds, tt] = subset_md(md, mscs, fs, ts_range)
    #Make file name to save outputs
    out_name = name_file(mscs, gp) + '_testing'

    #Create PSD dfs in specified time intervals
    step_size = 30
    mdsm = melt_mds(mds, fs, out_name, step_size)
    df_psd = convert_psd(mdsm, fs, out_name)

    #Create PHS dfs
    freq_thresh = np.array([60, 90])
    df_phs = compute_phs(df_psd, fs, freq_thresh, out_name)
    
    #Plot phs
    out_name = 'RCS10L_phs_' + out_name_custom
    plot_title = make_plot_title(out_name, step_size, mscs, tt)
    plot_phs(df_psd, df_phs, out_name, plot_title, gp)

    #add additional info for final table
    df_phs['stim_amp'] = mscs.loc[0, 'amplitude_ma']
    df_phs['stim_freq'] = mscs.loc[0, 'stimfrequency_hz']
    df_psd['stim_amp'] = mscs.loc[0, 'amplitude_ma']
    df_psd['stim_freq'] = mscs.loc[0, 'stimfrequency_hz']
   
    phs_final = pd.concat([phs_final, df_phs]) 
    psd_final = pd.concat([psd_final, df_psd])
    print(tt*60)       
    return [phs_final, psd_final]
 
#ts of int for 4.8 / 149.8 (a lot of gamma entrainment)
ts_min = 1620065700015
ts_max = 1620066972791
[phs_final, psd_final] = proc_tsofint(ts_min, ts_max, buffer, md, msc, gp, phs_final, psd_final, '4.8_149.8')

#ts of int for 0mA / 149.8 Hz (none)
ts_min = 1620067224550
ts_max = 1620067286824
[phs_final, psd_final] = proc_tsofint(ts_min, ts_max, buffer, md, msc, gp, phs_final, psd_final, '0_149.8')

#ts of int for 0mA / 130Hz (none)
ts_min = 1620067635902
ts_max = 1620067705955
[phs_final, psd_final] = proc_tsofint(ts_min, ts_max, buffer, md, msc, gp, phs_final, psd_final, '0_130')

#ts of int for 2.5 / 130Hz (none)
ts_min = 1620067705955
ts_max = 1620067890122
[phs_final, psd_final] = proc_tsofint(ts_min, ts_max, buffer, md, msc, gp, phs_final, psd_final, '2.5_130')

#ts of int for 6.5 / 130Hz (a lot of gamma entrainment)
ts_min = 1620068090898
ts_max = 1620068262643
[phs_final, psd_final] = proc_tsofint(ts_min, ts_max, buffer, md, msc, gp, phs_final, psd_final, '6.5_130')

#ts of int for 2.5mA / 138.9Hz
ts_min = 1620068373894
ts_max = 1620068453495
[phs_final, psd_final] = proc_tsofint(ts_min, ts_max, buffer, md, msc, gp, phs_final, psd_final, '2.5_138.9')

#ts of int for 5mA / 138.9 (gamma entrainment)
ts_min = 1620068453495
ts_max = 1620068671005
[phs_final, psd_final] = proc_tsofint(ts_min, ts_max, buffer, md, msc, gp, phs_final, psd_final, '5_138.9')

#ts of int for 5.5mA / 138.9 (gamma entrainment)
ts_min = 1620068671005
ts_max = 1620068723203
[phs_final, psd_final] = proc_tsofint(ts_min, ts_max, buffer, md, msc, gp, phs_final, psd_final, '5.5_138.9')

#ts of int for 6mA / 138.9 (gamma entrainment)
ts_min = 1620068723203
ts_max = 1620068981938 
[phs_final, psd_final] = proc_tsofint(ts_min, ts_max, buffer, md, msc, gp, phs_final, psd_final, '6_138.9')

#ts of int for 2.5 / 149.3 (no gamma entrainment)
ts_min = 1620074230559
ts_max = 1620074297148
[phs_final, psd_final] = proc_tsofint(ts_min, ts_max, buffer, md, msc, gp, phs_final, psd_final, '2.5_149.3')

#ts of int for 5.0 / 149.3 (little entrainment)
ts_min = 1620074297148
ts_max = 1620074365242
[phs_final, psd_final] = proc_tsofint(ts_min, ts_max, buffer, md, msc, gp, phs_final, psd_final, '5_149.3')

#ts of int for 5.5 / 149.3
ts_min = 1620074365242
ts_max = 1620074410431
[phs_final, psd_final] = proc_tsofint(ts_min, ts_max, buffer, md, msc, gp, phs_final, psd_final, '5.5_149.3')

#ts of int for 6.0 / 149.3
ts_min = 1620074410431
ts_max = 1620074604135
[phs_final, psd_final] = proc_tsofint(ts_min, ts_max, buffer, md, msc, gp, phs_final, psd_final, '6_149.3')

#temporarily add these as ylim min max values of psd plots    
psd_final_ss = psd_final[(psd_final['channel'] == 4) & (psd_final['f_0'] < 100)]
if (psd_final_ss['Pxx_den'].min() < min_amp):
    min_amp = psd_final_ss['Pxx_den'].min()
if (psd_final_ss['Pxx_den'].max() > max_amp):
    max_amp = psd_final_ss['Pxx_den'].max()

#create scatterplot of peak height score vs. amplitude for each freq:
    #freqs: 130Hz, 138.9Hz, 149.8Hz
    #channel: 4
    #all steps
    #maybe can do it all in 1 table w/ groupings for each freq
phs_final_ss = phs_final[(phs_final['channel'] == 4) & ((phs_final['f_max'] == 74.21875) | (phs_final['f_max'] ==70.3125) | (phs_final['f_max'] == 64.453125))]

from matplotlib import pyplot as plt
phs_final_ss_1 = phs_final_ss[(phs_final_ss['stim_freq'] == 130.2)]
phs_final_ss_2 = phs_final_ss[(phs_final_ss['stim_freq'] == 138.9)]
phs_final_ss_3 = phs_final_ss[(phs_final_ss['stim_freq'] == 149.3)]

c = 'stim_amp'
plt.scatter(phs_final_ss_1[c], phs_final_ss_1['phs'], label = "stim freq: 130Hz")
plt.scatter(phs_final_ss_2[c], phs_final_ss_2['phs'], label = "stim freq: 140Hz")
plt.scatter(phs_final_ss_3[c], phs_final_ss_3['phs'], label = "stim freq: 150Hz")
plt.xlim([4.5, 7.0])
plt.legend()

out_plot_fp = gp + '/plots/RCS10L_phs_stimamp'
plt.savefig(out_plot_fp + ".svg")



