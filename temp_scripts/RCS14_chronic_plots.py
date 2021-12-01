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

path_list = "/Users/mariaolaru/google_drive/UCSF_Neuroscience/starr_lab_MO/studies/gamma_entrainment/meta_data/RCS14_chronic_session_list_temp.txt"

#modify funcs to get and set
[msc, gp] = preprocess_settings(path_list)
#md = preprocess_data(path_list, msc) #separate fn b/c can take much longer time to process data
md = pd.read_csv(gp + '/' + 'RCS14_meta_data.csv', header=None)

# Create longitudinal PSDs for session with STIM OFF
indx_int = 30
indx_int_last = 32
sesh_id = msc['session_id'][indx_int]
date = msc['timestamp'][indx_int]

md = preprocess_data_chronic(path_list, msc, sesh_id)
mds = subset_md_chronic(md)

step_size = 120 #120s increments for PSD plot lines
fs = msc['ch1_sr'][indx_int] #HOW COME THERE IS STIM FREQ W/ STIM STATE 0?
dfp = melt_mds_chronic(mds, step_size, fs)
df_psd = dfp_psd_chronic(dfp, fs)

out_name = 'chronic_longPSD' + ' session: ' + sesh_id + '; date ' + str(date)
plot_title = get_plot_title_chronic(msc, indx_int, step_size, out_name, indx_int_last)

plot_PSD_long_chronic(md, msc, gp, indx_int, step_size, out_name, plot_title, df_psd)







indx_int = 888
indx_int_last = 890
sesh_id = msc['session_id'][indx_int]
date = msc['timestamp'][indx_int]

md = preprocess_data_chronic(path_list, msc, sesh_id)
mds = subset_md_chronic(md)

step_size = 120 #120s increments for PSD plot lines
fs = msc['ch1_sr'][indx_int] #HOW COME THERE IS STIM FREQ W/ STIM STATE 0?
dfp = melt_mds_chronic(mds, step_size, fs)
df_psd = dfp_psd_chronic(dfp, fs)

out_name = 'chronic_longPSD' + ' session: ' + sesh_id + '; date ' + str(date)
plot_title = get_plot_title_chronic(msc, indx_int, step_size, out_name, indx_int_last)

plot_PSD_long_chronic(md, msc, gp, indx_int, step_size, out_name, plot_title, df_psd)



# Create longitudinal PSDs for session with STIM ON low freq day
indx_int = 872
indx_int_last = 875
sesh_id = msc['session_id'][indx_int]
date = msc['timestamp'][indx_int]

md = preprocess_data_chronic(path_list, msc, sesh_id)
mds = subset_md_chronic(md)

step_size = 120 #120s increments for PSD plot lines
fs = msc['ch1_sr'][indx_int] #HOW COME THERE IS STIM FREQ W/ STIM STATE 0?
dfp = melt_mds_chronic(mds, step_size, fs)
df_psd = dfp_psd_chronic(dfp, fs)

out_name = 'chronic_longPSD' + ' session: ' + sesh_id + '; date ' + str(date)
plot_title = get_plot_title_chronic(msc, indx_int, step_size, out_name, indx_int_last)

plot_PSD_long_chronic(md, msc, gp, indx_int, step_size, out_name, plot_title, df_psd)




