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

dir_name = '/Users/mariaolaru/Documents/temp/RCS11L/RCS11L_pre-stim/'

[msc, df_notes, gp] = preprocess_settings(dir_name)
md = preprocess_data(dir_name, msc, gp) #separate fn b/c can take much longer time to process data
subj_id = msc['subj_id'][0]

[msc_ds, df_ds] = downsample_data(msc, md, 250) #downsamples separately for each msc label of sr
sr = int(msc['sr'][0])

df = link_data(msc, df_ds, gp) #create combined dataframe
df_phs = make_phs(df, sr, [4, 100], 30, 0, gp)

plot_phs(df_phs, gp, subj_id)

