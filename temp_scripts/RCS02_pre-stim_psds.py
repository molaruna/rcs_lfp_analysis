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

sr = 250
[msc_ds, md_ds] = downsample_data(msc, md, sr) #downsamples separately for each msc label of sr

contacts = [msc['ch0_sense_contacts'].unique()[0], msc['ch1_sense_contacts'].unique()[0], msc['ch2_sense_contacts'].unique()[0], msc['ch3_sense_contacts'].unique()[0]]
#df_psd = convert_psd_long_old(md_ds, gp, contacts, 120, 119, sr, 'total')
df_psd_periodic = convert_psd_long_old(md_ds, gp, contacts, 120, 119, sr, 'periodic')
df_psd_aperiodic = convert_psd_long_old(md_ds, gp, contacts, 120, 119, sr, 'aperiodic')

md_ds = md_ds.rename({'ch0_mv': contacts[0], 'ch1_mv': contacts[1], 'ch2_mv': contacts[2], 'ch3_mv': contacts[3]}, axis = 1)
df_coh = convert_coh_long_old(md_ds, contacts, gp, 120, 119, sr)

#df_phs = make_phs_old(md_ds, contacts, sr, [4, 100], 120, 119, gp)
#plot_phs(df_phs, gp, subj_id)
