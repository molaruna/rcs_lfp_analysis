#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct  6 13:20:19 2021

@author: mariaolaru
"""

import preproc.preprocess_funcs as preproc
import plts.plot_funcs as plts
import proc.process_funcs as proc
 
dir_name = '/Users/mariaolaru/Documents/temp/RCS12/RCS12L/RCS12L_pre-stim/3day_sprint/montage_pkg'
#dir_name = '/Users/mariaolaru/Documents/temp/RCS12/RCS12L/RCS12L_pre-stim/montage'
#dir_name = '/Users/mariaolaru/Documents/temp/GRCS02/GRCS02L/GRCS02L_pre-stim/3day_sprint/montage_3/'
#dir_name = '/Users/mariaolaru/Documents/temp/GRCS01/GRCS01R/GRCS01R_pre-stim/3day_sprint/montage_1/'

[msc, df_notes, gp] = preproc.preprocess_settings(dir_name)
md = preproc.preprocess_data(dir_name, msc, gp) #separate fn b/c can take much longer time to process data
subj_id = str(msc['subj_id'][0]) + " " + str(msc['implant_side'][0])

sr = 250
[msc_ds, md_ds] = proc.downsample_data(msc, md, sr) #downsamples separately for each msc label of sr

#remove data with stim_on
[msc_ds_rm, md_ds_rm] = proc.rm_stim(msc_ds, md_ds, 'stimON') #downsamples separately for each msc label of sr

i = 0
contacts = [msc['ch0_sense_contacts'].unique()[i], msc['ch1_sense_contacts'].unique()[i], msc['ch2_sense_contacts'].unique()[0], msc['ch3_sense_contacts'].unique()[0]]

md_ds = md_ds.rename({'ch0_mv': contacts[0], 'ch1_mv': contacts[1], 'ch2_mv': contacts[2], 'ch3_mv': contacts[3]}, axis = 1)

#md = md.rename({'ch0_mv': contacts[0], 'ch1_mv': contacts[1], 'ch2_mv': contacts[2], 'ch3_mv': contacts[3]}, axis = 1)

df_psd = proc.convert_psd_long_old(md_ds_rm, gp, contacts, 120, 119, sr, 'total')
#df_psd = proc.convert_psd_long_old(md, gp, contacts, 120, 119, sr, 'total')
#df_psd_periodic = convert_psd_long_old(md_ds, gp, contacts, 120, 119, sr, 'periodic')
#df_psd_aperiodic = convert_psd_long_old(md_ds, gp, contacts, 120, 119, sr, 'aperiodic')

#df_coh = convert_coh_long_old(md_ds, contacts, gp, 120, 119, sr)

#df_phs = make_phs_old(md_ds, contacts, sr, [4, 100], 120, 119, gp)
#plot_phs(df_phs, gp, subj_id)


#Plot longitudinal PSDs
#plts.plot_long_psd(contacts, df_psd_init, 0.01, gp)
plts.plot_long_psd(contacts, df_psd, 0.01, gp)
