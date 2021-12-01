#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Process the outputs of Matlab's ProcessRCS() function
Outputs: power spectra plots

@author: mariaolaru
"""
import preproc.preprocess_funcs as preproc
import proc.process_funcs as proc
import plts.plot_funcs as plot

dir_name = '/Users/mariaolaru/Documents/temp/WRCS01/WRCS01R/WRCS01R_postop_montage/'
labels = ["medON_stimOFF"]
#labels = ["medON_stimOFF", "medON_stimON"]
#labels = ["medOFF_stimON", "medON_stimON", "medOFF_stimOFF", "medON_stimOFF"]
[msc, df_notes, gp] = preproc.preprocess_settings(dir_name)
md = preproc.preprocess_data(dir_name, msc, gp) #separate fn b/c can take much longer time to process data
msc = preproc.label_montage(msc, labels) #requires 1 label/montage session file

#Plot contact pair combinations and conditions at downsampled rate
sr_ds = 500
[msc_ds, md_ds] = proc.downsample_data(msc, md, sr_ds)

df_linked_ds = preproc.link_data(msc_ds, md_ds, gp)
df_linked_ds = df_linked_ds[df_linked_ds['sr'] >= sr_ds]

psds_ds = proc.convert_psd_montage(df_linked_ds, sr_ds)

plot.plot_PSD_montage_channels(psds_ds, msc_ds, sr_ds, gp) 
plot.plot_PSD_montage_conditions(psds_ds, msc_ds, sr_ds, gp)

#Plot HFOs for high sampling rates
df_linked = preproc.link_data(msc, md, gp)

sr_hfo = 1000
df_linked = df_linked[df_linked['sr'] == sr_hfo]

psds_hfo = proc.convert_psd_montage(df_linked, sr_hfo)

plot.plot_PSD_montage_channels(psds_hfo, msc, sr_hfo, gp) 
plot.plot_PSD_montage_conditions(psds_hfo, msc, sr_hfo, gp)

