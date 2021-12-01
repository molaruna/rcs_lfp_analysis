#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Process the outputs of Matlab's ProcessRCS() function
Outputs: power spectra plots

@author: mariaolaru
"""
from preproc.preprocess_funcs import *
from plts.plot_funcs import *
from proc.process_funcs import *

dir_name = '/Users/mariaolaru/Documents/temp/RCS14_3mo_montages/'
labels = ["medON_stimOFF", "medON_stimON"]


labels = ["medOFF_stimON", "medOFF_stimOFF", "medON_stimOFF", "medON_stimON"]

[msc, df_notes, gp] = preprocess_settings(dir_name)
md = preprocess_data(dir_name, msc, gp) #separate fn b/c can take much longer time to process data
msc = label_montage(msc, labels) #requires 1 label/montage session file
df = link_data(msc, md) #create combined dataframe

[psds_500, psds_1000] = convert_psd_montage(df)

plot_PSD_montage_channels(psds_500, msc, gp, 500) 
plot_PSD_montage_channels(psds_1000, msc, gp, 1000) 

plot_PSD_montage_conditions(psds_500, msc, gp, 500)
plot_PSD_montage_conditions(psds_1000, msc, gp, 1000)

