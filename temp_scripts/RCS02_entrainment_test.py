#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 21 13:12:25 2021

@author: mariaolaru
"""
import math
import scipy.signal as signal
import matplotlib as plt
import numpy as np
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

df_linked = link_data(msc, mds, gp) #create combined dataframe
df_psd = convert_psd_wide(df_linked, sr) #psds for all combinations of freq & amp
df_phs = plot_phs(df_psd)
   
def plot_psd(df_psd, stim_freq, stim_amp, gp):       
    plt.plot(df_psd['f_0'], np.log10(df_psd.iloc[:,1]), alpha = 0.8)
    plt.axvline(stim_freq/2, alpha = 0.2)
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("log10(Power)")
    plt.ylim([-10, -5])
    plt.rcParams["figure.figsize"] = (4.5, 4.5)
    plt.tight_layout()
    #plt.legend()
    plt.title('stim freq: ' + str(stim_freq) + '; stim amp: ' + str(stim_amp))
    fp = gp + '/plots/' + 'RCS02_ge_' + str(stim_freq) + str(stim_amp)
    plt.savefig(fp + '.svg')
    plt.savefig(fp + '.pdf')
    plt.close()

settings = dfc['settings'].unique()
df_phs_all = pd.DataFrame()
for i in range(len(settings)):
    curr_setting = settings[i]
    print(curr_setting)
    dfs = dfc[(dfc['settings'] == curr_setting) & (dfc['sense_contacts'] == '+9-11')]
    indx_start = dfs.head(1).index[0]
    indx_end = dfs.tail(1).index[0]
    stim_freq = dfs['stim_freq'].unique()[0]
    stim_amp = dfs['stim_amp'].unique()[0]
    buffer = 0
    if (i != 0):
        if (stim_freq != freq_prev):
            buffer = 15 * 250
    dfb = dfs.iloc[buffer:(len(dfs)-1), :]     
    if(len(dfb) > 250*10):
        df_psd = convert_psd(dfb['voltage'], 250, dfb['sense_contacts'].unique()[0])
        df_psdf = flatten_psd(df_psd['f_0'], df_psd.iloc[:,1], [4, 100])
        df_phs_ind = compute_phs(df_psdf['f_0'], df_psdf['fooof_flat'])
        df_phs_ind['stim_amp'] = stim_amp
        df_phs_ind['stim_freq'] = stim_freq
        df_phs_all = pd.concat([df_phs_all, df_phs_ind])
        #plot_psd(df_psd, stim_freq, stim_amp, gp)
    else:
        continue
    freq_prev = dfs['stim_freq'].unique()[0]
    
df_phs = df_phs_all[df_phs_all['band'] == 'gamma'].reset_index(drop=True)    

indx_noentrain = df_phs[(df_phs['freq'] < (df_phs['stim_freq']/2-2)) | (df_phs['freq'] > (df_phs['stim_freq']/2+2))].index
df_entrain = df_phs
df_entrain.loc[indx_noentrain, 'max_amp'] = 0

x_labels = np.sort(df_entrain['stim_freq'].unique())
y_labels = np.sort(df_entrain['stim_amp'].unique())

#x_label_list = np.arange(111.1, 169.6, 0.1)
x_str = ["%.1f" % x for x in x_labels]

#y_label_list = np.round(np.arange(0, 3.2, 0.1), 2)
df_mat = pd.DataFrame(data = -1, columns = x_str, index = y_labels)

for i in range(len(df_entrain)):
    stim_freq = str(df_entrain['stim_freq'][i])
    stim_amp = df_entrain['stim_amp'][i]
    df_mat.loc[stim_amp, stim_freq] = df_entrain.loc[i, 'max_amp']

plt.imshow(df_mat, cmap='hot', interpolation='nearest')
plt.show()