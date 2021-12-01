#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 21 13:12:25 2021

@author: mariaolaru
"""
import math
from preproc.preprocess_funcs import *
from plts.plot_funcs import *
from proc.process_funcs import *

dir_name = '/Users/mariaolaru/RCS11L_data_day2_offmed'
#modify funcs to get and set
[msc, gp] = preprocess_settings(dir_name)
md = preprocess_data(dir_name, msc, gp) #separate fn b/c can take much longer time to process data

#Step 1: manually subset data into timechunks > 45s w/ X amp X freq
psd_final = pd.DataFrame()
buffer = 5*1000 #add 15s buffer time to beginning

ts_min = 1631917056561
ts_max = 1631917273753

ts_min = ts_min + buffer
ts_max = ts_max - 1
ts_range = [ts_min, ts_max]
mscs = subset_msc(msc, ts_range)
mscs = mscs.head(1).reset_index()
fs = int(mscs['ch1_sr'].iloc[0])
[mds, tt] = subset_md(md, mscs, fs, ts_range)

mds_still = mds[0:250*15]
mds_move = mds[250*15:250*30]

f_0, Pxx_den = signal.welch(mds_still['ch4_mV'], fs, average = 'median', window = 'hann', nperseg=fs)
f_0_m, Pxx_den_m = signal.welch(mds_move['ch4_mV'], fs, average = 'median', window = 'hann', nperseg=fs)
f_0_o, Pxx_den_o = signal.welch(mds['ch4_mV'], fs, average = 'median', window = 'hann', nperseg=fs)
plt.plot(f_0_o, np.log10(Pxx_den_o), alpha = 0.8, label = "total time")
plt.plot(f_0, np.log10(Pxx_den), alpha = 0.8, label = "still")
plt.plot(f_0_m, np.log10(Pxx_den_m), alpha = 0.8, label = "move")
plt.xlabel("Frequency (Hz)")
plt.ylabel("Power spectrum (log10(V^2/Hz))")
plt.ylim([-10, -5])
plt.rcParams["figure.figsize"] = (4.5, 4.5)
plt.tight_layout()
plt.legend()
plt.savefig('/Users/mariaolaru/RCS_entrainment.svg')

b, a = signal.butter(2, 0.5)   
filtered = signal.filtfilt(b, a, mds['ch4_mV'])
f, t, Sxx = signal.spectrogram(filtered, fs)
plt.pcolormesh(t, f, np.log10(Sxx), cmap ='rainbow')
plt.colorbar()

def plot_psd(df, title, label):
    return plt






