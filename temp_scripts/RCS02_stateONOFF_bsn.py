#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Aug  8 13:43:14 2021

@author: mariaolaru
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as signal

#Import tables
study_dir = '/Users/mariaolaru/RCS02 Un-Synced Data/SummitData/SummitContinuousBilateralStreaming/RCS02L/'
fp_table = study_dir + 'tables/'
fs = 250;

mdsm_off = pd.read_csv(fp_table+'RCS02_1558206391425_OFF_mdsm.csv')
psd_off = pd.read_csv(fp_table+'RCS02_1558206391425_OFF_psd.csv')
phs_off = pd.read_csv(fp_table+'RCS02_1558206391425_OFF_phs.csv')
msc_off = pd.read_csv(fp_table+'RCS02_1558206391425_OFF_msc.csv')

mdsm_on = pd.read_csv(fp_table+'RCS02_1558130697111_ON_mdsm.csv')
psd_on = pd.read_csv(fp_table+'RCS02_1558130697111_ON_psd.csv')
phs_on = pd.read_csv(fp_table+'RCS02_1558130697111_ON_phs.csv')
msc_on = pd.read_csv(fp_table+'RCS02_1558130697111_ON_msc.csv')
msc_on = msc_on.reset_index(drop=True)

def format_avg(df_ch, value):
    df_wide = df_ch.pivot_table(index=["f_0"], columns = ['step'], values =[value])
    df_avg = df_wide.mean(axis=1).to_frame()
    df_avg['f_0'] = df_avg.index
    df_avg = df_avg.reset_index(drop=True)
    df_avg = df_avg.rename(columns={0: value})
    df_avg['std'] = df_wide.std(axis=1).array    
    return df_avg

    
#####################################################################
#Plot psd with all 12 epochs for cortical and subcortical channels
def format_psd_df(df, ch):
    df_ch = df[df['channel'] == ch]
    df_avg = format_avg(df_ch, 'Pxx_den')
    return df_avg

def plot_psd(df, title, label):
    plt.plot(df['f_0'], np.log10(df['Pxx_den']), label = label)
    plt.fill_between(df['f_0'], np.log10(df['Pxx_den'] - df['std']), np.log10(df['Pxx_den'] + df['std']), alpha = 0.3)
    plt.legend() 
    plt.title(title)
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Power spectrum (log10(V^2/Hz))")
    plt.ylim([-10, -3])
    plt.rcParams["figure.figsize"] = (4.5, 4.5)
    plt.tight_layout()
    return plt

psd_on_ch4_avg = format_psd_df(psd_on, 4)
psd_on_ch2_avg = format_psd_df(psd_on, 2)

title = "psd with dyskinesia"
plot_psd(psd_on_ch2_avg, title, "subcortical")
plot_psd(psd_on_ch4_avg, title, "cortical")

plt.savefig(study_dir + 'plots/' + 'psd_dyskinesia_on_12epochs.svg')
plt.close()

psd_off_ch4_avg = format_psd_df(psd_off, 4)
psd_off_ch2_avg = format_psd_df(psd_off, 2)

title = "psd without dyskinesia"
plot_psd(psd_off_ch2_avg, title, "subcortical")
plot_psd(psd_off_ch4_avg, title, "cortical")

plt.savefig(study_dir + 'plots/' + 'psd_dyskinesia_off_12epochs.svg')
plt.close()

#####################################################################
#Plot msc with all 12 epochs 
def format_msc_df(df, ch_subcort, ch_cort):
    df_ch = df[(df['ch_subcort'] == ch_subcort) & (df['ch_cort'] == ch_cort)]
    df_avg = format_avg(df_ch, 'Cxy')
    return df_avg

def plot_msc(df, title):
    plt.plot(df['f_0'], (df['Cxy']), color = 'green')
    plt.fill_between(df['f_0'], (df['Cxy'] - df['std']), (df['Cxy'] + df['std']), alpha = 0.3, color = 'green')
    plt.title(title)
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("MSC")
    plt.ylim([0, 0.35])
    plt.rcParams["figure.figsize"] = (4.5, 4.5)
    plt.tight_layout()
    return plt

msc_on_avg = format_msc_df(msc_on, 2, 4)
msc_off_avg = format_msc_df(msc_off, 2, 4)

plot_msc(msc_on_avg, "MSC with dyskinesia")
plt.savefig(study_dir + 'plots/' + 'msc_dyskinesia_on_12epochs.svg')
plt.close()

plot_msc(msc_off_avg, "MSC without dyskinesia")
plt.savefig(study_dir + 'plots/' + 'msc_dyskinesia_off_12epochs.svg')
plt.close()

#####################################################################
#Plot bs avg amplitude and duration
#Create meta-table with information from both conditions
def compute_bsf(mdsm, freq_thresh, burst_dur, fs):
    """
    Note: dataframes have additional column of 'condition' * will fix soon
    
    Parameters
    ----------
    mdsm : dataframe, output of melt_mds()
    freq_thresh : int array of length 2, frequencies with which to bandpass
    burst_dur : int, minimum duration necessary for a burst in ms
    fs : int, sampling rate

    Returns
    -------
    table with burst amplitude and duration

    """
    #Step 1: independently bandpass using 2nd order butterworth, then smooth
    mdsm['voltage_filtered'] = 0
    mdsm['voltage_rectified'] = 0
    mdsm['voltage_smoothed'] = 0
    mdsm['thresh_mV'] = 0
    
    for i in range(1,len(mdsm['channel'].unique())+1):       
        order = 3
        bp = freq_thresh
        b, a = signal.butter(order, bp, btype = 'bandpass', fs = fs)
        indices = mdsm[(mdsm['channel'] == i)].index
        filtered = signal.lfilter(b, a, mdsm.loc[indices, 'voltage'])           
        mdsm.loc[indices, 'voltage_filtered'] = filtered
        mdsm.loc[indices, 'voltage_rectified'] = abs(filtered)

        dur_kernel = 25 #in ms
        fs_ms = (1/fs)*1000 #ms dur of each fs
        dp_smoo = int(dur_kernel/fs_ms)

        smooth = mdsm.loc[indices, 'voltage_rectified'].rolling(window=dp_smoo, center = True).mean()
        mdsm.loc[indices, 'voltage_smoothed'] = smooth
        
        thresh = np.nanpercentile(mdsm.loc[indices, 'voltage_smoothed'], 75)
        mdsm.loc[indices, 'thresh_mV'] = thresh              
        mdsm['thresh_pass'] = mdsm['voltage_smoothed'] > mdsm['thresh_mV'] 
    
    #Step 4: find periods longer than minimum duration that pass threshold
    df_bsr = compute_burstdur(mdsm, burst_dur, fs)
          
    """
    #temp plotting for beta tests
    ss = mdsm[(mdsm['channel'] == 1) & (mdsm['condition'] == 'OFF')]
    ss = ss.reset_index(drop=True)

    out_fp = '/Users/mariaolaru/google_drive/UCSF_Neuroscience/starr_lab_MO/studies/gamma_entrainment/figures/aim1_plots'            
    from matplotlib import pyplot as plt   
    
    #plt.plot(ss.loc[1:200, 'voltage'])
    plt.plot(ss.loc[0:300, 'voltage_filtered'], label = "filtered")
    plt.plot(ss.loc[0:300, 'voltage_rectified'], label = 'rectified')
    plt.plot(ss.loc[0:300, 'voltage_smoothed'], label = 'smoothed')
    plt.axhline(mdsm['thresh_mV'].unique()[0], label = 'thresh', color = 'red')
    plt.axvline(75, label = "burst start")
    plt.legend()
    plt.savefig(out_fp + '/OFF_bsn_example.svg')

    #Step 4: Determine how many chunks pass threshold for each step using inflection pt            
    """    
    return df_bsr



mdsm_on['dyskinesia'] = '1'
mdsm_off['dyskinesia'] = '0'
mdsm = pd.concat([mdsm_off, mdsm_on])
mdsm = mdsm.reset_index(drop = True)

#Create bsn dfs
freq_thresh = np.array([60, 90])
burst_dur = 25 #in ms
df_bsr = compute_bsn(mdsm, freq_thresh, burst_dur, fs)

df_bsr[(df_bsr['condition'] == 'OFF') & (df_bsr['channel'] == 4)]['bsr'].mean()
df_bsr[(df_bsr['condition'] == 'ON') & (df_bsr['channel'] == 4)]['bsr'].mean()

##########

phs_on['dyskinesia'] = '1'
phs_off['dyskinesia'] = '0'
phs = pd.concat([phs_on, phs_off])
phs = phs.reset_index(drop = True)





