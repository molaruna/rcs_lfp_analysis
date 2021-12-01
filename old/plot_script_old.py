# RCS14_entrainment_naive.py
# Generate timeseries analysis and power estimate
# Author: maria.olaru@
"""
Created on Mon May  3 18:22:44 2021

@author: mariaolaru
"""

import numpy as np
from matplotlib import pyplot as plt
import matplotlib.dates as md
import scipy.signal as signal
import pandas as pd
import math
import os
from preprocess_script import * 

def subset_md(md, ts_range):
    """
    Parameters
    ----------
    md : meta data output from preprocess_data()
    msc : meta settings data
    i_int: row index of interest in msc
    padding: 2 integer vector +/- amount of seconds to pad 
    ts_int : timestamp of interest (must contain UNIX millisecond scale).
    front_pad : amount of seconds to front-pad ts_int.
    back_pad : amount of seconds to back-pad ts_int.
        DESCRIPTION.

    Returns
    -------
    a subset of the meta data with preferred times
        
    """
    ts_int = msc['timestamp_unix'].iloc[i_int]

    ts_start = ts_int - padding[0] * 1000
    ts_stop = ts_int + padding[1] * 1000
    
    ts_starta = md['timestamp_unix'].sub(ts_start).abs().idxmin()
    ts_stopa = md['timestamp_unix'].sub(ts_stop).abs().idxmin()
    mds = md.iloc[ts_starta:ts_stopa, :]
    mds = mds.reset_index(drop=True)
    
    amp1 = msc['amplitude_ma'].iloc[i_int-1]
    amp2 = msc['amplitude_ma'].iloc[i_int]
    mds = mds.assign(amp=np.where(mds['timestamp_unix'] < ts_int, amp1, amp2))
    
    ts_dt = convert_unix2dt(mds['timestamp_unix'])
    mds.insert(1, 'timestamp', ts_dt)
   
    return mds

def melt_mds(mds, step_size, fs):
    """
    Parameters
    ----------
    df : wide-form meta-data as pandas object
    step_size : increment in seconds with which to group data
    fs : sample rate

    Returns
    -------
    long-form of meta-data
    """
    
    step_rep = step_size * fs
    num_steps = round(len(mds)/step_rep) #Note: Can be buggy if user does not enter in reasonable times and step-size combos
    
    steps = np.arange(0, num_steps, 1)    
    step_arr = np.repeat(steps, [step_rep] * num_steps, axis = 0)
    
    mds = mds.rename(columns={"ch1_mV": "1", "ch2_mV": "2", "ch3_mV": "3", "ch4_mV": "4"})
    mds = mds.iloc[0:len(step_arr), :]
    mds['step'] = step_arr[0:len(mds)]

    amp_1 = mds['amp'].unique()[0]    
    if (len(mds[mds['amp'] == amp_1]['step'].unique()) > 1):
        tail_row = mds[mds['amp'] == amp_1].tail(1)
        amp_1_tail_step = tail_row['step'].values[0]
        amp_1_matched_step = amp_1_tail_step - 1
        mds.loc[tail_row.index, 'step'] = amp_1_matched_step
        print("Changing last data point step value at " + str(amp_1) + "mA from PSD step " + str(amp_1_tail_step) + " to PSD step " + str(amp_1_matched_step))

    
    
    dfp = pd.melt(mds, id_vars=['timestamp', 'step', 'amp'], value_vars = ['1', '2', '3', '4'], var_name = 'channel', value_name = 'voltage')
    
    return dfp

def dfp_psd(dfp, fs):
    df_psd = pd.DataFrame() #initialize metadata table of settings information
    
    for i in range(len(dfp['channel'].unique())):
        dfps = dfp.loc[dfp['channel'] == dfp['channel'].unique()[i]]
        for j in range(len(dfps['step'].unique())):
            dfpss = dfps.loc[dfps['step'] == dfps['step'].unique()[j]]
            f_0, Pxx_den = signal.welch(dfpss['voltage'], fs, average = 'median', window = 'hann')
            
            psd_ind = pd.DataFrame(np.array([f_0, Pxx_den]).T, columns=['f_0', 'Pxx_den'])

            amp_idxs = dfp.loc[dfp.step == j, 'amp']
            amp_val = dfp['amp'][amp_idxs.index[0]]
            psd_ind.insert(0, 'amp', amp_val)
            psd_ind.insert(1, 'channel', dfp['channel'].unique()[i])
            psd_ind.insert(2, 'step', dfps['step'].unique()[j])
            
            df_psd = pd.concat([df_psd, psd_ind])
    
    df_psd = df_psd.reset_index(drop=True)
    return df_psd

def dfp_spect(dfp, fs, ch, padding):
    b, a = signal.butter(2, 0.5)
    
    dfps = dfp.loc[dfp['channel'] == str(ch)]
    filtered = signal.filtfilt(b, a, dfps['voltage'])
    f, t, Sxx = signal.spectrogram(filtered, fs)
    t = t - padding[0]
    
    return [f, t, Sxx]

def get_name(gp, out_name_full):
    
    out_plot_dir = gp + '/' + 'plots/' 
    
    if not os.path.isdir(out_plot_dir):
        os.mkdir(out_plot_dir)
       
    out_plot_fp = out_plot_dir + out_name_full
    if os.path.isfile(out_plot_fp + '.svg'):
        count = 2
        out_plot_fp_count = out_plot_fp + '_v' + str(count)
        while os.path.isfile(out_plot_fp_count + '.svg'):
            count = count + 1
            out_plot_fp_count = out_plot_fp + '_v' + str(count)        
        out_plot_fp = out_plot_fp_count  

    return out_plot_fp

def get_plot_title(out_name, ss, padding, step_size):
    subj_id = ss['subj_id'][0]
    amp_start = str(ss['amplitude_ma'][0])
    amp_stop = str(ss['amplitude_ma'][1])
    stim_contact = str(int(ss['stim_contact_cath'][0]))
    step_size = str(step_size)
    pad_start = str(padding[0])
    pad_stop = str(padding[1])
    tt = str(padding[0] + padding[1])
    
    plot_title = subj_id + '_' + out_name +'\n amps: ' + amp_start + '->' + amp_stop + '; stim contact: ' + stim_contact + '\n -' + pad_start + 's to +' + pad_stop + 's; time/PSD = ' + step_size + 's (' + tt + 's total)'
    return plot_title

def plot_PSD_long(md, msc, gp, i_int, padding, step_size, out_name):
    subj_id = msc['subj_id'].loc[i_int]    

    fs = msc['ch1_sr'].iloc[i_int]
    
    sense_contacts = [msc['ch1_sense_contact_an'].iloc[i_int], msc['ch1_sense_contact_cath'].iloc[i_int], msc['ch2_sense_contact_an'].iloc[i_int], msc['ch2_sense_contact_cath'].iloc[i_int], msc['ch3_sense_contact_an'].iloc[i_int], msc['ch3_sense_contact_cath'].iloc[i_int], msc['ch4_sense_contact_an'].iloc[i_int], msc['ch4_sense_contact_cath'].iloc[i_int]]    
    stim_freq = msc['stimfrequency_hz'].iloc[i_int]
    
    ss = msc.iloc[[i_int-1,i_int], :].reset_index()
    plot_title = get_plot_title(out_name, ss, padding, step_size)

    mds = subset_md(md, msc, i_int, padding)
    dfp = melt_mds(mds, step_size, fs)
    df_psd = dfp_psd(dfp, fs)   
    
    fig, axs = plt.subplots(len(df_psd['channel'].unique()), figsize=(15, 15))
    fig.suptitle(plot_title)
    amps = [df_psd['amp'].unique()[0], df_psd['amp'].unique()[1]]
    colors = ['royalblue', 'hotpink']

    for i in range(len(df_psd['channel'].unique())):
        ax_title = 'ch' + str(i+1) + ': contacts ' + str(int(sense_contacts[i*2])) + '-' + str(int(sense_contacts[i*2+1]))
        axs[i].set_title(ax_title)
        
        for ax in fig.get_axes():
            ax.label_outer()
            
        axs[i].set(xlabel = 'frequency (Hz)', ylabel = 'mV**2/Hz')
        #axs[i].axvline(13, 0, 1, c = 'indianred', alpha = 0.4)
        #axs[i].axvline(30, 0, 1, c = 'indianred', alpha = 0.4)
        #axs[i].axvline(stim_freq, 0, 1, c = 'sandybrown', alpha = 0.4)
        axs[i].axvline(stim_freq/2, 0, 1, c = 'sandybrown', alpha = 0.4, label = '1/2 stim freq')
        axs[i].axvline(stim_freq/2 - 5, 0, 1, c = 'olivedrab', alpha = 0.4, label = '1/2 stim freq +/- 5Hz')
        axs[i].axvline(stim_freq/2 + 5, 0, 1, c = 'olivedrab', alpha = 0.4)
        axs[i].set_xlim([0, 100])
        
        for j in range(len(df_psd['step'].unique())):
            df_psds = df_psd.loc[df_psd['channel'] == df_psd['channel'].unique()[i]]
            df_psdss = df_psds.loc[df_psds['step'] == df_psds['step'].unique()[j]]
            
            if (df_psdss['amp'].unique()[0] == amps[0]):
                cl = amps[0]
                cc = colors[0]
                lbl = 'pre-transition'
            elif(df_psdss['amp'].unique()[0] == amps[1]):
                cl = amps[1]
                cc = colors[1]
                lbl = 'post-transition'
            
            if (j == 0 or j==len(df_psd['step'].unique())-1):
                cla = str(cl) + "mA " + lbl
                axs[i].semilogy(df_psdss['f_0'], df_psdss['Pxx_den'], label = cla, c = cc, alpha = 0.4)
                axs[i].legend()
            else:
                axs[i].semilogy(df_psdss['f_0'], df_psdss['Pxx_den'], c = cc, alpha = 0.4)
    
    out_name_full = subj_id + "_" + out_name
    out_plot_fp = get_name(gp, out_name_full)
                 
    fig.savefig(out_plot_fp + ".svg")
    print("Plotting: \n" + out_name_full + "\n")

def plot_spectrogram(md, msc, gp, i_int, padding, step_size, out_name):
    
    subj_id = msc['subj_id'].loc[i_int]    

    fs = msc['ch1_sr'].iloc[i_int]
    
    sense_contacts = [msc['ch1_sense_contact_an'].iloc[i_int], msc['ch1_sense_contact_cath'].iloc[i_int], msc['ch2_sense_contact_an'].iloc[i_int], msc['ch2_sense_contact_cath'].iloc[i_int], msc['ch3_sense_contact_an'].iloc[i_int], msc['ch3_sense_contact_cath'].iloc[i_int], msc['ch4_sense_contact_an'].iloc[i_int], msc['ch4_sense_contact_cath'].iloc[i_int]]    
    stim_freq = msc['stimfrequency_hz'].iloc[i_int]
    
    ss = msc.iloc[[i_int-1,i_int], :].reset_index()
    plot_title = get_plot_title(out_name, ss, padding, step_size)

    mds = subset_md(md, msc, i_int, padding)
    dfp = melt_mds(mds, step_size, fs)
    ch = 4
    [f, t, Sxx] = dfp_spect(dfp, fs, ch, padding)


    fig, axs = plt.subplots(2, 1)
    fig.suptitle(plot_title + "testing")
    
    i = ch-1
    ax_title = 'ch' + str(i+1) + ': contacts ' + str(int(sense_contacts[i*2])) + '-' + str(int(sense_contacts[i*2+1]))

    axs[1].set_title(ax_title)      
    axs[1].set(xlabel = 'Time (seconds)', ylabel = 'Frequency (Hz)')

    axs[1].axhline(stim_freq/2, 0, 1, c = 'indianred', alpha = 0.4, label = '1/2 stim freq')
    axs[1].set_ylim([stim_freq/2 - 10, stim_freq/2 + 10])
    
    im = axs[1].pcolormesh(t, f, np.log10(Sxx)) #frequencies are off b/c scaling
    fig.colorbar(im, ax=axs[1])
    im

    axs[1].legend()

    out_name_full = subj_id + "_" + out_name
    out_plot_fp = get_name(gp, out_name_full)
                 
    fig.savefig(out_plot_fp + ".svg")
    print("Plotting: \n" + out_name_full + "\n")

