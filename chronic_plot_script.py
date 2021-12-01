#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul  6 10:41:05 2021

@author: mariaolaru
"""
import pandas as pd
import numpy as np
import scipy.signal as signal
from preprocess_script import * 
from matplotlib import pyplot as plt
import os
from scipy.stats import linregress

def subset_md_chronic(md, msc, i_int, sesh_id, padding = None):
    """
    Parameters
    ----------
    md : meta data output from preprocess_data()
    ts_int : timestamp of interest (must contain UNIX millisecond scale).
    front_pad : amount of seconds to front-pad ts_int.
    back_pad : amount of seconds to back-pad ts_int.
        DESCRIPTION.

    Returns
    -------
    a subset of the meta data with preferred times
        
    """

    mds = md.loc[md['session_id'] == int(sesh_id)].reset_index(drop=True) #subset equiv functions
    ts_dt = convert_unix2dt(mds['timestamp_unix']) 
    mds.insert(2, 'timestamp', ts_dt)
    tt = len(mds)/fs
  
    return [mds, tt]

def melt_mds_chronic(mds, step_size, fs):
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
    if (num_steps == 0):
        print("Warning: this file is too small, step size rounded to 0")
    steps = np.arange(0, num_steps, 1)    
    step_arr = np.repeat(steps, [step_rep] * num_steps, axis = 0)
    
    mds = mds.rename(columns={"ch1_mV": "1", "ch2_mV": "2", "ch3_mV": "3", "ch4_mV": "4"})
    mds = mds.iloc[0:len(step_arr), :]
    mds['step'] = step_arr[0:len(mds)]    
    
    dfp = pd.melt(mds, id_vars=['timestamp', 'step'], value_vars = ['1', '2', '3', '4'], var_name = 'channel', value_name = 'voltage')
    
    return dfp

def dfp_psd_chronic(dfp, fs):
    df_psd = pd.DataFrame() #initialize metadata table of settings information
    
    for i in range(len(dfp['channel'].unique())):
        dfps = dfp.loc[dfp['channel'] == dfp['channel'].unique()[i]]
        print("completing indx i: " + str(i))
        for j in range(len(dfps['step'].unique())):
            print("completing indx : " + str(j))
            dfpss = dfps.loc[dfps['step'] == dfps['step'].unique()[j]]
            f_0, Pxx_den = signal.welch(dfpss['voltage'], fs, average = 'median', window = 'hann')
            
            psd_ind = pd.DataFrame(np.array([f_0, Pxx_den]).T, columns=['f_0', 'Pxx_den'])

            psd_ind.insert(1, 'channel', dfp['channel'].unique()[i])
            psd_ind.insert(2, 'step', dfps['step'].unique()[j])
            
            df_psd = pd.concat([df_psd, psd_ind])
    
    df_psd = df_psd.reset_index(drop=True)
    return df_psd

def get_plot_title_chronic(msc, indx_int, step_size, tt, out_name):
    subj_id = msc['subj_id'][indx_int]
    stim_state = str(int(msc['stim_status'][indx_int]))
    stim_amp = str(msc['amplitude_ma'][indx_int])
    stim_contact = str(int(msc['stim_contact_cath'][indx_int]))
    step_size_str = str(step_size)
    tt = str(round(tt, 2))
    
    plot_title = subj_id + '_' + out_name +'\nstim state: ' +  stim_state + '; stim contact: ' + stim_contact + '; stim amp: ' + stim_amp + '\ntime/PSD = ' + step_size_str + 's (' + tt + 's total)'
    return plot_title

def get_name_chronic(gp, out_name_full):
    
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

def plot_PSD_long_chronic(md, msc, gp, indx_int, step_size, out_name, plot_title, df_psd):
    subj_id = msc['subj_id'].loc[indx_int]    
    
    sense_contacts = [msc['ch1_sense_contact_an'].iloc[indx_int], msc['ch1_sense_contact_cath'].iloc[indx_int], msc['ch2_sense_contact_an'].iloc[indx_int], msc['ch2_sense_contact_cath'].iloc[indx_int], msc['ch3_sense_contact_an'].iloc[indx_int], msc['ch3_sense_contact_cath'].iloc[indx_int], msc['ch4_sense_contact_an'].iloc[indx_int], msc['ch4_sense_contact_cath'].iloc[indx_int]]    
    stim_freq = msc['stimfrequency_hz'].iloc[indx_int]
    
    fig, axs = plt.subplots(len(df_psd['channel'].unique()), figsize=(15, 15))
    fig.suptitle(plot_title)
    colors = ['royalblue', 'hotpink']

    for i in range(len(df_psd['channel'].unique())):
        ax_title = 'ch' + str(i+1) + ': contacts ' + str(int(sense_contacts[i*2])) + '-' + str(int(sense_contacts[i*2+1]))
        axs[i].set_title(ax_title)
        
        for ax in fig.get_axes():
            ax.label_outer()
            
        axs[i].set(xlabel = 'frequency (Hz)', ylabel = 'mV**2/Hz')
        axs[i].axvline(13, 0, 1, c = 'indianred', alpha = 0.4)
        axs[i].axvline(30, 0, 1, c = 'indianred', alpha = 0.4)
        axs[i].axvline(60, 0, 1, c = 'olivedrab', alpha = 0.4)
        axs[i].axvline(90, 0, 1, c = 'olivedrab', alpha = 0.4)
        axs[i].axvline(stim_freq, 0, 1, c = 'sandybrown', alpha = 0.4)
        axs[i].axvline(stim_freq/2, 0, 1, c = 'sandybrown', alpha = 0.4)

        for j in range(len(df_psd['step'].unique())):
            df_psds = df_psd.loc[df_psd['channel'] == df_psd['channel'].unique()[i]]
            df_psdss = df_psds.loc[df_psds['step'] == df_psds['step'].unique()[j]]
            
            axs[i].semilogy(df_psdss['f_0'], df_psdss['Pxx_den'], c = colors[0], alpha = 0.4)
    
    out_name_full = subj_id + "_" + out_name
    out_plot_fp = get_name_chronic(gp, out_name_full)
                 
    fig.savefig(out_plot_fp + ".svg")
    print("Plotting: \n" + out_name_full + "\n")
    
def df_phs_plot(msc, indx_int, df_psd, freq_thresh, freq_comp, fs, out_name, plot_title, gp):
    subj_id = msc['subj_id'].loc[indx_int] 
    #test
    for i in range(len(df_psd['step'].unique())):
#        i=0
        df_psds = df_psd.loc[df_psd['channel'] == '4']
        df_psdss = df_psds.loc[df_psds['step'] == i]
        df_psdss = df_psdss.reset_index(drop = True)

        df_psdsss = df_psdss[(df_psdss['f_0'] >= freq_thresh[0]) & (df_psdss['f_0'] <= freq_thresh[1])]
        df_psdsss = df_psdsss.reset_index(drop=True)

        max_freq_series = df_psdsss.loc[df_psdsss['Pxx_den'] == df_psdsss['Pxx_den'].max()]
        max_freq = float(max_freq_series['f_0'])
        max_freq_i = df_psdss[df_psdss['f_0'] == max_freq].index[0]

        cutoff = 20
        df_psdss['Pxx_den_smooth'] = butter_lowpass_filtfilt(df_psdss.loc[:, 'Pxx_den'], cutoff, fs, 1)


        df_psdss_diff = diff(df_psdss['Pxx_den_smooth'], df_psdss['f_0'])  
       
        hc_f_0 = [df_psdss['f_0'][max_freq_i - 5], df_psdss['f_0'][max_freq_i + 5]]
        hc_Pxx_den = [df_psdss['Pxx_den'][max_freq_i - 5], df_psdss['Pxx_den'][max_freq_i + 5]]
            
        end_i_vec = df_psdss_diff[df_psdss_diff['Pxx_den_diff']<0]
        end_i_vec = end_i_vec.reset_index()
        end_ii = end_i_vec[end_i_vec['index'] == max_freq_i].index[0]
        end_i = end_i_vec.loc[end_ii-2, 'index'] 
        end_i_refl = max_freq_i + (max_freq_i - end_i)

        slope_f_0 = [df_psdss['f_0'][end_i], df_psdss['f_0'][end_i_refl]]
        slope_Pxx_den = [df_psdss['Pxx_den_smooth'][end_i], df_psdss['Pxx_den_smooth'][end_i_refl]]


        #min_amp_pxx_den = np.mean(slope_Pxx_den)
        min_amp_pxx_den = np.exp((np.log(slope_Pxx_den[0])+np.log(slope_Pxx_den[1]))/2)
        
#        plt.vlines(np.mean(slope_f_0), df_psdss['Pxx_den'].min(), df_psdss['Pxx_den'].max())
#        plt.hlines(np.mean(slope_Pxx_den), 0, 120)
        plt.semilogy(df_psdss['f_0'], df_psdss['Pxx_den'], label = "raw")
        plt.plot(df_psdss['f_0'], df_psdss['Pxx_den_smooth'], alpha = 0.8, linewidth = 1, label = "1st order filtfilt")

        plt.plot(hc_f_0, hc_Pxx_den, alpha = 0.5, c = 'red', marker = "o", markersize = 2)        
        plt.plot(slope_f_0, slope_Pxx_den, alpha = 0.5, c = 'purple', marker = "o", markersize = 2)

        plt.plot([max_freq, max_freq], [min_amp_pxx_den, df_psdss['Pxx_den'][max_freq_i]], alpha = 1, c = 'purple', marker = "o", markersize = 2, label = "peak height: local d(x)=0 of max freq")
        plt.plot([max_freq, max_freq], [np.mean(hc_Pxx_den), df_psdss['Pxx_den'][max_freq_i]], alpha = 0.5, c = 'red', marker = "o", markersize = 2, label = "peak height: avg +/- 5Hz max freq")

        plt.legend()

 #       plt.xlim(60, 90)
 #       plt.ylim(df_psdsss['Pxx_den'].min(), df_psdsss['Pxx_den'].max())
        #plt.show()
        out_name_full = subj_id + "_" + out_name
        out_plot_fp = get_name_chronic(gp, out_name_full)
        plt.title(plot_title)
        plt.tight_layout()
        plt.savefig(out_plot_fp + "_sensech" + str(4) + "_step" + str(i) + ".svg")
        print("Plotting: \n" + out_name_full + "\n")
        plt.close()

def butter_lowpass_filtfilt(data, cutoff, fs, order=5):
    b, a = butter_lowpass(cutoff, fs, order=order)
    y = signal.filtfilt(b, a, data)
    return y    
    
def butter_lowpass(cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = signal.butter(order, normal_cutoff, btype = 'low', analog = False)
    
    return b, a

def diff(Pxx_den, f_0):
    d = np.diff(Pxx_den)
    freq_diff = f_0[1]-f_0[0]
    d_f0=f_0[0:len(f_0)-1] + freq_diff/2
    df_smooth_diff = pd.DataFrame({'f_0_diff':d_f0, 'Pxx_den_diff':d})
 
    return df_smooth_diff


