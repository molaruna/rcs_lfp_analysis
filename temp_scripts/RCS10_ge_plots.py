#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul  5 11:35:21 2021

@author: mariaolaru
"""
from preprocess_script import *
from plot_script import *
import numpy as np


def find_transitions(msc, trial_len):
    a = np.array([])
    for i in range(len(msc)):
        if (i == 0 or i == len(msc)-1):
            continue
        elif (msc['amplitude_ma'][i] != msc['amplitude_ma'][i-1]):
            indx_pass = pass_params(msc, i, trial_len)
        else:
            continue
        
        if indx_pass == True:

            a = np.append(a, i)
            a = a.astype(int)
                    
    a = a.astype(int)
    return a

def pass_params(msc, indx, trial_len):
    vars_pass = np.array([0, 0])
    cs = np.array([-1, 1])
    freq_c = msc['stimfrequency_hz'][indx]
    amp_c = msc['amplitude_ma'][indx]
    
    for i in range(len(cs)):
        val_ts = msc['timestamp_unix'][indx]
        val_i_ts = msc['timestamp_unix'][indx+cs[i]]    
        trial_i_len = abs(val_ts - val_i_ts)
        
        if (trial_i_len >= trial_len * 1000):
            vars_pass[i] = 1
        else:
            count = indx + cs[i]
            while (trial_i_len < trial_len * 1000):
            
                amp_i = msc['amplitude_ma'][count]
                freq_i = msc['stimfrequency_hz'][count]
                if (freq_i != freq_c or (i == 1 and amp_i != amp_c)):
                    vars_pass[i] = 0
                    break
                else:
                    count = count + cs[i]
                    if (count == len(msc) or count < 0):
                        vars_pass[i] = 0
                        break
                    val_ii_ts = msc['timestamp_unix'][count]
                    trial_i_len = abs(val_ts - val_ii_ts)
                    amp_ii = msc['amplitude_ma'][count]
                    if (amp_ii != amp_i and trial_i_len < trial_len * 1000):
                        vars_pass[i] = 0
                        break
                    elif (amp_ii == amp_i and trial_i_len < trial_len * 1000):
                        vars_pass[i] = 0
                    elif (amp_ii == amp_i and trial_i_len > trial_len * 1000):
                        vars_pass[i] = 1

    if vars_pass.sum() == len(vars_pass):
        indx_pass = True
    else: 
        indx_pass = False
    return indx_pass

path_list = "/Users/mariaolaru/google_drive/UCSF_Neuroscience/starr_lab_MO/studies/gamma_entrainment/meta_data/RCS10_ge_session_list.txt"

#modify funcs to get and set
[msc, gp] = preprocess_settings(path_list)
md = preprocess_data(path_list, msc) #separate fn b/c can take much longer time to process data

#get all amp transition timepoints where trial length is >= 60s
trial_len = 60 #seconds before and after transition at constant stim/amp
i_ints = find_transitions(msc, trial_len)

#first timepoint of transition, multiple PSD step sizes
ss = np.array([1, 5, 10, 20, 30, 60])
for i in range(len(ss)):
    step_size = ss[i]
    padding = [60, 60] #seconds     
    step_size = ss[i]
    out_name = 'ge_transitions' + '_' + 'step_size' + str(step_size)
    plot_PSD_long(md, msc, gp, i_ints[0], padding, step_size, out_name)

# first timepoint of transition, experiment with padding lengths
tpls = [np.arange(10, 70, 10)]
tpls = np.append(tpls, [90, 120])
for i in range(len(tpls)):
    tpl = tpls[i]
    padding = [tpl, tpl] #seconds     
    step_size = 10
    out_name = 'ge_transitions' + '_' + 'pad' + str(tpl)
    plot_PSD_long(md, msc, gp, i_ints[0], padding, step_size, out_name)

#all timepoint transitions
tps = [2, 3]
for i in range(len(i_ints[tps])):
    i = 1
    i_int = i_ints[tps][i]
    
    padding = [30, 30] #start and stop padding in seconds      
    step_size = 5
    out_name = 'ge_transitions' + '_' + 'quals_tp' + str(i)
    plot_PSD_long(md, msc, gp, i_int, padding, step_size, out_name)

    i = 0
    i_int = i_ints[tps][i]
    padding = [30, 30] #start and stop padding in seconds      
    step_size = padding[0] + padding[1]
    out_name = 'ge_transitions' + '_' + 'spect_quals_tp' + str(i)
    plot_spectrogram(md, msc, gp, i_int, padding, step_size, out_name)
    
#get all amp transition timepoints where trial length is >= 45s
trial_len = 45 #seconds before and after transition at constant stim/amp
i_ints = find_transitions(msc, trial_len)
#all timepoint transitions
for i in range(len(i_ints)):
    i_int = i_ints[i]
    padding = [30, 30] #start and stop padding in seconds      
    step_size = 5
    out_name = 'ge_transitions' + '_' + 'tl' + str(trial_len) + '_' + 'tp' + str(i)
    plot_PSD_long(md, msc, gp, i_int, padding, step_size, out_name)
    
    