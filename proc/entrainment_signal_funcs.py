#!/usr/bin/python3


import scipy.signal as signal
from fooof import FOOOF
import numpy as np

def compute_entrainment(tdd, sr, stim_freq):
    """
    Parameters
    ----------
    tdd : time domain data (X channels, length: 5 seconds)
    sr : sampling rate of sense contacts
    stim_freq : frequency of stimulation

    Returns
    -------
    entrained : a boolean list of the 4 channels where True = entrainment
    """
    NUM_CHAN = tdd.shape[1]
    FREQ_ENTRAIN = stim_freq/2

    entrained = [0] * NUM_CHAN

    #Convert tdd into power spectra   
    voltage = tdd.T
    f_0, spectrum = signal.welch(voltage, sr, average = 'median', window = 'hann', nperseg=sr)

    #Compute periodic spectra    
    freq_range = [4, 100]
    f_0_flat = f_0[freq_range[0]:freq_range[1]+1]
    num_freqs = len(f_0_flat)

    #for loop b/c fooof library is not vectorized    
    spectrum_flat = np.empty((NUM_CHAN, num_freqs), dtype=float)
    for i in range(NUM_CHAN):
        fm = FOOOF(verbose=False) #initialize fooof object
        fm.fit(f_0, spectrum[i], freq_range)    
        spectrum_flat[i] = fm._spectrum_flat    

    #determine entrainment 
    freq_entrain_flat_i = int(FREQ_ENTRAIN-freq_range[0]) #note: rounds to nearest int
    entrained_power = spectrum_flat[:, freq_entrain_flat_i]
    
    spectrum_flat_mean = np.mean(spectrum_flat, axis = 1)
    spectrum_flat_std = np.std(spectrum_flat, axis = 1)
    entrain_thresh = spectrum_flat_mean + spectrum_flat_std

    entrained = entrained_power > entrain_thresh
    
    return entrained

def compute_trial_entrainment(tdd, sr, stim_freq):
    NUM_CHAN = tdd.shape[1]
    PSD_DUR = 5 #length of time-domain data (in seconds) for each power spectra
    NUM_SPECTRA = int(tdd.shape[0]/sr/PSD_DUR)
    tdd_3d = tdd.reshape(NUM_SPECTRA, int(tdd.shape[0]/NUM_SPECTRA),tdd.shape[1])
    
    entrained_arr = np.empty((NUM_SPECTRA, NUM_CHAN), dtype = bool)    
    
    for i in range(NUM_SPECTRA): #note: should vectorize
        entrained_arr[i] = compute_entrainment(tdd_3d[i], sr, stim_freq)

    return entrained_arr

def compute_entrainment_decision(entrained_arr):
    trial_entrained = False
    THRESH_NUM = 2
    sum_ch_entrain = np.sum(entrained_arr, axis = 0)
    
    if (sum_ch_entrain >= THRESH_NUM).any():
        trial_entrained = True
    
    return trial_entrained