#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul  5 14:36:54 2021

@author: mariaolaru
"""
import scipy.signal as signal


def plot_timeseries(df, channel, label_ch, label_title, out_dir):
    #%matplotlib qt
    
    data = df["ch" + str(channel) + "_mV"]

    b, a = signal.butter(2, 0.5)
    filtered = signal.filtfilt(b, a, data)
    
    fig = plt.figure()
    plt.plot(df["timestamp"], filtered)
    
    plt.xlabel('time (HH:MM:SS)')
    plt.ylabel('voltage (mV)')
    plt.title(label_title + "\n" + label_ch)
    
    plt.setp(plt.gca().xaxis.get_majorticklabels(), 'rotation', 30)
    plt.gca().xaxis.set_major_formatter(md.DateFormatter('%H:%M:%S'))
    
    #plt.show()
    fig.tight_layout()

    fig.savefig(out_dir + "timeseries_ch" + str(channel) + ".svg")

def plot_spectrogram(df, channel, label_ch, label_title, out_dir):
    
    #%matplotlib inline
    sr = df["samplerate"].values[0]

    data = df.loc[:, "ch" + str(channel) + "_mV"].values
    
    b, a = signal.butter(2, 0.5)
    filtered = signal.filtfilt(b, a, data)

    f, t, Sxx = signal.spectrogram(filtered, sr)

    fig = plt.figure()
    plt.pcolormesh(t, f, np.log10(Sxx)) #frequencies are off b/c scaling
    plt.colorbar()

    plt.ylabel('Frequency (Hz)')
    plt.xlabel('Time (sec)')   
    plt.title(label_title + "\n" + label_ch)

    #plt.show()
    fig.tight_layout()
   
    fig.savefig(out_dir + "spectrogram_ch" + str(channel) + ".jpg")

