# RCS14_entrainment_naive.py
# Generate timeseries analysis and power estimate
# Author: maria.olaru@
"""
Created on Mon May  3 18:22:44 2021

@author: mariaolaru
"""
import pathlib
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import matplotlib.colors as mcolors
import matplotlib
import matplotlib.dates as md
import matplotlib.ticker as mticker
import scipy.stats as stats
from scipy import io
import scipy.signal as signal
from datetime import datetime
     
def preproc_signal_df(df):
    """ renames timestamp to standard convention """
    df_rename = df.rename(columns={"DerivedTime": "timestamp_UNIX", "key0": "ch1_mV", "key1": "ch2_mV", "key2": "ch3_mV", "key3": "ch4_mV"})
    df_preproc = df_rename.drop(["localTime"], axis = 1)
    return df_preproc

def preproc_settings_stim_df(df):
    """ 
    1. renames timestamp column to standard convention
    2. splits columns into single variables
    3. removes unit strings to maintain floating point numerical values
    4. renames remaining columns to include unit information
    """
    df = df.rename(columns={"HostUnixTime": "timestamp_UNIX"})
     
    df_expanded_params = df["stimParams_prog1"].str.split(pat = ",", expand = True)
    df_expanded_params = df_expanded_params.rename(columns={0: "stim_contacts", 1: "amplitude_ma", 2: "pulsewidth_us", 3: "stimfrequency_hz"})

    df_expanded_params["amplitude_ma"] = df_expanded_params["amplitude_ma"].str.replace('mA', '')
    df_expanded_params["amplitude_ma"] = df_expanded_params["amplitude_ma"].astype(float)

    df_expanded_params["pulsewidth_us"] = df_expanded_params["pulsewidth_us"].str.replace('us', '')
    df_expanded_params["pulsewidth_us"] = df_expanded_params["pulsewidth_us"].astype(int)
    
    df_expanded_params["stimfrequency_hz"] = df_expanded_params["stimfrequency_hz"].str.replace('Hz', '')
    df_expanded_params["stimfrequency_hz"] = df_expanded_params["stimfrequency_hz"].astype(float)
     
    df_expanded_contact = df_expanded_params["stim_contacts"].str.split(pat = "+", expand = True)
    df_expanded_contact = df_expanded_contact.rename(columns={0: "stim_contact_an", 1: "stim_contact_cath"})

    df_expanded_contact["stim_contact_cath"] = df_expanded_contact["stim_contact_cath"].str.replace('-', '')
    df_expanded_contact["stim_contact_cath"] = df_expanded_contact["stim_contact_cath"].astype(int) 
    
    df_expanded_params = df_expanded_params.drop(["stim_contacts"], axis = 1)
    df_expanded = pd.concat([df["timestamp_UNIX"], df_expanded_contact, df_expanded_params], axis=1)
    
    return df_expanded

def preproc_settings_sense_df(df):
    """ 
    1. renames timestamp column to standard convention
    2. splits columns into single variables
    3. removes unit strings to maintain floating point numerical values
    4. renames remaining columns to include unit information
    """
    df = df.rename(columns={"timeStop": "timestamp_UNIX"}) #time stop of settings

    df_ch1 = expand_sense_params(df["chan1"], "ch1")
    df_ch2 = expand_sense_params(df["chan2"], "ch2")
    df_ch3 = expand_sense_params(df["chan3"], "ch3")
    df_ch4 = expand_sense_params(df["chan4"], "ch4")
     
    df_expanded = pd.concat([df["timestamp_UNIX"], df_ch1, df_ch2, df_ch3, df_ch4], axis = 1) 

    return df_expanded

def expand_sense_params(df, label):
    """ 
    Expand sense channel column so each column includes only:
        1. a single variable
        2. a single object type
    """
    df_expanded_params = df.str.split(pat = " ", expand = True)     
    df_expanded_params = df_expanded_params.rename(columns={0: (label+"_sense_contacts"), 1: (label+"_lfp1"), 2: (label+"_lfp2"), 3: (label+"_sr")})    
    
    df_expanded_params[(label+"_lfp1")] = df_expanded_params[(label+"_lfp1")].str.replace('LFP1-', '')
    df_expanded_params[(label+"_lfp1")] = df_expanded_params[(label+"_lfp1")].astype(int)
     
    df_expanded_params[(label+"_lfp2")] = df_expanded_params[(label+"_lfp2")].str.replace('LFP2-', '')
    df_expanded_params[(label+"_lfp2")] = df_expanded_params[(label+"_lfp2")].astype(int)     

    df_expanded_params[(label+"_sr")] = df_expanded_params[(label+"_sr")].str.replace('SR-', '')
    df_expanded_params[(label+"_sr")] = df_expanded_params[(label+"_sr")].astype(int)

    df_expanded_contact = df_expanded_params[(label+"_sense_contacts")].str.split(pat = "-", expand = True)
    df_expanded_contact = df_expanded_contact.rename(columns={0: (label+"_sense_contact_an"), 1: (label+"_sense_contact_cath")})
    df_expanded_contact[(label+"_sense_contact_an")] = df_expanded_contact[(label+"_sense_contact_an")].str.replace('+', '')
    df_expanded_contact[(label+"_sense_contact_an")] = df_expanded_contact[(label+"_sense_contact_an")].astype(int)
    df_expanded_contact[(label+"_sense_contact_cath")] = df_expanded_contact[(label+"_sense_contact_cath")].astype(int)     

    df_expanded_params = df_expanded_params.drop([(label+"_sense_contacts")], axis = 1)
    df_expanded = pd.concat([df_expanded_contact, df_expanded_params], axis=1)

    return df_expanded

def combine_data(df1, df2, df3, key):
    """ 
    1. Merges the 3 dataframes based on key value
    2. Sorts the 3 dataframes based on key value
    3. Backfills missing values
    4. Removes indices with empty values (end of file)
    5. Creates dateTime column with remaining timestamps
    """
    settings_df_join = pd.merge(df2, df3, how = "outer", on = key)
    settings_df_sort = settings_df_join.sort_values(key)
    settings_df_fill = settings_df_sort.fillna(method = 'bfill')
    settings_df = settings_df_fill.dropna()
    
    comb_df_join = pd.merge(df1, settings_df, how = "outer", on = key)
    comb_df_sort = comb_df_join.sort_values(key)
    comb_df_fill = comb_df_sort.fillna(method = 'ffill') #was bfill
    comb_df = comb_df_fill
#    comb_df = comb_df_fill.dropna()
    
    timestamp_UNIX_arr = comb_df[["timestamp_UNIX"]].squeeze()/1000
    timestamp_arr = np.zeros(len(timestamp_UNIX_arr), dtype='datetime64[ms]')
    
    for i in range(len(timestamp_arr)):
        timestamp_arr[i] = datetime.fromtimestamp(timestamp_UNIX_arr.iloc[i])
       
    comb_df.insert(1, 'timestamp', timestamp_arr)
    
    return comb_df

def plot_timeseries(df, ch_titles, out_dir, channel):
    """ Plots raw voltage over time. """
    %matplotlib qt
    
    data = df["ch" + str(channel) + "_mV"]

    b, a = signal.butter(2, 0.5)
    filtered = signal.filtfilt(b, a, data)
    
    fig = plt.figure()
    plt.plot(df["timestamp"], filtered)
    
    plt.xlabel('time (HH:MM:SS)')
    plt.ylabel('voltage (mV)')
    plt.title(ch_titles[0] + "\n" + ch_titles[channel])
    
    plt.setp(plt.gca().xaxis.get_majorticklabels(), 'rotation', 30)
    plt.gca().xaxis.set_major_formatter(md.DateFormatter('%H:%M:%S'))
    
    plt.show()
    fig.tight_layout()
    
    save_label = concat_label(ch_titles[0])
    plt.show()
    fig.savefig(out_dir + "/" + "timeseries_" + save_label + "_ch" + str(channel) + ".svg")
    
#    fig, ax = plt.subplots(nrows=2, ncols=2)

#    ax[0,0].set(xlabel = 'time (HH:MM)', ylabel = 'voltage (mV)', title = ch_titles[1])
#    ax[0,0].plot(df["timestamp"], df["ch1_mV"])

#   THERE IS CURRENTLY A BUG IN NEWEST MATPLOTLIB LIBRARY    
#    ax[0,0].set_xticks(ax[0,0].get_xticks())
#    ax[0,0].set_xticklabels(ax[0,0].get_xticklabels(), rotation=30)
    
#    ax[0,1].set(xlabel = 'time (HH:MM)', ylabel = 'voltage (mV)', title = ch_titles[2])
#    ax[0,1].plot(df["timestamp"], df["ch2_mV"])
    
#    ax[1,0].set(xlabel = 'time (HH:MM)', ylabel = 'voltage (mV)', title = ch_titles[3])
#    ax[1,0].plot(df["timestamp"], df["ch3_mV"])  
    
#    ax[1,1].set(xlabel = 'time (HH:MM)', ylabel = 'voltage (mV)', title = ch_titles[4])
#    ax[1,1].plot(df["timestamp"], df["ch4_mV"])
    
        
    
#    fig.suptitle(ch_titles[0])
#    fig.tight_layout()
    
#    save_label = concat_label(ch_titles[0])

#    fig.savefig("timeseries_" + save_label + ".jpg")
    #plt.plot()
    
def concat_label(curr_label):
    label = curr_label.replace('\n', '_')
    label = label.replace(' ', '_')
    label = label.replace(';', '')
    label = label.replace(':', '')
    
    return label

def plot_spectrogram(df, ch_titles, out_dir, channel):
    """ Plots spectrogram. """
    %matplotlib inline
    sr = df["samplerate"].values[0]

    data = df.loc[:, "ch" + str(channel) + "_mV"].values
    
    b, a = signal.butter(2, 0.5)
    filtered = signal.filtfilt(b, a, data)

    #plt.specgram(filtered, Fs=sr)
    f, t, Sxx = signal.spectrogram(filtered, sr)

    fig = plt.figure()
    plt.pcolormesh(t, f, np.log10(Sxx)) #frequencies are off b/c scaling
    plt.colorbar()

    plt.ylabel('Frequency (Hz)')
    plt.xlabel('Time (sec)')   
    plt.title(ch_titles[0] + "\n" + ch_titles[channel])

    plt.show()
    fig.tight_layout()
    
    save_label = concat_label(ch_titles[0])
    
    fig.savefig(out_dir + "/" + "spectrogram_" + save_label + "_ch" + str(channel) + ".jpg")
 
#    fig, ax = plt.subplots(nrows=1, ncols=2)

#    ax[0,0].set(xlabel = 'frequency (Hz)', ylabel = 'PSD (uV**2/Hz)', title = ch_titles[1])
#    f_1, Pxx_den_1 = signal.periodogram(df.loc[:, 'ch1_mV'].values, sr)
#    df_1 = pd.DataFrame({'f': np.array(f_1), 'Pxx_den': np.array(Pxx_den_1)})
#    ax[0,0].plot(df_1['f'], df_1['Pxx_den'])

    
    #fig.suptitle(ch_titles[0])
    #fig.tight_layout()
    #save_label = concat_label(ch_titles[0])

    #fig.savefig("periodogram_" + save_label + ".jpg")
    
#    ax[0,0].plot(df["timestamp"], df["ch1_mV"])
    
#    sr = df[["samplerate"]].values[0][0]
    #f, Pxx_den = signal.periodogram(df.loc[:, 'ch1_mV'].values, sr)
    #plt.figure(figsize = (10, 8))
    #plt.plot(f, Pxx_den, color = 'r')
    #plt.ylabel('PSD (uV**2/Hz)')
    #plt.xlabel('frequency (Hz)')
    #plt.title('Periodogram -- testing')
    #plt.draw()
    #print("INP")

def plot_PSD(df, ch_titles, out_dir, channel):
    """ Plots PSD using pwelch method. """
    %matplotlib inline
    sr = df["samplerate"].values[0]

    f, Pxx_den = signal.welch(df.loc[:, "ch" + str(channel) + "_mV"].values, sr, average = 'median')

    fig = plt.figure()
    plt.semilogy(f, Pxx_den)
    #plt.plot(f, Pxx_den)

    plt.xlabel('frequency (Hz)')
    plt.ylabel('PSD (mV**2/Hz)')   
    plt.title(ch_titles[0] + "\n" + ch_titles[channel])
      
    plt.show()
    fig.tight_layout()
    
    save_label = concat_label(ch_titles[0])
    fig.savefig(out_dir + "/" + "PSD_" + save_label + "_ch" + str(channel) + ".svg")

def plot_PSD_amps(df, ch_titles, out_dir, channel):
    """ Plots PSD using pwelch method. """
    %matplotlib qt
    sr = df["samplerate"].values[0]
    
    df_0 = df.loc[df['amplitude_ma'] == df['amplitude_ma'].unique()[0]]
    df_1 = df.loc[df['amplitude_ma'] == df['amplitude_ma'].unique()[1]]
    df_2 = df.loc[df['amplitude_ma'] == df['amplitude_ma'].unique()[2]]
    df_3 = df.loc[df['amplitude_ma'] == df['amplitude_ma'].unique()[3]]
    df_4 = df.loc[df['amplitude_ma'] == df['amplitude_ma'].unique()[4]]
    df_5 = df.loc[df['amplitude_ma'] == df['amplitude_ma'].unique()[5]]
    
    f_0, Pxx_den_0 = signal.welch(df_0.loc[:, "ch" + str(channel) + "_mV"].values, sr, average = 'median')
    f_1, Pxx_den_1 = signal.welch(df_1.loc[:, "ch" + str(channel) + "_mV"].values, sr, average = 'median')
    f_2, Pxx_den_2 = signal.welch(df_2.loc[:, "ch" + str(channel) + "_mV"].values, sr, average = 'median')
    f_3, Pxx_den_3 = signal.welch(df_3.loc[:, "ch" + str(channel) + "_mV"].values, sr, average = 'median')
    f_4, Pxx_den_4 = signal.welch(df_4.loc[:, "ch" + str(channel) + "_mV"].values, sr, average = 'median')
    f_5, Pxx_den_5 = signal.welch(df_5.loc[:, "ch" + str(channel) + "_mV"].values, sr, average = 'median')

    fig = plt.figure()
    
    plt.semilogy(f_0, Pxx_den_0, label = '0mA', alpha = 0.6)
    plt.semilogy(f_1, Pxx_den_1, label = '1mA', alpha = 0.6)
    plt.semilogy(f_2, Pxx_den_2, label = '2mA', alpha = 0.6)
    plt.semilogy(f_3, Pxx_den_3, label = '3mA', alpha = 0.6)
    plt.semilogy(f_4, Pxx_den_4, label = '4mA', alpha = 0.6)
    plt.semilogy(f_5, Pxx_den_5, label = '5mA', alpha = 0.6)
    
    plt.axvline(13, 0, 1, c = 'indianred')
    plt.axvline(30, 0, 1, c = 'indianred')
    plt.axvline(60, 0, 1, c = 'seagreen')
    plt.axvline(90, 0, 1, c = 'seagreen')
    
    #plt.plot(f, Pxx_den)

    plt.xlabel('frequency (Hz)')
    plt.ylabel('PSD (mV**2/Hz)')   
    plt.title(ch_titles[0] + "\n" + ch_titles[channel])
    
    plt.legend()
    plt.show()
    
    fig.tight_layout()
    
    save_label = concat_label(ch_titles[0])
    fig.savefig(out_dir + "/" + "PSDamps_" + save_label + "_ch" + str(channel) + ".svg")
 
def main():
    #Get path of current script
    curr_fp = pathlib.Path().parent.absolute()
    
    #Load data
        

    data_dir = "/Users/mariaolaru/Box/RC-S_Studies_Regulatory_and_Data/Patient In-Clinic Data/RCS10/study_visits/v07_gamma_entrainment/SCBS/RCS10L"
    sess_ID = "Session1620065586099"
    device_ID = "DeviceNPC700436H"

    med_state = "ON"
    subj_ID = "RCS10"
    sidedness = "L"
    
    channel = 1
    
#    data_dir = "~/Desktop/data" #temporary b/c don't have admin access
#    sess_ID = "Session1616621680978" #at-home streamed data, 1st session
#    med_state = "UNK"
    
    signal_df_orig = pd.read_csv(data_dir + "/" + sess_ID + "/" + device_ID + "/" + "timeDomainData.csv")
    signal_df = preproc_signal_df(signal_df_orig)
    
    #NOTE: Should add settings for stim and sense of previous session too maybe? Otherwise don't have information
    settings_stim_df_orig = pd.read_csv(data_dir + "/" + sess_ID + "/" + device_ID + "/" + "stimLogSettings.csv")
    settings_stim_df = preproc_settings_stim_df(settings_stim_df_orig)

    settings_sense_df_orig = pd.read_csv(data_dir + "/" + sess_ID + "/" + device_ID + "/" + "timeDomainSettings.csv")
    settings_sense_df = preproc_settings_sense_df(settings_sense_df_orig)

    df = combine_data(signal_df, settings_stim_df, settings_sense_df, "timestamp_UNIX")

    ch_titles = [subj_ID + "  gamma entrainment; med: " + med_state + "; side: " + sidedness + "\nstim contact " + str(df["stim_contact_cath"][0].astype(int)) + "; stim freq " + str(df["stimfrequency_hz"][0]) + "; amps " + str(df["amplitude_ma"].min()) + "-" + str(df["amplitude_ma"].max()), "Ch 1 sense contacts " + str(df["ch1_sense_contact_an"][0].astype(int)) + "+ " + str(df["ch1_sense_contact_cath"][0].astype(int)) + "-", "Ch 2 sense contacts " + str(df["ch2_sense_contact_an"][0].astype(int)) + "+ " + str(df["ch2_sense_contact_cath"][0].astype(int)) + "-", "Ch 3 sense contacts " + str(df["ch3_sense_contact_an"][0].astype(int)) + "+ " + str(df["ch3_sense_contact_cath"][0].astype(int)) + "-", "Ch 4 sense contacts " + str(df["ch4_sense_contact_an"][0].astype(int)) + "+ " + str(df["ch4_sense_contact_cath"][0].astype(int)) + "-"]
    out_dir = "plots/" + subj_ID + "/" + sess_ID
    
    plot_timeseries(df, ch_titles, out_dir, channel)
    plot_spectrogram(df, ch_titles, out_dir, channel)
    plot_PSD(df, ch_titles, out_dir, channel)
    plot_PSD_amps(df, ch_titles, out_dir, channel) 

if __name__ == '__main__': 
    main()