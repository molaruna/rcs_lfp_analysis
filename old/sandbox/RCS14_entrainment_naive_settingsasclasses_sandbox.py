# RCS14_entrainment_naive.py
# Generate timeseries analysis and power estimate
# Author: maria.olaru@
"""
Created on Mon May  3 18:22:44 2021

@author: mariaolaru
"""
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
#from sklearn import svm
#from sklearn.metrics import accuracy_score, f1_score
import scipy.stats as stats
from scipy import io
import scipy.signal as signal

class Channel:
    def __init__(self, stim_label, fs, pos, neg, lpf): 
        self.stim = Stim(stim_label, fs)
        self.sense = Sense(pos, neg, lpf)

class Stim:
    def __init__(self, label, fs):
        self.label = label
        self.fs = fs
        
class Sense:
    def __init__(self, pos, neg, lpf):
        self.label = Label(pos, neg)
        self.lpf = lpf
        
class Label:
    def __init__(self, pos, neg):
        self.pos = pos
        self.neg = neg

class Channels:
    def __init__(self, ch0, ch1, ch2, ch3):
        self.ch0 = ch0 #subcortical lower contacts
        self.ch1 = ch1 #subcortical upper contacts
        self.ch2 = ch2 #cortical 8-9
        self.ch3 = ch3 #cortical 10-11
        
def preproc_signal_df(df):
    """ renames timestamp to standard convention """
    df_rename = df.rename(columns={"DerivedTime": "timestamp", "key0": "ch1", "key1": "ch2", "key2": "ch3", "key3": "ch4"})
    
    return df_rename

def preproc_settings_stim_df(df):
    """ 
    1. renames timestamp column to standard convention
    2. splits columns into single variables
    3. removes unit strings to maintain floating point numerical values
    4. renames remaining columns to include unit information
    """
    df = df.rename(columns={"HostUnixTime": "timestamp"})
     
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
    
    df_expanded_params = df_expanded_params.drop(['contacts'], axis = 1)
    df_expanded = pd.concat([df["timestamp"], df_expanded_contact, df_expanded_params], axis=1)
    
    return df_expanded

def preproc_settings_sense_df(df):
    """ 
    1. renames timestamp column to standard convention
    2. splits columns into single variables
    3. removes unit strings to maintain floating point numerical values
    4. renames remaining columns to include unit information
    """
    df = df.rename(columns={"timeStop": "timestamp"}) #time stop of settings

    df_ch1 = expand_sense_params(df["chan1"], "ch1")
    df_ch2 = expand_sense_params(df["chan2"], "ch2")
    df_ch3 = expand_sense_params(df["chan3"], "ch3")
    df_ch4 = expand_sense_params(df["chan4"], "ch4")
     
    df_expanded = pd.concat([df["timestamp"], df_ch1, df_ch2, df_ch3, df_ch4], axis = 1) 

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
    1. Merges the three dataframes based on key value 
    2. Sorts the three dataframes based on key value
    3. Backfills missing values
    4. Removes indices with empty values (end of file)
    """
    settings_df_join = pd.merge(df2, df3, how = "outer", on = key)
    settings_df_sort = settings_df_join.sort_values(key)
    settings_df_fill = settings_df_sort.fillna(method = 'bfill')
    settings_df = settings_df_fill.dropna()
    
    comb_df_join = pd.merge(df1, settings_df, how = "outer", on = key)
    comb_df_sort = comb_df_join.sort_values(key)
    comb_df_fill = comb_df_sort.fillna(method = 'bfill')
    comb_df = comb_df_fill.dropna()
    
    return comb_df

def plot_timeseries(df, group_settings):
    """ Plots raw timeseries. """
    f, Pxx_den = signal.periodogram(df.loc[:, 'ch0'].values, group_settings.ch0.stim.fs)
    plt.figure(figsize = (10, 8))
    plt.plot(f, Pxx_den, color = 'r')
    plt.ylabel('PSD (uV**2/Hz)')
    plt.xlabel('frequency (Hz)')
    plt.title('Periodogram -- testing')
    plt.draw()
    print("INP")

def plot_power():
    """ Plots raw timeseries. """
    print("INP")

def main():
    #Load data
    signal_df_orig = pd.read_csv("/Users/mariaolaru/Box/RC-S_Studies_Regulatory_and_Data/Patient In-Clinic Data/RCS14/InClinicVisits/v03_4wk_preprogramming/Data/aDBS/RCS14L/Session1618936204126/DeviceNPC700481H/timeDomainData.csv")
    signal_df = preproc_signal_df(signal_df_orig)
    #timedomainsettings.channel has SENSE INFORMATION
    
    settings_stim_df_orig = pd.read_csv("/Users/mariaolaru/Box/RC-S_Studies_Regulatory_and_Data/Patient In-Clinic Data/RCS14/InClinicVisits/v03_4wk_preprogramming/Data/aDBS/RCS14L/Session1618936204126/DeviceNPC700481H/stimLogSettings.csv")
    settings_stim_df = preproc_settings_stim_df(settings_stim_df_orig)

    settings_sense_df_orig = pd.read_csv("/Users/mariaolaru/Box/RC-S_Studies_Regulatory_and_Data/Patient In-Clinic Data/RCS14/InClinicVisits/v03_4wk_preprogramming/Data/aDBS/RCS14L/Session1618936204126/DeviceNPC700481H/timeDomainSettings.csv")
    settings_sense_df = preproc_settings_sense_df(settings_sense_df_orig)

    df = combine_data(signal_df, settings_stim_df, settings_sense_df, "timestamp")
    
    group_settings = get_group_settings()
    
    plot_timeseries(signal_df, group_settings.grA)

    ## SANDBOX

    
    #plot_power()



if __name__ == '__main__': 
    main()