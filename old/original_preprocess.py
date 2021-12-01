# RCS14_entrainment_naive.py
# Generate timeseries analysis and power estimate
# Author: maria.olaru@
"""
Created on Mon May  3 18:22:44 2021

@author: mariaolaru
"""
import pathlib
import pandas as pd
from preprocess import *
from plot import *
 
def main():
    #input variables
    key = 'RCS14_L_unk_unk_unk_stimON0ma_2'

    #get paths
    curr_path = str(pathlib.Path().parent.absolute())
    user_dir = curr_path.split('/google')[0]
    study_dir = curr_path.split('/code')[0]
    
    fileinfo_all_fp = study_dir + '/meta_data/session_key.csv'

    #load data
    fileinfo_all = pd.read_csv(fileinfo_all_fp)
    
    fileinfo_all.set_index('key', inplace=True)
    fileinfo = fileinfo_all.loc[key]
    
    data_dir = user_dir + fileinfo['parent_dir']
    sess_ID = fileinfo['session_ID']
    device_ID = fileinfo['device_ID']
    med_state = fileinfo['med_state']
    subj_ID = fileinfo['subject_ID']
    sidedness = fileinfo['sidedness']
    
    channel = 1
    
    signal_df_orig = pd.read_csv(data_dir + "/" + sess_ID + "/" + device_ID + "/" + "timeDomainData.csv")    
    settings_stim_df_orig = pd.read_csv(data_dir + "/" + sess_ID + "/" + device_ID + "/" + "stimLogSettings.csv")
    settings_sense_df_orig = pd.read_csv(data_dir + "/" + sess_ID + "/" + device_ID + "/" + "timeDomainSettings.csv")

    #preprocess
    signal_df = preproc_signal_df(signal_df_orig)
    settings_stim_df = preproc_settings_stim_df(settings_stim_df_orig)
    settings_sense_df = preproc_settings_sense_df(settings_sense_df_orig)
    
    df = combine_data(signal_df, settings_stim_df, settings_sense_df, "timestamp_UNIX")

    #plot
    label_title = subj_ID + "  gamma entrainment; med: " + med_state + "; side: " + sidedness + "\nstim contact " + str(df['stim_contact_cath'].iloc[0].astype(int)) + "; stim freq " + str(df['stimfrequency_hz'].iloc[0]) + "; amps " + str(df['amplitude_ma'].min()) + "-" + str(df['amplitude_ma'].max())
    label_ch1 = "Ch 1 sense contacts " + str(df['ch1_sense_contact_an'].iloc[0].astype(int)) + "+ " + str(df['ch1_sense_contact_cath'].iloc[0].astype(int)) + "-"
    label_ch2 = "Ch 2 sense contacts " + str(df['ch2_sense_contact_an'].iloc[0].astype(int)) + "+ " + str(df['ch2_sense_contact_cath'].iloc[0].astype(int)) + "-"
    label_ch3 = "Ch 3 sense contacts " + str(df['ch3_sense_contact_an'].iloc[0].astype(int)) + "+ " + str(df['ch3_sense_contact_cath'].iloc[0].astype(int)) + "-"
    label_ch4 = "Ch 4 sense contacts " + str(df["ch4_sense_contact_an"].iloc[0].astype(int)) + "+ " + str(df["ch4_sense_contact_cath"].iloc[0].astype(int)) + "-"
    ch_titles = [label_title, label_ch1, label_ch2, label_ch3, label_ch4]
    out_dir = "plots/" + subj_ID + "/" + sess_ID
    
#    plot_timeseries(df, ch_titles, out_dir, channel)
#    plot_spectrogram(df, ch_titles, out_dir, channel)
#    plot_PSD(df, ch_titles, out_dir, channel)
#    plot_PSD_amps(df, ch_titles, out_dir, channel) 

if __name__ == '__main__': 
    main()