# RCS14_entrainment_naive.py
# Generate timeseries analysis and power estimate
# Author: maria.olaru@
"""
Created on Mon May  3 18:22:44 2021

@author: mariaolaru
"""
import pathlib
import pandas as pd
import os
from initialize import *
from preprocess import *
from plot import *  
import glob

def main():
    dir_list_fp = "/Users/mariaolaru/google_drive/UCSF_Neuroscience/starr_lab_MO/studies/gamma_entrainment/meta_data/RCS14_chronic_session_list_temp.txt"
    curr_path = str(pathlib.Path().parent.absolute()) #get curr path
    
    #read in as df
    paths = pd.read_csv(dir_list_fp, header=None)
    channels = [1, 2, 3, 4]  
    
    for i in range(0, len(paths)):
        p = paths[0][i]
        #p = '/Users/mariaolaru/RCS14 Un-Synced Data/SummitData/SummitContinuousBilateralStreaming/RCS14L/Session1616623122818/DeviceNPC700481H/'

        exists = len(find_file("timeDomainData.csv", p))
        if (exists == 0):
            print("could not process: " + p)
            continue
        else:
            print("processing: " + p)


        sls = pd.read_csv(p + "stimLogSettings.csv")
        tds = pd.read_csv(p + "timeDomainSettings.csv", index_col=False)
        elt = pd.read_csv(p + "eventLogTable.csv")
        md = pd.read_csv(p + "metaData.csv")        
        
        slsp = preprocess_sls(sls)
        tdsp = preprocess_tds(tds)
        eltp = preprocess_elt(elt)
        mdp = preprocess_md(md)
        
        sc = settings_combine(eltp, mdp, slsp, tdsp, p)
        
        med_state = "unk" #not sure whether this info is captured anywhere        
        subj_ID = mdp['subj_ID'][0]
        sidedness = mdp['implant_side'][0]
        session_ID = elt['SessionId'][0]
        
        p_out = p + "plots/"     
        if not os.path.isdir(p_out):
            os.mkdir(p_out)

#        label_title = subj_ID + str(session_ID) + " med: " + med_state + "; side: " + sidedness + "\nstim contact " + str(sc['stim_contact_cath'].iloc[0].astype(int)) + "; stim freq " + str(sc['stimfrequency_hz'].iloc[0]) + "; amps " + str(sc['amplitude_ma'].min()) + "-" + str(sc['amplitude_ma'].max())

        #tdd = pd.read_csv(p + "timeDomainData.csv")
        #tddp = preprocess_tdd(tdd)        
        
# =============================================================================
#         for j in range(0, len(channels)):
#             print("plotting channel: " + str(j))
#             
#             #NEED TO EDIT THIS TO BE AS SMALL AS POSSIBLE!!
#             df = td_combine(slsp, tdsp, tddp, channel,"timestamp_UNIX")
# 
#             label_ch = "Ch " + str(channels[j]) + " sense contacts " + str(df['ch' + str(channels[j]) + '_sense_contact_an'].iloc[0].astype(int)) + "+ " + str(df['ch' + str(channels[j]) + '_sense_contact_cath'].iloc[0].astype(int)) + "-" 
# 
#             plot_timeseries(df, channels[j], label_ch, label_title, p_out)
#             plot_spectrogram(df, channels[j], label_ch, label_title, p_out)
#             plot_PSD(df, channels[j], label_ch, label_title, p_out)
#             #plot_PSD_amps(df, channels[j], label_ch, label_title, out_dir) 
# 
# =============================================================================

    #Create master settings log table
    #This needs to be a separate script
    #Hand this script all the filepaths you'd like to use
    
    path = "/Users/mariaolaru/RCS14 Un-Synced Data/SummitData/SummitContinuousBilateralStreaming/RCS14L/"
    
    all_files = glob.glob(path + "*combined_settings.csv")
    li = []
    for filename in all_files:
        df_sc = pd.read_csv(p, index_col=None, header=0)
        li.append(df_sc)

    df_msc = pd.concat(li, axis=0, ignore_index=True)
    
if __name__ == '__main__': 
    main()
    
    
    
    
    