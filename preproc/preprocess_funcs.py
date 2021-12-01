# RCS14_entrainment_naive.py
# Generate timeseries analysis and power estimate
# Author: maria.olaru@
"""
Created on Mon May  3 18:22:44 2021

@author: mariaolaru
"""

import numpy as np
import pandas as pd
import glob
from datetime import datetime
import os
import re

def get_filepaths(dir_name):
    filepaths = glob.glob(dir_name + "Session*/" + "Device*")
    return filepaths


def find_file(file_name, parent_dir):
    
    #STEP 1: Get all files in all subdirectories of parent_dir
    array_all = np.array([])

    for root, subdirectories, files in os.walk(parent_dir):
        if file_name in files:
            file_match = os.path.join(root, file_name)
            array_all = np.append(array_all, file_match)
    
    return array_all

def preprocess_tdd(tdd, p):
    """
    Parameters
    ----------
    df : JSON to CSV converted time domain data.

    Returns
    -------
    df_preproc : Restructured and reformatted tdd.

    """
    tdd_rename = tdd.rename(columns={"DerivedTime": "timestamp", "key0": "ch0_mv", "key1": "ch1_mv", "key2": "ch2_mv", "key3": "ch3_mv"})
    tdd_preproc = tdd_rename.drop(["localTime"], axis = 1)
    tdd_preproc = tdd_preproc.drop(["samplerate"], axis = 1)
    
#    sesh_id = os.path.basename(os.path.abspath(os.path.join(p, "../")))
#    sesh_id = re.findall(r'\d+', sesh_id)[0]
    
#    tdd_preproc.insert(0, 'session_id', sesh_id)
#    tdd_preproc[['session_id']] = tdd_preproc[['session_id']].astype(int).astype(str)
    tdd_preproc[['timestamp']] = tdd_preproc[['timestamp']].astype(int)


    return tdd_preproc

def preprocess_sls(df):
    """
    Parameters
    ----------
    df : JSON to CSV converted  stim log settings data.

    Returns
    -------
    df_expanded : Restructured and reformatted sls data.

    """
    df = df.rename(columns={"HostUnixTime": "timestamp_unix", "therapyStatus": "stim_status"})
     
    df_expanded_params = df["stimParams_prog1"].str.split(pat = ",", expand = True)
    df_expanded_params = df_expanded_params.rename(columns={0: "stim_contacts", 1: "stim_amp", 2: "stim_pw", 3: "stim_freq"})

    df_expanded_params["stim_amp"] = df_expanded_params["stim_amp"].str.replace('mA', '')
    df_expanded_params["stim_amp"] = df_expanded_params["stim_amp"].astype(float)

    df_expanded_params["stim_pw"] = df_expanded_params["stim_pw"].str.replace('us', '')
    df_expanded_params["stim_pw"] = df_expanded_params["stim_pw"].astype(int)
    
    df_expanded_params["stim_freq"] = df_expanded_params["stim_freq"].str.replace('Hz', '')
    df_expanded_params["stim_freq"] = df_expanded_params["stim_freq"].astype(float)
     
    df_expanded_contact = df_expanded_params["stim_contacts"].str.split(pat = "+", expand = True)
    df_expanded_contact = df_expanded_contact.rename(columns={0: "stim_contact_an", 1: "stim_contact"})

    df_expanded_contact["stim_contact"] = df_expanded_contact["stim_contact"].str.replace('-', '')
    df_expanded_contact["stim_contact"] = df_expanded_contact["stim_contact"].astype(int) 
    
    df_expanded_params = df_expanded_params.drop(["stim_contacts"], axis = 1)
    df_expanded = pd.concat([df.loc[:, ["timestamp_unix", "stim_status"]], df_expanded_contact, df_expanded_params], axis=1)
    
    indx = np.array(df_expanded[(df_expanded['stim_status'] == 1) & (df_expanded['stim_amp'] == 0)].index)

    #include low stimulation amplitudes in field
    if indx.size != 0:
        df_expanded.loc[indx, 'stim_amp'] = 0.001

    #change amplitude to reflect stimulation status
    indx = np.array(df_expanded[(df_expanded['stim_status'] == 0) & (df_expanded['stim_amp'] != 0)].index)
    
    if indx.size != 0:
        df_expanded.loc[indx, 'stim_amp'] = 0
    
    return df_expanded

def preprocess_tds(df):
    """
    Parameters
    ----------
    df : JSON to CSV converted time domain settings data.

    Returns
    -------
    df_expanded : Restructured and reformatted tds data.
    """
    #NEED DECIDE WHICH TIMESTAMP TO KEEP, TIMESTOP, OR TIMESTART
    df = df.rename(columns={"timeStart": "timestamp_unix"}) #time start of settings
    df = df.rename(columns={"timeStop": "timestamp_unix_stop"}) #time stop of settings

    df_ch1 = expand_sense_params(df["chan1"], "ch0")
    df_ch2 = expand_sense_params(df["chan2"], "ch1")
    df_ch3 = expand_sense_params(df["chan3"], "ch2")
    df_ch4 = expand_sense_params(df["chan4"], "ch3")
     
    df_expanded = pd.concat([df["timestamp_unix"], df["timestamp_unix_stop"], df_ch1, df_ch2, df_ch3, df_ch4], axis = 1) 
    df_expanded = df_expanded.drop(['ch1_sr', 'ch2_sr', 'ch3_sr'], axis = 1)
    df_expanded = df_expanded.rename(columns={'ch0_sr': 'sr'}) 

    return df_expanded

def expand_sense_params(df, label):
    """
    Parameters
    ----------
    df : data from a single tds channel.
    label : label of tds channel from df.

    Returns
    -------
    df_expanded : expanded columns for each input datatype

    """
    df_expanded_params = df.str.split(pat = " ", expand = True)     
    df_expanded_params = df_expanded_params.rename(columns={0: (label+"_sense_contacts"), 1: (label+"_lfp1"), 2: (label+"_lfp2"), 3: (label+"_sr")})    
    
    df_expanded_params[(label+"_lpf1")] = df_expanded_params[(label+"_lfp1")].str.replace('LFP1-', '')
#    df_expanded_params[(label+"_lfp1")] = df_expanded_params[(label+"_lfp1")].astype(int)
     
    df_expanded_params[(label+"_lpf2")] = df_expanded_params[(label+"_lfp2")].str.replace('LFP2-', '')
#    df_expanded_params[(label+"_lfp2")] = df_expanded_params[(label+"_lfp2")].astype(int)     
    
    df_expanded_params[(label+"_lpfs")] = df_expanded_params[label+"_lpf1"] + '-' + df_expanded_params[label+"_lpf2"]

    df_expanded_params[(label+"_sr")] = df_expanded_params[(label+"_sr")].str.replace('SR-', '')

    df_expanded_params = df_expanded_params.drop([label + '_lfp1', label + '_lfp2', label + '_lpf1', label + '_lpf2'], axis = 1)
    
    #Need to edit this later
    if ((df_expanded_params[(label+"_sr")] == 'Disabled').any()):
        indx_vals = df_expanded_params[df_expanded_params[(label+"_sr")]=='Disabled'].index
        df_expanded_params[(label+"_sr")][indx_vals] = 0
        print("Warning: hardcoding sr of 0 for Disabled value")
    df_expanded_params[(label+"_sr")] = df_expanded_params[(label+"_sr")].astype(int)

    #df_expanded_contact = df_expanded_params[(label+"_sense_contacts")].str.split(pat = "-", expand = True)
    #df_expanded_contact = df_expanded_contact.rename(columns={0: (label+"_sense_contact_an"), 1: (label+"_sense_contact_cath")})
    #df_expanded_contact[(label+"_sense_contact_an")] = df_expanded_contact[(label+"_sense_contact_an")].str.replace('+', '', regex=True)
    #df_expanded_contact[(label+"_sense_contact_an")] = df_expanded_contact[(label+"_sense_contact_an")].astype(int)
    #df_expanded_contact[(label+"_sense_contact_cath")] = df_expanded_contact[(label+"_sense_contact_cath")].astype(int)     

    #df_expanded_params = df_expanded_params.drop([(label+"_sense_contacts")], axis = 1)
    #df_expanded = pd.concat([df_expanded_contact, df_expanded_params], axis=1)

    return df_expanded_params

def preprocess_elt(df, p):
    """
    Parameters
    ----------
    df : JSON to CSV converted event log table data

    Returns
    -------
    df_rename : Restructured and reformatted elt data.

    """

        
    if not "SessionId" in df:
        sesh_id = os.path.basename(os.path.abspath(os.path.join(p, "../")))
        sesh_id = float(re.findall(r'\d+', sesh_id)[0])
    
    if df.empty: 
        df = pd.DataFrame([sesh_id], columns = ['session_id'])
        return df
    
    df_rename = df.rename(columns={"HostUnixTime": "timestamp_unix", "SessionId": "session_id", "EventType": "event_type", "EventSubType" : "event_subtype"})
    df_subset = df_rename[["session_id", "timestamp_unix", "event_type", "event_subtype"]]

    
    #Get comments from older version of RCS implementation
    partial_match = ["Feeling", "Balance", "Slowness", "Dyskinesia", "Dystonia", "Rigidity", "Speech", "Tremor", "Mania", "Sleep"]

    import math    
    indx = np.array([])
    for i in range(len(df_subset)):
        entry = df_subset.loc[i, 'event_type']
        if type(entry) != str:
            if math.isnan(entry):
                continue
        for j in range(len(partial_match)):
            pm = partial_match[j]
            if entry.startswith(pm):
                indx = np.append(indx, i)
    
    if indx.size > 0:
        df_reformat = df_subset.iloc[indx, :]
        df_reformat = df_reformat.rename(columns = {'event_type': 'conditions', 'event_subtype': 'extra_comments'})
        df_standard = pd.melt(df_reformat, id_vars=['session_id', 'timestamp_unix'], value_vars  = ['conditions', 'extra_comments'])
        df_standard = df_standard.rename(columns = {'variable': 'event_type', 'value': 'event_subtype'})
        df_subset = pd.concat([df_subset, df_standard])

    ls_keep = ["conditions", "medication", "extra_comments"]
    df_select = df_subset.loc[df_subset['event_type'].isin(ls_keep)]
  
    if (df_select.size == 0):
        df_standard = df_subset[["session_id", "timestamp_unix"]]
        df_reformat = df_standard.iloc[0:1, :]
    else:
        dfp = df_select.pivot(columns = 'event_type')['event_subtype']
        if (not "conditions" in dfp):
            dfp = dfp.assign(conditions = np.nan)
        if (not "medication" in dfp):
            dfp = dfp.assign(medication = np.nan)
        if (not "extra_comments" in dfp):
            dfp = dfp.assign(extra_comments = np.nan)

        df_reformat = df_select[["session_id", "timestamp_unix"]]
        
        df_reformat = df_reformat.assign(medication = pd.Series(dfp['medication']))
        df_reformat = df_reformat.assign(symptoms = pd.Series(dfp['conditions']))
        df_reformat = df_reformat.assign(comments = pd.Series(dfp['extra_comments']))
        
    
    return df_reformat

def preprocess_md(df, p):
    """
    Parameters
    ----------
    df : JSON to CSV converted meta data.

    Returns
    -------
    df : Restructured and reformatted md.

    """
    if (df['implant_side'] == 'Undefined').any():
        implant_side = os.path.abspath(os.path.join(p, "../..")) 
        implant_side = implant_side[-1]
        df.implant_side[df.implant_side=="Undefined"]=implant_side
    else:
        df_implant_expanded = df[("implant_side")].str.split(pat = " ", expand = True)
        df["implant_side"] = df_implant_expanded.iloc[:,0]
    df_rename = df.rename(columns={"subj_ID": "subj_id"})
    df_rename['subj_id'] = df_rename['subj_id'][0][:-1]
    return df_rename
    
def settings_combine(eltp, mdp, slsp, tdsp, out_dir):
    """
    Parameters
    ----------
    eltp : preprocessed event log table data.
    mdp : preprocessed meta data.
    slsp : preprocessed stim log settings data.
    tdsp : preprocessed time domain settings data.
    out_dir : fullpath to parent directory of output.

    Returns
    -------
    df : a single dataframe containing all input data.

    """  
    subj_id = mdp['subj_id'].unique()
    subj_id = subj_id[~pd.isnull(subj_id)][0]

    hemi = mdp['implant_side'].unique()
    hemi = hemi[~pd.isnull(hemi)][0]
    
    sesh_id = eltp['session_id'].unique()
    sesh_id = sesh_id[~pd.isnull(sesh_id)][0]

    tdspc = tdsp.drop(['timestamp_unix_stop'], axis=1)
    df = slsp.append(tdspc)

    df.insert(0, 'subj_id', subj_id)
    df.insert(1, 'implant_side', hemi)
    df.insert(2, 'session_id', sesh_id)
    
    df = df.sort_values('timestamp_unix')
    df = df.reset_index(drop = True)

    timestamp_dt = convert_unix2dt(df["timestamp_unix"])
       
    df.insert(4, 'timestamp', timestamp_dt)
    
    df[['timestamp_unix']] = df[['timestamp_unix']].astype(int)
    df[['session_id']] = df[['session_id']].astype(int).astype(str)
    
    df.to_csv(out_dir + 'combined_settings.csv', index = False, header=True)
    eltp.to_csv(out_dir + 'session_notes.csv', index = False, header = True)
    
    return df

def convert_unix2dt(series):
    """
    Parameters
    ----------
    series : column from pandas dataframe in UNIX microsecond formatting

    Returns
    -------
    timestamp_dt : series in date-time format

    """
    if (len(series) == 1):
        unix_s = series/1000
    else:    
        unix_s = series.squeeze()/1000
        
    timestamp_dt = np.zeros(len(unix_s), dtype='datetime64[ms]')
    
    for i in range(len(timestamp_dt)):
        timestamp_dt[i] = datetime.fromtimestamp(unix_s.iloc[i])
        
    return timestamp_dt

def preprocess_settings(dir_name):
    """
    Parameters
    ----------
    path_list : full-path list of all directories to individually process, with data one level lower than head directory. 

    Returns
    -------    
    Meta-settings table of all individual session settings tables
    """
    paths = get_filepaths(dir_name)
    msc = pd.DataFrame() #initialize metadata table of settings information
    meltp = pd.DataFrame()

    p_temp = paths[0]
    gp = os.path.abspath(os.path.join(p_temp, "../.."))       
    subj_id = os.path.basename(gp)
    
    msc_fp = gp + '/' + subj_id + '_meta_combined_settings.csv'
    meltp_fp = gp + '/' + subj_id + '_meta_session_notes.csv'

    if (os.path.exists(msc_fp)):
        msc = pd.read_csv(msc_fp, header=0)
        meltp = pd.read_csv(meltp_fp, header=0)

    else:    
        for i in range(0, len(paths)):
            p = paths[i] + '/'
            # for testing purposes:
            #p = '/Users/mariaolaru/RCS02 Un-Synced Data/SummitData/SummitContinuousBilateralStreaming/RCS02L/Session1557951903217/Device/'
        
            exists = len(find_file("timeDomainData.csv", p))
            if (exists == 0):
                print("Can't process, timeDomainData does not exist: \n" + p + "\n")
                continue
            else:
                print("Processing settings: \n" + p + "\n")
              
            sls = pd.read_csv(p + "stimLogSettings.csv")
            tds = pd.read_csv(p + "timeDomainSettings.csv", index_col=False)
            
            if os.stat(p + "eventLogTable.csv").st_size > 1:
                elt = pd.read_csv(p + "eventLogTable.csv")
            else:
                elt = pd.DataFrame()
                
            md = pd.read_csv(p + "metaData.csv")                
        
            slsp = preprocess_sls(sls)
            tdsp = preprocess_tds(tds)
            eltp = preprocess_elt(elt, p)        
            mdp = preprocess_md(md, p)
            
            sc = settings_combine(eltp, mdp, slsp, tdsp, p)
            msc = pd.concat([msc, sc])
            meltp = pd.concat([meltp, eltp])
            
        
        msc['session_id'] = msc['session_id'].astype(int)

#        col_dont_fill = 0 #do not fill for med/symp/comments
#        if ('medication' in msc):
#            col_dont_fill = col_dont_fill + 1
#        if ('symptoms' in msc):
#            col_dont_fill = col_dont_fill + 1
#        if ('comments' in msc):
#            col_dont_fill = col_dont_fill + 1

#        fill_cols = msc.columns[0:len(msc.columns)-col_dont_fill] 
        msc = msc.sort_values('timestamp_unix')
        meltp = meltp.sort_values('timestamp_unix')

        msc = msc.fillna(method='ffill')
#        msc = msc.fillna(method='bfill')
        msc.drop(index=msc.index[0], axis = 0, inplace = True) #remove first index
        
        msc = msc.reset_index(drop = True)
        
        gp = os.path.abspath(os.path.join(p, "../.."))       
        msc.to_csv(msc_fp, index = False, header=True)
        meltp.to_csv(meltp_fp, index = False, header=True)

    return [msc, meltp, gp]

def preprocess_data(dir_name, msc, gp):
    paths = get_filepaths(dir_name)
    md = pd.DataFrame() #initialize metadata table of settings information
    subj_id = os.path.basename(gp)
    
    md_fp = gp + '/' + subj_id + '_meta_data.csv'
    
    if (os.path.exists(md_fp)):
        md = pd.read_csv(md_fp, header=0)
    else:    
        for i in range(0, len(paths)):
            p = paths[i] + '/'
            # for testing purposes:
                #p = '/Users/mariaolaru/RCS02 Un-Synced Data/SummitData/SummitContinuousBilateralStreaming/RCS02L/Session1557952808964/Device'
                
            exists = len(find_file("timeDomainData.csv", p))
            if (exists == 0):
                print("Can't process, timeDomainData does not exist: \n" + p + "\n")
                continue
            else:
                print("Processing data: \n" + p + "\n")
            tdd = pd.read_csv(p + "timeDomainData.csv")
            tddp = preprocess_tdd(tdd, p)
            md = pd.concat([md, tddp])
    
        md = md.sort_values('timestamp')
        md = md.reset_index(drop=True)
        md.to_csv(md_fp, index = False, header=True)
    
    return md

def melt_md(md):
    """
    Parameters
    ----------
    md : dataframe, wide-form meta-data
    fs : int, sample rate
    out_name : str, filename of output file
    step_size : int, increment in seconds with which to group data

    Returns
    -------
    long-form of meta-data
    """  

    print("Converting meta data to long-form")
    md = md.rename(columns={"ch0_mv": "0", "ch1_mv": "1", "ch2_mv": "2", "ch3_mv": "3"})
    
    mdm = pd.melt(md, id_vars=['timestamp'], value_vars = ['0', '1', '2', '3'], var_name = 'channel', value_name = 'voltage')

    mdm['channel'] = mdm['channel'].astype(int)
    mdm = mdm[mdm['voltage'].notna()]
    
#    out_name_mdsm = gp + '/' + out_name + "_mdsm.csv"
#    mdsm.to_csv(out_name_mdsm, index = False)    
    return mdm

def link_data_wide(msc, md, gp):    
    subj_id = os.path.basename(gp)    
    ld_fp = gp + '/' + subj_id + '_linked_data_wide.csv'

    if (os.path.exists(ld_fp)):
        df_linked = pd.read_csv(ld_fp, header=0)
    else:    
        mscc = msc.drop(['subj_id', 'implant_side', 'timestamp', 'stim_status', 'stim_contact_an'], axis = 1)
        mscc = mscc.rename({'timestamp_unix' : 'timestamp'}, axis = 1)
 
        df_linked = mscc.append(md)    
        df_linked = df_linked.sort_values('timestamp').reset_index(drop = True)       
        df_linked.loc[:, mscc.columns] = df_linked.loc[:, mscc.columns].fillna(method='ffill')

        #df_linked = df_linked[df_linked['ch1_mV'].notna()]
        
        df_linked[['session_id']] = df_linked[['session_id']].astype(int)
        df_linked[['stim_contact']] = df_linked[['stim_contact']].astype(int)
        df_linked[['sr']] = df_linked[['sr']].astype(int)
        df_linked[['stim_pw']] = df_linked[['stim_pw']].astype(int)
    
        df_linked.to_csv(ld_fp, index = False, header=True)
        
    return df_linked


def link_data(msc, md, gp):
    subj_id = os.path.basename(gp)    
    ld_fp = gp + '/' + subj_id + '_linked_data.csv'

#    if (os.path.exists(ld_fp)):
#        df_linked = pd.read_csv(ld_fp, header=0)
#    else:    
    mscc = msc.drop(['subj_id', 'implant_side', 'timestamp', 'stim_status', 'stim_contact_an'], axis = 1)
    mscc = mscc.rename({'timestamp_unix' : 'timestamp'}, axis = 1)
    mdl = melt_md(md)

    df_linked_ch0 = link_ch(mscc, mdl, 0)
    df_linked_ch1 = link_ch(mscc, mdl, 1)
    df_linked_ch2 = link_ch(mscc, mdl, 2)
    df_linked_ch3 = link_ch(mscc, mdl, 3)
    
    df_linked = pd.concat([df_linked_ch0, df_linked_ch1, df_linked_ch2, df_linked_ch3])
    df_linked[['session_id']] = df_linked[['session_id']].astype(int)
    df_linked[['stim_contact']] = df_linked[['stim_contact']].astype(int)
    df_linked[['sr']] = df_linked[['sr']].astype(int)
    df_linked[['stim_pw']] = df_linked[['stim_pw']].astype(int)

    col = df_linked['sense_contacts']
    df_linked = df_linked.drop(columns = 'sense_contacts')
    df_linked.insert(1, 'sense_contacts', col)
    
    df_linked = df_linked.drop('channel', axis = 1)
    #df_linked = df_linked.rename(columns = {'stim_contact_cath':'stim_contact', 'amplitude_ma': 'stim_amp', 'pulsewidth_us': 'stim_pw', 'stimfrequency_hz':'stim_freq'})
    df_linked = df_linked.sort_values(['session_id', 'sense_contacts', 'timestamp']).reset_index(drop = True)   
    df_linked.to_csv(ld_fp, index = False, header=True)
        
    return df_linked

def concat_data(df):
    df['settings'] = df['stim_freq'].astype(str) + '-' + df['stim_amp'].astype(str)
    df = df.drop(['stim_contact', 'stim_pw', 'lpfs'], axis = 1)
    return df

def link_ch(mscc, mdl, label):
    mdl_ch = mdl[mdl['channel'] == label]
    
    if 'label' in mscc:
        mscc_ch = mscc[['session_id', 'timestamp', 'stim_contact', 'stim_amp', 'stim_pw', 'stim_freq', 'sr', 'ch' + str(label) +'_sense_contacts', 'ch' + str(label) + '_lpfs', 'label']]
    else:
        mscc_ch = mscc[['session_id', 'timestamp', 'stim_contact', 'stim_amp', 'stim_pw', 'stim_freq', 'sr', 'ch' + str(label) +'_sense_contacts', 'ch' + str(label) + '_lpfs']]
    
    df_linked_ch = mscc_ch.append(mdl_ch)

    df_linked_ch = df_linked_ch.sort_values('timestamp').reset_index(drop = True)
    
    df_linked_ch.loc[:, mscc_ch.columns] = df_linked_ch.loc[:, mscc_ch.columns].fillna(method='ffill')
    df_linked_ch = df_linked_ch[df_linked_ch['voltage'].notna()]
    df_linked_ch = df_linked_ch.rename(columns = {'ch' + str(label) + '_sense_contacts':'sense_contacts', 'ch' + str(label) + '_lpfs':'lpfs'})

    return df_linked_ch

def label_montage(msc, labels):
    labels = np.array(labels)
    sesh = msc['session_id'].unique()
    for i in range(len(sesh)):
        i_true = msc.index[msc['session_id'] == sesh[i]]
        msc.loc[i_true, 'label'] = labels[i]
    return msc








