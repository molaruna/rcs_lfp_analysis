# RCS14_entrainment_naive.py
# Generate timeseries analysis and power estimate
# Author: maria.olaru@
"""
Created on Mon May  3 18:22:44 2021

@author: mariaolaru
"""

import numpy as np
import os
import scipy.signal as signal
import pandas as pd
import math
from preproc.preprocess_funcs import convert_unix2dt
from fooof import FOOOF


def name_file(mscs, gp):

    out_plot_dir = gp + '/' + 'tables/' 
    
    if not os.path.isdir(out_plot_dir):
        os.mkdir(out_plot_dir)

    subj_id = mscs['subj_id'].iloc[0]
    session_id = str(int(mscs['session_id'].iloc[0]))
    out_name = subj_id + '_' + session_id
    
    out_fp = out_plot_dir + out_name

    return out_fp

def qc_msc(mscs):
    num_fs = len(mscs['ch1_sr'].unique())
    num_sesh = len(mscs['session_id'].unique())
    num_amps = len(mscs['amplitude_ma'].unique())
    mscs_qc = mscs.iloc[:, 5:len(mscs.columns)-3].drop(['stim_contact_an'], axis = 1)
    if ('timestamp' in mscs_qc.columns):
        mscs_qc = mscs_qc.drop(['timestamp'], axis = 1)
    
    mscs_qc_avg = mscs_qc.mean()
    
    if (num_fs > 1):
        raise Exception("Timestamp range cannot have multiple sampling rates")

    if (num_sesh > 1):
        raise Exception("Timestamp range cannot have multiple session ids... code still inp")

    if (num_amps > 1):
        raise Exception("Timestamp range cannot have multiple stim amplitudes... code still inp")

    if ((round((mscs_qc_avg - mscs_qc), 3)!=0).all().any()):
        raise Exception("A setting was not held constant in current session... code still inp")
    return    

def subset_msc(msc, ts_range):
    mscs_start = msc[(msc['timestamp_unix'] <= ts_range[0])].tail(1)
    mscs_end = msc[(msc['timestamp_unix'] >= ts_range[0]) & (msc['timestamp_unix'] <= ts_range[1])]
    if mscs_start.equals(mscs_end):
        return mscs_start
    else:
        mscs = pd.concat([mscs_start, mscs_end])
        return mscs

def subset_md(md, mscs, fs, ts_range, ts_int = None):
    """
    Parameters
    ----------
    md : meta data output from preprocess_data()
    ts_range: vector of UNIX timestamps with min and max values
    Returns
    -------
    a subset of the meta data with modified timestamps 
        
    """
    print("Subsetting the meta data")
    ts_min = md['timestamp_unix'].iloc[md['timestamp_unix'].sub(ts_range[0]).abs().idxmin()]
    ts_max = md['timestamp_unix'].iloc[md['timestamp_unix'].sub(ts_range[1]).abs().idxmin()]
    md_i_min = md[md['timestamp_unix'] == ts_min].index[0]
    md_i_max = md[md['timestamp_unix'] == ts_max].index[0]

    mds = md.iloc[md_i_min:md_i_max, :]

    if (ts_int != None):
        i_int = mscs[mscs['timestamp'] == ts_int].index[0]    
        amp1 = mscs['amplitude_ma'].iloc[i_int-1]
        amp2 = mscs['amplitude_ma'].iloc[i_int]
        mds = mds.assign(amp=np.where(mds['timestamp_unix'] < ts_int, amp1, amp2))   
        mds = md.loc[md_i_min:md_i_max, :]
        mds = mds.reset_index(drop=True)

    ts_dt = convert_unix2dt(mds['timestamp_unix'])
    mds.insert(1, 'timestamp', ts_dt)
    tt = len(mds)/fs/60
   
    return [mds, tt]

def convert_psd_montage(df, sr):
    """
    Parameters
    ----------
    df : linked dataframe

    Returns
    -------
    df_psd : dataframe, power spectra using Welch's in long-form

    """
    print('Creating PSDs for sr' + str(sr))
    cols = df['sense_contacts'].unique()
    labels = np.array(df['label'].unique())
    
    df_collection = {}
    
    for label in labels:
        df_collection[label] = pd.DataFrame(columns = ['f_0'])
        for i in range(len(cols)):
            dfs = df[(df['label'] == label) & (df['sr'] == sr) & (df['sense_contacts'] == cols[i])] 
            df_curr = convert_psd(dfs['voltage'], sr, cols[i])
            df_collection[label] = df_collection[label].merge(df_curr, how = 'outer')
    
    return df_collection

def convert_psd(voltage, fs, col_header):
    f_0, Pxx_den = signal.welch(voltage, fs, average = 'median', window = 'hann', nperseg=fs)
    psd_ind = pd.DataFrame(np.array([f_0, Pxx_den]).T, columns=['f_0', col_header])
    return psd_ind


def convert_psd_wide(df, sr):
    #df is a linked df
    stim_freqs = np.sort(df['stim_freq'].unique())
    sense_contacts = df['sense_contacts'].unique()
    
    df_collection = {}
    
    for stim_freq in stim_freqs:
        df_collection[stim_freq] = {}
        dfs = df[df['stim_freq'] == stim_freq]
        stim_amps = np.sort(dfs['stim_amp'].unique())
        for stim_amp in stim_amps:
            df_collection[stim_freq][stim_amp] = pd.DataFrame(columns = ['f_0'])
            for i in range(len(stim_amps)):
                dfss = dfs[dfs['stim_amp'] == stim_amp]
                for j in range(len(sense_contacts)):
                    dfsss = dfss[dfss['sense_contacts'] == sense_contacts[j]]
                    df_psd_ind = convert_psd(dfsss['voltage'], sr, sense_contacts[j])
                    df_collection[stim_freq][stim_amp] = df_collection[stim_freq][stim_amp].merge(df_psd_ind, how = 'outer')
    return df_collection
   
def convert_psd_long_old(df, gp, contacts, time_duration, time_overlap, sr, spec):
    """    
    Parameters
    ----------
    df : timedomain data (not linked)
    spec: type of psd (period, aperiodic, gross)

    Returns
    -------
    df_psd : dataframe, power spectra using Welch's in long-form

    #Assumes sense contacts don't change    
    """
    
    out_name = os.path.basename(gp)
    df_psd = pd.DataFrame() #initialize metadata table of settings information
    nperseg = time_duration * sr
    overlapseg = time_overlap * sr
            
    channels = df.columns[1:5]
    for i in range(len(channels)):
        ch = channels[i]
        df_ch = df.loc[:,ch]
        df_ch.name = contacts[i]
        indx_start = int(0)
        indx_stop = int(nperseg)
        num_spectra = math.floor((len(df_ch)-nperseg)/(nperseg - overlapseg))

#        ts_sec = df['timestamp'].round(-3)/1000
#        ts_wholemin = ts_sec.iloc[np.where(ts_sec % 60 == 0)[0]]

        for j in range(num_spectra):
            ts_start = int(df.iloc[indx_start, 0])
            ts_stop = int(df.iloc[indx_stop, 0])
            ts_sec = round(ts_stop, -3)/1000
            if (ts_sec % 60 != 0):
                indx_start = int(indx_start + (nperseg - overlapseg))
                indx_stop = int(indx_stop + (nperseg - overlapseg))
                continue
                
            ts_diff = ts_stop - ts_start
            if (ts_diff > (time_duration*1000)*1.1):
                indx_start = int(indx_start + (nperseg - overlapseg))
                indx_stop = int(indx_stop + (nperseg - overlapseg))
                continue
            
            else:            
                print("ch: " + contacts[i] + " (" + str(j) + "/" + str(num_spectra) + ")")
                voltage = df_ch[indx_start:indx_stop] 
                nan_indx = np.where(np.isnan(voltage) == True)[0]
                if (len(nan_indx) > 0):
                    if (len(nan_indx) > 0.1*len(voltage)):
                        continue
                    else:
                        voltage = voltage.drop(voltage.index[nan_indx])
                #print(indx_start)
                #print(indx_stop)
                #print(ts_tail)
                df_psd_ind = convert_psd(voltage, sr, 'spectra')
                if (spec == 'aperiodic'):
                    spectrum = df_psd_ind[df_psd_ind.columns[len(df_psd_ind.columns)-1]]
                    freq_range = [4, 100]
                    fooof_psd = flatten_psd(df_psd_ind['f_0'], spectrum, freq_range)
                    df_psd_ind = fooof_psd.loc[:, ['f_0', 'fooof_peak_rm']]
                if (spec == 'periodic'):
                    spectrum = df_psd_ind[df_psd_ind.columns[len(df_psd_ind.columns)-1]]
                    freq_range = [4, 100]
                    fooof_psd = flatten_psd(df_psd_ind['f_0'], spectrum, freq_range)
                    df_psd_ind = fooof_psd.loc[:, ['f_0', 'fooof_flat']]
                df_psd_ind['contacts'] = df_ch.name
                df_psd_ind['timestamp'] = int(ts_sec*1000)
                df_psd = pd.concat([df_psd, df_psd_ind])
                indx_start = int(indx_start + (nperseg - overlapseg))
                indx_stop = int(indx_stop + (nperseg - overlapseg))

    out_name = gp + '/' + os.path.basename(gp) + '_psd_' + spec + '.csv'
    #df_psd['timestamp'] = df_psd['timestamp'].astype(int).round(-3)/1000
    df_psd.to_csv(out_name, index = False)

    return df_psd

def convert_coh_long_old(md_ds, contacts, gp, time_duration, time_overlap, sr):

    out_name = os.path.basename(gp)
    df_coh = pd.DataFrame() #initialize metadata table of settings information
    nperseg = time_duration * sr
    overlapseg = time_overlap * sr
          
    cx = [contacts[0], contacts[0], contacts[1], contacts[1]]
    cy = [contacts[2], contacts[3], contacts[2], contacts[3]]
    for i in range(len(cx)):
        x = md_ds[[cx[i]]]
        y = md_ds[[cy[i]]]
    
        indx_start = int(0)
        indx_stop = int(nperseg)
        num_spectra = math.floor((len(md_ds)-nperseg)/(nperseg - overlapseg))
    
        for j in range(num_spectra):
            ts_start = int(md_ds.iloc[indx_start, 0])
            ts_stop = int(md_ds.iloc[indx_stop, 0])
            ts_sec = round(ts_stop, -3)/1000
            if (ts_sec % 60 != 0):
                indx_start = int(indx_start + (nperseg - overlapseg))
                indx_stop = int(indx_stop + (nperseg - overlapseg))
                continue
                
            ts_diff = ts_stop - ts_start
            if (ts_diff > (time_duration*1000)*1.1):
                indx_start = int(indx_start + (nperseg - overlapseg))
                indx_stop = int(indx_stop + (nperseg - overlapseg))
                continue
            
            else:            
                print(x.columns[0] + '_' + y.columns[0] + " (" + str(j) + "/" + str(num_spectra) + ")")
                voltage_x = x[indx_start:indx_stop] 
                voltage_y = y[indx_start:indx_stop]
    
                #voltage_x = remove_nan(voltage_x)
                #voltage_y = remove_nan(voltage_y)
    
                df_coh_ind = pd.DataFrame([])
                
                [f, Cxy] = signal.coherence(voltage_x.iloc[:,0], voltage_y.iloc[:,0], fs = sr)
                df_coh_ind['Cxy'] = Cxy
                df_coh_ind['freqs'] = f
                df_coh_ind['timestamp'] = int(ts_sec*1000)
                df_coh_ind['contacts'] = x.columns[0] + '_' + y.columns[0]
          
                df_coh = pd.concat([df_coh, df_coh_ind])
                indx_start = int(indx_start + (nperseg - overlapseg))
                indx_stop = int(indx_stop + (nperseg - overlapseg))

    out_name = gp + '/' + os.path.basename(gp) + '_coh.csv'
    #df_psd['timestamp'] = df_psd['timestamp'].astype(int).round(-3)/1000
    df_coh.to_csv(out_name, index = False)
    
    return df_coh

def remove_nan(voltage):
    nan_indx = np.where(np.isnan(voltage) == True)[0]
    if (len(nan_indx) > 0):
        if (len(nan_indx) < 0.1*len(voltage)):
            voltage = voltage.drop(voltage.index[nan_indx])
    return voltage

def make_phs_wide(df_psds, gp):
    """
    Parameters
    ----------
    df_psds : output of convert_psd_wide -- dict object of stim frequencies, with dict objects of stim amplitudes, with df_psds

    Returns
    -------
    df_phs_all : TYPE
        DESCRIPTION.

    """
    df_phs_all = pd.DataFrame()    
    stim_freqs = np.array([*df_psds.keys()])
    for stim_freq in stim_freqs:
        df_sf = df_psds[stim_freq]
        stim_amps = np.array([*df_sf.keys()])
        for stim_amp in stim_amps:
            df_psd_sense = df_sf[stim_amp]
            sense_contacts = df_psd_sense.columns[1:len(df_psd_sense.columns)]
            for i in range(len(sense_contacts)):
                spectrum = df_psd_sense[sense_contacts[i]]
                df_flat = flatten_psd(df_psd_sense['f_0'], spectrum, [4, 100])
                df_phs_ind = compute_phs(df_flat['f_0'], df_flat['fooof_flat'])
                df_phs_ind['stim_amp'] = stim_amp
                df_phs_ind['stim_freq'] = stim_freq
                df_phs_ind['contacts'] = sense_contacts[i]
                df_phs_all = pd.concat([df_phs_all, df_phs_ind])
    out_name = gp + '/' + os.path.basename(gp) + '_phs.csv'
    df_phs_all.to_csv(out_name, index = False)
    return df_phs_all

def make_phs_old(df, contacts, sr, freq_range, time_duration, time_overlap, gp):
    """
    Parameters
    ----------
    Assumes that sr is constant & contacts are constant

    df : md
    freq_range : frequency range across which to model spectrum with fooof
    time_duration : amount of time for each PSD, in seconds
    time_overlap : amount of overlap between PSD calculations, in seconds

    Returns
    -------
    peak height scores with timestamp for last entry used
    Automatically outputs results in 1m increments to mimic PKG results
    """
    nperseg = time_duration * sr
    overlapseg = time_overlap * sr
    
    df_phs_all = pd.DataFrame()
    
    channels = df.columns[1:5]
    for i in range(len(channels)):
        ch = channels[i]
        df_ch = df.loc[:,ch]
        df_ch.name = contacts[i]
        indx_start = 0
        indx_stop = nperseg
        num_spectra = math.floor((len(df_ch)-nperseg)/(nperseg - overlapseg))
        for j in range(num_spectra):
            
            ts_start = int(df.iloc[indx_start, 0])
            ts_stop = int(df.iloc[indx_stop, 0])
            ts_sec = round(ts_stop, -3)/1000
            if (ts_sec % 60 != 0):
                indx_start = int(indx_start + (nperseg - overlapseg))
                indx_stop = int(indx_stop + (nperseg - overlapseg))
                continue
                
            ts_diff = ts_stop - ts_start
            if (ts_diff > (time_duration*1000)*1.1):
                indx_start = int(indx_start + (nperseg - overlapseg))
                indx_stop = int(indx_stop + (nperseg - overlapseg))
                continue
            else: 
                print("ch: " + contacts[i] + " (" + str(j) + "/" + str(num_spectra) + ")")
                voltage = df_ch[indx_start:indx_stop]
                nan_indx = np.where(np.isnan(voltage) == True)[0]
                if (len(nan_indx) > 0):
                    if (len(nan_indx) > 0.1*len(voltage)):
                        continue
                    else:
                        voltage = voltage.drop(voltage.index[nan_indx])

                #print(indx_start)
                #print(indx_stop)
                ts_tail = int(ts_sec*1000)
                #print(ts_tail)
                df_phs_ind = make_phs_ind(voltage, sr, freq_range, ts_tail)
                df_phs_all = pd.concat([df_phs_all, df_phs_ind])
                indx_start = indx_start + (nperseg - overlapseg)
                indx_stop = indx_stop + (nperseg - overlapseg)

    out_name = gp + '/' + os.path.basename(gp) + '_phs.csv'
    df_phs_all.to_csv(out_name, index = False)

    return df_phs_all


def make_phs(df, sr, freq_range, time_duration, time_overlap, gp):
    """
    Parameters
    ----------
    Assumes that sr is constant

    df : output of link_data()
    freq_range : frequency range across which to model spectrum with fooof
    time_duration : amount of time for each PSD, in seconds
    time_overlap : amount of overlap between PSD calculations, in seconds

    Returns
    -------
    peak height scores with timestamp for last entry used
    """

    nperseg = time_duration * sr
    overlapseg = time_overlap * sr
    
    df_phs_all = pd.DataFrame()
    
    channels = df['sense_contacts'].unique()
    for i in range(len(channels)):
        ch = channels[i]
        df_ch = df[df['sense_contacts'] == ch]['voltage'].reset_index(drop=True)
        df_ch.name = ch

        indx_start = 0
        indx_stop = nperseg
        num_spectra = math.floor((len(df_ch)-nperseg)/(nperseg - overlapseg))

        for j in range(num_spectra):
            print("ch: " + ch + " (" + str(j) + "/" + str(num_spectra) + ")")
            voltage = df_ch[indx_start:indx_stop] 
            nan_indx = np.where(np.isnan(voltage) == True)[0]
            if (len(nan_indx) > 0):
                if (len(nan_indx) > 0.1*len(voltage)):
                    continue
                else:
                    voltage = voltage.drop(voltage.index[nan_indx])
            ts_tail = df.loc[indx_stop, 'timestamp']
            df_phs_ind = make_phs_ind(voltage, sr, freq_range, ts_tail)
            df_phs_all = pd.concat([df_phs_all, df_phs_ind])
            indx_start = indx_start + (nperseg - overlapseg)
            indx_stop = indx_stop + (nperseg - overlapseg)

    out_name = gp + '/' + os.path.basename(gp) + '_phs.csv'
    df_phs_all.to_csv(out_name, index = False)

    return df_phs_all

def make_phs_ind(voltage, sr, freq_range, ts_end):
    """
    Parameters
    ----------
    Assumes there is a single set of contacts being used

    df : output of link_data()
    freq_range : frequency range across which to model spectrum with fooof
    time_duration : amount of time for each PSD, in seconds
    overlap : amount of overlap between PSD calculations, in seconds

    Returns
    -------
    peak height scores with timestamp for last entry used
    """
    
    col_name = voltage.name
    df_psd = convert_psd(voltage, sr, col_name)

    spectrum = df_psd[df_psd.columns[len(df_psd.columns)-1]]
    df_flat = flatten_psd(df_psd['f_0'], spectrum, freq_range)
    
    if (df_flat['fooof_flat'].unique()[0] == None):
        df_phs = pd.DataFrame()
        return df_phs       

    df_phs = compute_phs(df_flat['f_0'], df_flat['fooof_flat'])
    df_phs.insert(0, 'contacts', spectrum.name)
    df_phs.insert(1, 'timestamp_end', ts_end)

    return df_phs

def flatten_psd(f_0, spectrum, freq_range):
    """ 
    Parameters
    ----------
    freq_range : frequency range across which to model spectrum
    out_name : str, name of out file

    Returns
    -------
    df_phs : dataframe, peak height scores

    """
    fm = FOOOF() #initialize fooof object
    br = freq_range[0]
    tr = freq_range[1]
    
    fm.fit(f_0.to_numpy(), spectrum.to_numpy(), freq_range)
    
    df = pd.DataFrame()
    df['contacts'] = [spectrum.name] * (tr-br+1)
    df['f_0'] = f_0[br:tr+1].to_numpy()
    df['spectrum'] = spectrum
    df['fooof_spectrum'] = fm.fooofed_spectrum_
    df['fooof_peak_rm'] = fm._spectrum_peak_rm
    df['fooof_flat'] = fm._spectrum_flat
    return df

def compute_phs(f_0, spect_flat):

    """
    both inputs must be Series pd objects
    """   
    data = {'f_0':f_0.to_numpy(), 'spectrum':spect_flat.to_numpy()}
    df_in = pd.DataFrame(data)
    
    df_out = pd.DataFrame(columns = ['band', 'freq', 'max_amp'])
    bands = ['theta', 'alpha', 'beta', 'low-gamma', 'gamma']
    bands_freqs = [[4, 8], [8, 13], [13, 30], [30, 60], [60, 90]]
    df_out['band'] = bands

    for i in range(len(df_out['band'])):
        band_freqs = bands_freqs[i]
        df_band = df_in[(df_in['f_0']> band_freqs[0]) & (df_in['f_0'] <= band_freqs[1])]
        max_amp = df_band['spectrum'].max()
        max_freq = df_band[df_band['spectrum'] == max_amp]['f_0'].values[0]        
        df_out.loc[i, 'freq'] = max_freq
        df_out.loc[i, 'max_amp'] = max_amp
   
    return df_out


def compute_diff(col1, col2):
    """
    Parameters
    ----------
    col1 : 1D array, returns mid-point between each value in row X and X+1
    col2 : 1D array, y-axis values from which derivatives are taken for each value in row X and X+1

    Returns
    -------
    diff_df : dataframe, col1 + col2

    """
 
    col1_sub = col1[1:len(col1)].reset_index(drop=True)
    col1_mean = pd.Series(np.mean(np.stack((col1[0:len(col1)-1], col1_sub)), axis=0))

    col2_diff = col2.diff()[1:len(col2)].reset_index(drop=True)
    col2_name = col2_diff.name + '_diff'
    
    diff_df = pd.DataFrame({col1.name: col1_mean, col2_name: col2_diff})
    
    return diff_df


def dfp_spect(dfp, fs, ch, padding):
    b, a = signal.butter(2, 0.5)
    
    dfps = dfp.loc[dfp['channel'] == str(ch)]
    filtered = signal.filtfilt(b, a, dfps['voltage'])
    f, t, Sxx = signal.spectrogram(filtered, fs)
    t = t - padding[0]
    
    return [f, t, Sxx]

def compute_msc(mdsm, fs, out_name):
    """
    Parameters
    ----------
    mdsm : dataframe, output of melt_mds()
    fs : int, sampling rate
    out_name : str, filename of output

    Returns
    -------
    df_msc : dataframe, mean squared coherence
    """
    df_msc = pd.DataFrame() #initialize metadata table of settings information
    ch_cxys = [[1, 3], [1, 4], [2, 3], [2,4]]

    for i in range(len(ch_cxys)):
        ch_subcort = ch_cxys[i][0]
        ch_cort = ch_cxys[i][1]
        
        for j in range(len(mdsm['step'].unique())):
            mdsms = mdsm[mdsm['step'] == j]
            x = mdsms[mdsms['channel'] == ch_subcort]['voltage']
            y = mdsms[mdsms['channel'] == ch_cort]['voltage']
            [f, Cxy] = signal.coherence(x, y, fs)
            msc_ind = pd.DataFrame(np.array([f, Cxy]).T, columns=['f_0', 'Cxy'])

            ts_start = mdsms['timestamp'].iloc[0]
            msc_ind.insert(1, 'timestamp_start', ts_start)
            msc_ind.insert(2, 'ch_subcort', ch_subcort)
            msc_ind.insert(3, 'ch_cort', ch_cort)
            msc_ind.insert(4, 'step', mdsm['step'].unique()[j])
            
            df_msc = pd.concat([df_msc, msc_ind])
    out_name_msc = out_name + "_msc.csv"
    df_msc.to_csv(out_name_msc, index = False)
    return df_msc
"""
def compute_bsn(mdsm, freq_thresh, burst_dur, fs):

    Note: dataframes have additional column of 'condition' * will fix soon
    Burst minimum duration is 25ms
    
    Parameters
    ----------
    mdsm : dataframe, output of melt_mds()
    freq_thresh : int array of length 2, frequencies with which to bandpass
    burst_dur : int, minimum duration necessary for a burst in ms
    fs : int, sampling rate

    Returns
    -------
    bsn: dataframe, burst score number

    #Step 1: independently bandpass using 2nd order butterworth, then smooth
    mdsm['voltage_filtered'] = 0
    mdsm['voltage_rectified'] = 0
    mdsm['voltage_smoothed'] = 0
    mdsm['thresh_mV'] = 0
    mdsm['avg_thresh_mv'] = 0
    
    for x in range(len(mdsm['condition'].unique())):
        condition = mdsm['condition'].unique()[x]

        for i in range(len(mdsm['channel'].unique())):
            channel = i + 1
            order = 3
            bp = freq_thresh
            b, a = signal.butter(order, bp, btype = 'bandpass', fs = fs)
            indices = mdsm[(mdsm['condition'] == condition) & (mdsm['channel'] == channel)].index
            filtered = signal.lfilter(b, a, mdsm.loc[indices, 'voltage'])           
            mdsm.loc[indices, 'voltage_filtered'] = filtered

            mdsm.loc[indices, 'voltage_rectified'] = abs(filtered)

            dur_kernel = 25 #in ms
            fs_ms = (1/fs)*1000 #ms dur of each fs
            dp_smoo = int(dur_kernel/fs_ms)

            smooth = mdsm.loc[indices, 'voltage_rectified'].rolling(window=dp_smoo, center = True).mean()
            mdsm.loc[indices, 'voltage_smoothed'] = smooth
            
            thresh = np.nanpercentile(mdsm.loc[indices, 'voltage_smoothed'], 75)
            mdsm.loc[indices, 'thresh_mV'] = thresh
               
    #Step 2: determine whether signal passes threshold for each ts
    for x in range(len(mdsm['channel'].unique())):
        channel = mdsm['channel'].unique()[x]
        indices = mdsm[(mdsm['channel'] == channel)].index
        mdsm.loc[indices,'avg_thresh_mV'] = mdsm[mdsm['channel'] == channel]['thresh_mV'].mean()
    
    mdsm['thresh_pass'] = mdsm['voltage_smoothed'] > mdsm['avg_thresh_mV'] #calculate average again

    #Step 4: find periods longer than minimum duration that pass threshold
    df_bsr = compute_burstdur(mdsm, burst_dur, fs)
              

    #temp plotting for beta tests
    ss = mdsm[(mdsm['channel'] == 1) & (mdsm['condition'] == 'OFF')]
    ss = ss.reset_index(drop=True)

    out_fp = '/Users/mariaolaru/google_drive/UCSF_Neuroscience/starr_lab_MO/studies/gamma_entrainment/figures/aim1_plots'            
    from matplotlib import pyplot as plt   
    
    #plt.plot(ss.loc[1:200, 'voltage'])
    plt.plot(ss.loc[0:300, 'voltage_filtered'], label = "filtered")
    plt.plot(ss.loc[0:300, 'voltage_rectified'], label = 'rectified')
    plt.plot(ss.loc[0:300, 'voltage_smoothed'], label = 'smoothed')
    plt.axhline(mdsm['thresh_mV'].unique()[0], label = 'thresh', color = 'red')
    plt.axvline(75, label = "burst start")
    plt.legend()
    plt.savefig(out_fp + '/OFF_bsn_example.svg')

    #Step 4: Determine how many chunks pass threshold for each step using inflection pt             
    return df_bsr
"""
"""
def compute_burstdur(mdsm, burst_dur, fs):

    fs_ms = (1/fs)*1000 #ms dur of each fs
    burst_thresh = burst_dur/fs_ms #minimum datapoints needed above thresh

    df_bsr = pd.DataFrame() #initialize metadata table of settings information
    for x in range(len(mdsm['condition'].unique())):
        condition = mdsm['condition'].unique()[x]
        for i in range(len(mdsm['channel'].unique())):
            channel = mdsm['channel'].unique()[i]
            for j in range(len(mdsm[mdsm['condition'] == condition]['step'].unique())): 
                step = mdsm['step'].unique()[j]
                count = 0
                num_bursts = 0
                burst = False
                ss = mdsm[(mdsm['condition'] == condition) & (mdsm['channel'] == channel) & (mdsm['step'] == step)]
                ss = ss.reset_index(drop = True)
                timestamp = ss.head(1)['timestamp'][0]
                for k in range(len(ss)):
                    if (ss.loc[k, 'thresh_pass'] == True):
                        count = count + 1
                        if ((count >= burst_thresh) & (burst == False)):
                            num_bursts = num_bursts + 1
                            burst = True
                    else:
                        count = 0
                        burst = False
                bsr_ind = pd.DataFrame([[timestamp, condition, channel, step, num_bursts]], columns=['timestamp', 'condition', 'channel', 'step', 'bsr'])
                df_bsr = pd.concat([df_bsr, bsr_ind])
    
    return df_bsr
"""

def downsample_data(msc, df, out_sr):
    """
    Downsample entire dataframe to out_sr
    """
    ts_ranges = find_fs_ranges(msc, df, out_sr)
    if ts_ranges.size == 0:
        return df
    df_ds = df.copy()
    msc_ds = msc.copy()
    for i in range(len(ts_ranges)):
        print("Downsampling chunk: " + str(i) + "/" + str(len(ts_ranges)))
        dfc = df_ds[(df_ds.loc[:, 'timestamp'] >= ts_ranges['timestamp_start'][i]) & (df_ds.loc[:, 'timestamp'] <= ts_ranges['timestamp_stop'][i])]
        if (dfc.size) == 0:
            continue        
        msc_indx = np.where((msc.loc[:, 'timestamp_unix'] >= ts_ranges['timestamp_start'][i]) & (msc.loc[:, 'timestamp_unix'] <= ts_ranges['timestamp_stop'][i]))[0][0]
        in_sr = msc.loc[msc_indx,'sr']      
        dfp = downsample_data_chunk(dfc, in_sr, out_sr)
        df_ds.iloc[dfp.index, :] = dfp
        msc_ds.loc[msc_indx, 'sr'] = out_sr  
    df_ds = df_ds[df_ds.iloc[:,1].notna()]
        
    return [msc_ds, df_ds]

def downsample_data_chunk(df, in_sr, out_sr): #check ts for this option
    dfp = df.copy()
    indx = np.arange(len(dfp.columns)-4, len(dfp.columns), 1)
    ch_list = dfp.columns[indx]    
    for i in range(len(ch_list)):
        ch_label = ch_list[i]
        voltage = dfp.loc[:, ch_label]
        dfp.loc[:, ch_label] = downsample_ch(voltage, in_sr, out_sr)  
        
    ts_start = dfp['timestamp'].min()
    ts_end = dfp['timestamp'].max()
    
    ts_ds = dfp['timestamp'][0::int(in_sr/out_sr)].values
    ts_na = np.array([np.nan] * len(ts_ds) * (int(in_sr/out_sr) - 1))
    ts_total = np.concatenate([ts_ds, ts_na])
    dfp.loc[:, 'timestamp'] = ts_total[0:len(dfp)]

    return dfp

def downsample_ch(voltage, in_sr, out_sr):    
    secs = len(voltage)/in_sr # Number of seconds in signal X
    samps = int(secs*out_sr)     # Number of samples to downsample    
    ds_signal = np.array(signal.resample(voltage, samps))
    remaining_signal = np.array([np.nan] * (len(voltage) - samps))
                                                 
    Y = np.concatenate([ds_signal, remaining_signal]) 
    return Y

def find_fs_ranges(msc, md, fs):
    """
    Find timestamp ranges that do not contain the desired/input fs
    """
    md_ts_tail = md['timestamp'].tail(1).values[0]

    indx = np.where(msc['sr'] > fs)[0]
    if indx.size == 0:
        print('Warning – no samples to downsample in this dataset')
        ts_range = pd.DataFrame()
        return ts_range
    ts_change = msc.loc[indx, 'timestamp_unix']

    ts_start = msc.loc[indx, 'timestamp_unix']
    indx_stop = indx+1
    
    if (indx[(len(indx)-1)] == msc.tail(1).index[0]):
        indx_stop = indx_stop[0:(len(indx)-1)]
        
    ts_stop = msc.loc[indx_stop,'timestamp_unix']-1

    if (indx[(len(indx)-1)] == msc.tail(1).index[0]):
        ts_stop = ts_stop.append(pd.Series(md_ts_tail))
        
    ts_start = ts_start.reset_index(drop=True)    
    ts_stop = ts_stop.reset_index(drop=True)
         
    ts_range = pd.concat([ts_start, ts_stop], axis = 1).reset_index(drop=True)   
    ts_range.columns = ['timestamp_start', 'timestamp_stop']
    return ts_range

"""   
def find_ts_range(msc, md_ts_tail):
    drop_columns = np.array(['timestamp_unix', 'timestamp'])

    msc_compare = msc.drop(drop_columns, axis = 1)

    row_comp = pd.Series(np.zeros(shape = len(msc_compare.columns)))
    ts_change = np.array([])
    for i in range(len(msc_compare)):
        row_curr = msc_compare.iloc[i,:]
        if (row_comp.values != row_curr.values).any():
            ts_change = np.append(ts_change, i)            
            row_comp = row_curr

    ts_start = msc['timestamp_unix'][ts_change]
    ts_stop = msc['timestamp_unix'][ts_change[1:len(ts_change)]]-1


    ts_stop = ts_stop.append(pd.Series(md_ts_tail))
        
    ts_start = ts_start.reset_index(drop=True)    
    ts_stop = ts_stop.reset_index(drop=True)
         
    ts_range = pd.concat([ts_start, ts_stop], axis = 1).reset_index(drop=True)   
    ts_range.columns = ['timestamp_start', 'timestamp_stop']
    return [ts_range, ts_change]
"""
    
def find_stimfreq_artifact(md, msc):
    """
    Remove first 10s of recordings following a stimulation frequency change

    """
    stimfreq_prev = msc.loc[0, 'stim_freq']
    ts_start = np.array([])
    ts_stop = np.array([])
    for i in range(len(msc)):
        stimfreq_curr = msc.loc[i, 'stim_freq']
        if (stimfreq_curr != stimfreq_prev).any():
            ts_start = np.append(ts_start, msc.loc[i, 'timestamp_unix'])
            if (i == len(msc)-1):
                ts_start_next = md['timestamp'].tail(1)-1
            else:
                ts_start_next = msc.loc[i+1, 'timestamp_unix']
            ts_dur = (ts_start_next - (ts_start[len(ts_start)-1]))/1000
            if (ts_dur < 10): 
                ts_stop = ts_start + ts_dur
            else: 
                ts_stop = ts_start + (1000*10)
            stimfreq_prev = stimfreq_curr
        
    #ts_start = ts_start.reset_index(drop=True)    
    #ts_stop = ts_stop.reset_index(drop=True)
         
    ts_range = pd.DataFrame({'timestamp_start': ts_start, 'timestamp_stop': ts_stop})
    return ts_range  

def find_short_recordings(md, msc, min_dur):
    ts_start = np.array([])
    ts_stop = np.array([])
    for i in range(len(msc)):
        ts_curr = msc.loc[i, 'timestamp_unix']
        if (i == len(msc)-1):
            ts_start_next = md['timestamp'].tail(1)-1
        else:
            ts_start_next = msc.loc[i+1, 'timestamp_unix']
        ts_dur = (ts_start_next - ts_curr)/1000
        if (ts_dur < min_dur).any():
            ts_start = np.append(ts_start, ts_curr)
            ts_stop = np.append(ts_stop, ts_start_next)
                 
    ts_range = pd.DataFrame({'timestamp_start': ts_start, 'timestamp_stop': ts_stop})
    return ts_range  
    

def remove_stim_artifact(md, msc):
    mds = md.copy()
    ts_range = find_stimfreq_artifact(md, msc)
    for i in range(len(ts_range)):
        mds = mds[(mds['timestamp'] < ts_range.loc[i, 'timestamp_start']) | (mds['timestamp'] > ts_range.loc[i, 'timestamp_stop'])] 
    return mds
 

def remove_short_recordings(mds, msc, min_dur):
    mdb = mds.copy()
    ts_range = find_short_recordings(mdb, msc, min_dur)
    for i in range(len(ts_range)):
        mdb = mdb[(mdb['timestamp'] < ts_range.loc[i, 'timestamp_start']) | (mdb['timestamp'] > ts_range.loc[i, 'timestamp_stop'])] 
    return mdb
        
def get_entrainment_score(df_phs):
    df_phs_g = df_phs[(df_phs['band'] == 'gamma') & (df_phs['contacts'] == '+9-11')].reset_index(drop=True)    
    indx_noentrain = df_phs_g[(df_phs_g['freq'] < (df_phs_g['stim_freq']/2-2)) | (df_phs_g['freq'] > (df_phs_g['stim_freq']/2+2))].index
    df_entrain = df_phs_g.copy()
    df_entrain.loc[indx_noentrain, 'max_amp'] = 0
    return df_entrain

