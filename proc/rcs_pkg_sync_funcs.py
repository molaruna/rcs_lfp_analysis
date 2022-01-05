#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov  4 13:45:30 2021

@author: mariaolaru
"""

import scipy.signal as signal
import pandas as pd
import numpy as np
import glob
import os
import re
from matplotlib import pyplot as plt
pd.options.display.float_format = '{:.2f}'.format
#pd.reset_option('display.float_format')
import scipy.stats as stat
import sklearn
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.utils import Bunch
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn import metrics
from sklearn.metrics import roc_auc_score, roc_curve, auc
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.decomposition import PCA

from regressors import stats
import seaborn as sb

def import_apple_file(apple_dir, keyword):
    ['remor', 'yskinesia']
    if (keyword == 0):
        paths = glob.glob(apple_dir + '*remor*')
    elif (keyword == 1):
        paths = glob.glob(apple_dir + '*yskinesia*')
        
    gp = os.path.abspath(os.path.join(paths[0], "../..")) 
    subj_id = os.path.basename(gp)
    
    md = pd.DataFrame(columns = ['timestamp'])

    for i in range(0, len(paths)):
        p = paths[i]
        df = pd.read_csv(p)
        df['timestamp'] = df['time'].astype(int)
        if (p.find('yskinesia') != -1):
            df['apple_dk'] = df['probability']

        elif(p.find('remor') != -1):
            df['apple_tremor'] = df['probability']

        df = df.drop(['probability', 'time'], axis = 1)           
        md = md.merge(df, how = 'outer')   

    return md


def import_apple_files(apple_dir):
    md_tremor = import_apple_file(apple_dir, 0)  
    md_dk = import_apple_file(apple_dir, 1) 
    md_fp = '/Users/mariaolaru/Documents/temp/testing.csv'

    md = md_tremor.merge(md_dk, how = 'outer', on = 'timestamp')                    
    md = md.sort_values('timestamp')
    md = md.reset_index(drop=True)
    md.to_csv(md_fp, index = False, header=True)
    md['timestamp'] = md['timestamp'].astype(int)
    #md['timestamp'] = md['timestamp']*1000
    return md

def import_pkg_files(pkg_dir):
    paths = glob.glob(pkg_dir + 'scores*')
    gp = os.path.abspath(os.path.join(paths[0], "../..")) 
    subj_id = os.path.basename(gp)
    
    md_fp = pkg_dir + subj_id + '_pkg_data_combined.csv'
    md = pd.DataFrame([])
    
    if (os.path.exists(md_fp)):
        md = pd.read_csv(md_fp, header=0)
    else:    
        for i in range(0, len(paths)):
            p = paths[i]
            df = pd.read_csv(p)
            df['timestamp'] = pd.to_datetime(df['Date_Time']).astype(int) / 10**9
            md = pd.concat([md, df])
    
        md = md.sort_values('timestamp')
        md = md.reset_index(drop=True)
        md.to_csv(md_fp, index = False, header=True)
        md['timestamp'] = md['timestamp'].astype(int)
        md['timestamp'] = md['timestamp']*1000
    return md

def normalize_data(x, min_x, max_x):
    x_norm = (x-min_x)/(max_x-min_x)
    return x_norm

def preproc_apple(apple_dir):
    apple_df = import_apple_files(apple_dir)
    apple_df['timestamp'] = apple_df['timestamp'].astype(int)*1000
    #pkg_df['timestamp'] = pkg_df['timestamp'] +  60000 #add 1min to end time
    #pkg_df['timestamp'] = pkg_df['timestamp'] + 25200000 #add 7hr to end time (b/c Datetime in PT, not GMT)
    start_time = apple_df['timestamp'].head(1).values[0]
    stop_time = apple_df['timestamp'].tail(1).values[0]    
    return [apple_df, start_time, stop_time]

def remove_outliers(col):
    x = col.copy()
    y = x.median() + 5*x.std()
    indx = x[x > y].index
    x[indx] = np.nan
    return x

def preproc_pkg(pkg_dir):
    pkg_df = import_pkg_files(pkg_dir)
    pkg_df['timestamp'] = pkg_df['timestamp'].astype(int)*1000
    pkg_df['timestamp'] = pkg_df['timestamp'] +  60000 #add 1min to end time
    pkg_df['timestamp'] = pkg_df['timestamp'] + 25200000 #add 7hr to end time (b/c Datetime in PT, not GMT)

        
    #Include data where watch is on wrist, from even indices
    pkg_df = pkg_df[(pkg_df['Off_Wrist'] == 0)] 
    
    start_time = pkg_df['timestamp'].head(1).values[0]
    stop_time = pkg_df['timestamp'].tail(1).values[0]
    
    #BK score 40-80 = on couch; BK score > 80 = asleep
    pkg_df['inactive'] = 0
    indx = pkg_df[pkg_df['BK'] < -40].index
    pkg_df.loc[indx, 'inactive'] = 1 

    return [pkg_df, start_time, stop_time]

def preproc_notes(fp_notes, start_time, stop_time):
    df_notes = pd.read_csv(fp_notes)
    df_notes = df_notes.rename(columns = {'timestamp_unix': 'timestamp'})
    
    rm = np.where(np.isnan(df_notes['timestamp']))[0]
    df_notes = df_notes.drop(rm)
    
    df_notes['timestamp'] = df_notes['timestamp'].astype(int)
    df_notes = df_notes[(df_notes['timestamp'] >= start_time) & (df_notes['timestamp'] <= stop_time)]
    df_notes = df_notes.reset_index(drop=True)
    return df_notes

def preproc_phs(phs_fp, start_time, stop_time):
    phs_df = pd.read_csv(phs_fp)
    phs_df = phs_df.rename({'timestamp_end': 'timestamp'}, axis = 1)
    phs_df['timestamp'] = phs_df['timestamp'].astype(int)
    #subset data
    phs_df = phs_df[(phs_df['timestamp'] > start_time) & (phs_df['timestamp'] < stop_time)]
    
    #create wide phs df
    phs_dfw = phs_df.pivot_table(index = ['timestamp', 'contacts'], values = ['max_amp'], columns = ['band'])
    phs_dfw.columns = [f'{x}_{y}' for x,y in phs_dfw.columns]
    phs_dfw = phs_dfw.reset_index(level=['timestamp', 'contacts'])
    phs_dfw = phs_dfw.sort_values('timestamp')
    
    phs_dfwb = phs_dfw[phs_dfw['contacts'] == '+3-1']
    phs_dfwb = pd.concat([phs_dfwb['timestamp'], phs_dfwb['max_amp_beta']], axis = 1)
    
    phs_dfwg = phs_dfw[phs_dfw['contacts'] == '+3-1']
    phs_dfwg = pd.concat([phs_dfwg['timestamp'], phs_dfwg['max_amp_gamma']], axis = 1)

    #Merge phs tables
    phs_merged = pd.merge(phs_dfwb, phs_dfwg, how = 'inner', on = 'timestamp')
    
    #Round ts to nearest second
    phs_merged['timestamp'] = round(phs_merged['timestamp'], -3)/1000
    phs_merged['timestamp'] = phs_merged['timestamp'] * 1000
    phs_merged['timestamp'] = phs_merged['timestamp'].astype(int)

    return phs_merged

def preproc_psd(fp_psd, start_time, stop_time):
    psd_df = pd.read_csv(fp_psd)
    psd_df = psd_df.rename({'timestamp_end': 'timestamp'}, axis = 1)
#    psd_df['timestamp'] = psd_df['timestamp'].astype(int)*1000
    #subset data
    psd_df = psd_df[(psd_df['timestamp'] > start_time) & (psd_df['timestamp'] < stop_time)]
    
    #create wide phs df
    psd_dfw = psd_df.pivot_table(index = ['timestamp'], values = [psd_df.columns[1]], columns = ['f_0', 'contacts'])
#    psd_dfw = psd_dfw['spectra']
    
    psd_dfw.columns = [''.join(str(col).split()) for col in psd_dfw.columns]
    psd_dfw = psd_dfw.reset_index()

    return psd_dfw

def preproc_coh(fp_coh, start_time, stop_time, sr):
    df_coh = pd.read_csv(fp_coh)
    freqs = df_coh['freqs'].unique()
    if (sr == 250):
        freq_rm = freqs[np.array([np.where(np.round(freqs) == 21)[0][0], 
                         np.where(np.round(freqs) == 62)[0][0],
                         np.where(np.round(freqs) == 104)[0][0]])] #freqs are repeated
    else:
        print("WARNING, need to remove duplicate freqs for diff sr")
        
    i_rm = np.array([])
    for i in range(len(freq_rm)):
        i_rmc = np.where(df_coh.freqs == freq_rm[i])[0]
        i_rm = np.append(i_rmc, i_rm)

    df_coh = df_coh.drop(i_rm)   
    df_coh['freqs'] = np.round(df_coh['freqs'])
    df_coh = df_coh[(df_coh['timestamp'] > start_time) & (df_coh['timestamp'] < stop_time)]
    df_coh = df_coh.pivot_table(index = ['timestamp'], values = ['Cxy'], columns = ['freqs', 'contacts'])
#    df_coh = df_coh['Cxy']

    df_coh.columns = [''.join(str(col).split()) for col in df_coh.columns]
    df_coh = df_coh.reset_index()
    return df_coh

def merge_pkg_df(df_pkg, df_apple, df_phs, df_psd, df_coh, df_meds, df_dys):
    #Both dfs must have 'timestamp' column to merge on
    if (df_apple.empty == False):
        df_merged = pd.merge(df_apple, df_psd, how = 'inner', on = 'timestamp')

    elif (df_pkg.empty == False):
        df_merged = pd.merge(df_pkg, df_psd, how = 'inner', on = 'timestamp')

    if (df_phs.empty == False):
        df_merged = pd.merge(df_merged, df_phs, how = 'inner', on = 'timestamp')
    
    if (df_coh.empty == False):
        df_merged = pd.merge(df_merged, df_coh, how = 'inner', on = 'timestamp')


    if (df_meds.empty == False):
        df_merged = pd.merge(df_merged, df_meds, how = 'left', on = 'timestamp')
    
    if (df_dys.empty == False):
        df_merged = pd.merge(df_merged, df_dys, how = 'outer', on = 'timestamp')
    
    if (df_meds.empty == False):
        indx = np.where(np.isnan(df_merged['med_time']))[0]
        df_merged.loc[indx, 'med_time'] = 0
    
    if (df_dys.empty == False):
        indx = np.where(np.isnan(df_merged['dyskinesia']))[0]
        df_merged.loc[indx, 'dyskinesia'] = 0   
    
    df_merged = df_merged.sort_values('timestamp')
    df_merged = df_merged.reset_index(drop=True)
    return df_merged

def process_pkg(df_merged):
    return df_merged

def find_nan_chunks(col, num_nans):
    #find nan values
    x = col.isnull().astype(int).groupby(col.notnull().astype(int).cumsum()).cumsum()
    #find indices of nan vals >= num_nan
    indx_remove = x[x >= num_nans].index
    return indx_remove    

"""
def process_merged(df_merged):
    df_merged['BK_rev'] = df_merged['BK']*-1
    df_merged['BK_rev'] = df_merged['BK_rev'] + (df_merged['BK_rev'].min()*-1)
    
    #Remove outliers from each max_amp column separately
    df_merged['DK'] = remove_outliers(df_merged['DK'])
    df_merged['BK_rev'] = remove_outliers(df_merged['BK_rev'])
    
    #Normalize data from 0-1
    df_merged['BK'] = normalize_data(df_merged['BK_rev'], 0, np.nanmax(df_merged['BK_rev']))
    df_merged['DK'] = normalize_data(df_merged['DK'], 0, np.nanmax(df_merged['DK']))
    df_merged['phs_beta'] = normalize_data(df_merged['max_amp_beta'], np.nanmin(df_merged['max_amp_beta']), np.nanmax(df_merged['max_amp_beta']))
    df_merged['phs_gamma'] = normalize_data(df_merged['max_amp_gamma'], np.nanmin(df_merged['max_amp_gamma']), np.nanmax(df_merged['max_amp_gamma']))
    #df_merged['max_amp_diff_norm'] = df_merged['max_amp_beta_norm'] - df_merged['max_amp_gamma_norm']
    #df_merged['max_amp_diff_norm'] = normalize_data(df_merged['max_amp_diff'], 0, np.nanmax(df_merged['max_amp_diff']))

    cols = np.array(df_merged.columns)
    indices = [i for i, s in enumerate(list(cols)) if '+' in s]
    for i in indices:
        col_name = cols[i]
        df_merged[col_name] = normalize_data(df_merged[col_name], np.nanmin(df_merged[col_name]), np.nanmax(df_merged[col_name]))

    return df_merged
"""

def process_merged(df_merged, dt):

    if (dt == 'PKG'):
        df_merged['BK_rev'] = df_merged['BK']*-1
        df_merged['BK_rev'] = df_merged['BK_rev'] + (df_merged['BK_rev'].min()*-1)
        
        #Remove outliers from each max_amp column separately
        df_merged['DK'] = remove_outliers(df_merged['DK'])
        df_merged['BK_rev'] = remove_outliers(df_merged['BK_rev'])
        
        #Normalize data from 0-1
        df_merged['BK'] = normalize_data(df_merged['BK_rev'], 0, np.nanmax(df_merged['BK_rev']))
        df_merged['DK'] = normalize_data(df_merged['DK'], 0, np.nanmax(df_merged['DK']))

    if (dt == 'phs'):
        df_merged['phs_beta'] = normalize_data(df_merged['max_amp_beta'], np.nanmin(df_merged['max_amp_beta']), np.nanmax(df_merged['max_amp_beta']))
        df_merged['phs_gamma'] = normalize_data(df_merged['max_amp_gamma'], np.nanmin(df_merged['max_amp_gamma']), np.nanmax(df_merged['max_amp_gamma']))
        #df_merged['max_amp_diff_norm'] = df_merged['max_amp_beta_norm'] - df_merged['max_amp_gamma_norm']
        #df_merged['max_amp_diff_norm'] = normalize_data(df_merged['max_amp_diff'], 0, np.nanmax(df_merged['max_amp_diff']))

    cols = np.array(df_merged.columns)
    indices = [i for i, s in enumerate(list(cols)) if '+' in s]
    for i in indices:
        col_name = cols[i]
        df_merged[col_name] = normalize_data(df_merged[col_name], np.nanmin(df_merged[col_name]), np.nanmax(df_merged[col_name]))

    return df_merged

def process_dfs(df_pkg, df_apple, df_phs, df_psd, df_coh, df_meds, df_dys):
    df_merged = merge_pkg_df(df_pkg, df_apple, df_phs, df_psd, df_coh, df_meds, df_dys)

    if (df_pkg.empty == False):
        df_merged = process_merged(df_merged, 'PKG')
        
    if (df_apple.empty == False):
        df_merged = process_merged(df_merged, 'apple')
        
    if (df_coh.empty == False):
        df_merged = process_merged(df_merged, 'phs')
        print('warning: need to combine PKG & phs scoring')
        
    return df_merged

def add_sleep_col(df_merged):
    df_merged['timestamp_dt'] = pd.to_datetime(df_merged['Date_Time'])
    df_merged['timestamp_dt_h'] = [i.hour for i in df_merged['timestamp_dt']]
    df_merged['asleep'] = (df_merged['timestamp_dt_h'] >= 22) | (df_merged['timestamp_dt_h'] <= 8)
    df_merged['asleep'] = df_merged['asleep'].astype(int)
    return df_merged

def plot_pkg_sync(df_merged, freq_band, contacts):
    #Plotting
    #plt.plot(np.arange(1, len(df_mergedfb)+1, 1), df_mergedfb['BK_norm'], alpha = 0.7, label = 'PKG-BK', markersize = 1, color = 'indianred')
    
    #plt.plot(np.arange(1, len(df_mergedfb)+1, 1), df_mergedfb['max_amp_gamma_norm'], alpha = 0.7, label = 'RCS-gamma', markersize = 1, color = 'darkorange')
    #plt.plot(np.arange(1, len(df_mergedfb)+1, 1), df_mergedfb['max_amp_beta_norm'], alpha = 0.7, label = 'RCS-beta', markersize = 1, color = 'mediumpurple')
    breaks = find_noncontinuous_seg(df_merged['timestamp'])
    
    title = ("freq_band: " + str(freq_band) + "Hz")
    plt.title(title)
    plt.rcParams["figure.figsize"] = (30,3.5)

    #plt.plot(np.arange(1, len(df_merged)+1, 1), df_merged[(contacts[0]+'_norm')], alpha = 0.7, label = contacts[0], markersize = 1, color = 'orchid')
    #plt.plot(np.arange(1, len(df_merged)+1, 1), df_merged[(contacts[1] + '_norm')], alpha = 0.7, label = contacts[1], markersize = 1, color = 'mediumpurple')
    plt.plot(np.arange(1, len(df_merged)+1, 1), df_merged[(contacts[2] + '_norm')], alpha = 0.7, label = contacts[2], markersize = 1, color = 'darkkhaki')
    #plt.plot(np.arange(1, len(df_merged)+1, 1), df_merged[(contacts[3]+'_norm')], alpha = 0.7, label = contacts[3], markersize = 1, color = 'darkorange')

    plt.vlines(df_merged[df_merged['dyskinesia'] == 1].index, 0, 1, color = 'black', label = 'dyskinesia')
    plt.vlines(df_merged[df_merged['med_time'] == 1].index, 0, 1, color = 'green', label = 'meds taken')
    plt.vlines(np.where(df_merged['asleep'] == 1)[0], 0, 1, alpha = 0.1, label = 'asleep', color = 'grey')
    plt.vlines(breaks, 0, 1, alpha = 0.7, label = 'break', color = 'red')

    plt.plot(np.arange(1, len(df_merged)+1, 1), df_merged['DK_norm'], alpha = 0.7, label = 'PKG-DK', markersize = 1, color = 'steelblue')
    
    #plt.plot(df_mergedfb['timestamp'], df_mergedfb['max_amp_diff_norm'], alpha = 0.7, markersize = 1, label = 'RCS (beta-gamma)')
    plt.legend(ncol = 5, loc = 'upper right')
    plt.ylabel('scores (normalized)')
    plt.xlabel('time (samples)')

    out_dir = '/Users/mariaolaru/Documents/temp/RCS02/RCS02L_pkg_rcs/plots'
    plt.savefig(out_dir + '/' + 'psd_' + 'freq' + str(freq_band) + '.pdf')
    plt.close()
 
def plot_corrs(df_corr, PKG_measure, out_dir):
    plt.close()
    df_corr_s = df_corr.loc[PKG_measure, :]
    df_corr_s.freq_band = df_corr_s.freq_band.astype('float')
    plt.rcParams["figure.figsize"] = (5,5)
    plt.tight_layout()

    plt.plot(df_corr_s['freq_band'], df_corr_s.iloc[:, 2], label = df_corr_s.columns[2], color = 'orchid')
    plt.plot(df_corr_s['freq_band'], df_corr_s.iloc[:, 3], label = df_corr_s.columns[3], color = 'mediumpurple')
    plt.plot(df_corr_s['freq_band'], df_corr_s.iloc[:, 0], label = df_corr_s.columns[0], color = 'darkkhaki')
    plt.plot(df_corr_s['freq_band'], df_corr_s.iloc[:, 1], label = df_corr_s.columns[1], color = 'darkorange')

    plt.title(PKG_measure)
    plt.legend(ncol = 2, loc = 'upper right')
    plt.ylim(-1, 1)
    plt.ylabel('Pearson coef (r)')
    plt.xlabel('Frequency (Hz)')
    
    plt.savefig(out_dir + '/' + 'psd_corr_' + PKG_measure + '.pdf')
    #plt.close()    
    
def get_med_times():
    #Initialize times for May 15th - May 30th
    #7:30AM, 10:30AM, 1:30PM, 4:30PM, 10:30PM (also @ 2:30 AM, but not added)
    df_meds = pd.DataFrame(columns = ['timestamp', 'med_time'])
    ts = np.array([1557905400000, 1557916200000, 1557927000000, 1557937800000, 1557959400000])
    ts = ts + 60000 #add 1min to match w/ pkg times
    ts = ts + 25200000 #add 7hrs to be in PT 
    ts_curr = ts
    #only append last 5 values     
    for i in range(16):
        ts_curr = ts_curr+86400000
        ts = np.append(ts, ts_curr)
    
    df_meds['timestamp'] = ts
    df_meds['med_time'] = 1
    
    return df_meds

def find_noncontinuous_seg(timestamps):
    seg = 60000*10 #more than 10 minutes of a break
    indx = np.array([])
    
    for i in range(1,len(timestamps)):
        ts = timestamps[i]
        ts_prev = timestamps[i-1]
        if (ts - ts_prev > seg):
            indx = np.append(indx, i)
    return indx
        
def find_dyskinesia(df_notes):
    indx = np.where(df_notes['symptoms'].str.contains('Dyskinesia') == True)[0]
    df_dys = df_notes.loc[indx, ['timestamp', 'symptoms']]
    df_dys = df_dys.rename(columns = {'symptoms': 'dyskinesia'})
    df_dys['dyskinesia'] = 1
    return df_dys

def compute_correlation(df_merged, keyword, corr_vals):

    indices = [j for j, s in enumerate(list(df_merged.columns)) if keyword in s]    
    col_names = df_merged.columns[indices]   
    
    freqs = get_frequencies(col_names)
    df_corr = pd.DataFrame([])

    cols = df_merged.columns
    for freq in freqs:
        freq_str ="('" + keyword + "'," + str(freq)
        indices = [i for i, s in enumerate(list(cols)) if freq_str in s]        
        col_names = np.array(cols[indices])
                
        contacts = np.array([])
        for i in range(len(col_names)):
            x = re.findall("([^']*)", col_names[i])[6]
            contacts = np.append(contacts, x)
        
        col_names = np.append(col_names, corr_vals)
    
        #get corr coefs
        df_vars = df_merged.loc[:, col_names]
        
        for i in range(len(contacts)):
            col_name = df_vars.columns[i]
            df_vars = df_vars.rename(columns={col_name:contacts[i]})
            
        df_corr_ind = df_vars.corr()
    
        df_corr_ind = df_corr_ind.round(3)
        df_corr_ind['freq_band'] = freq
        df_corr = pd.concat([df_corr, df_corr_ind])
    return [df_corr, contacts]

def split(df_merged, feature_key, target_key):
    feature_i = [x for x, s in enumerate(list(df_merged.columns)) if feature_key in s]
    target_i = np.where(df_merged.columns == target_key)[0]

    return [feature_i, target_i]

def shuffle(df_merged, feature_col_indxs, target_col_indx):
    df_temp = df_merged.copy()
    df_temp = df_temp.dropna().reset_index(drop=True)
    
    x = df_temp.iloc[:, feature_col_indxs].to_numpy()
    x_names = df_temp.columns[feature_col_indxs]
    y = df_temp.iloc[:, target_col_indx].to_numpy()

    [x_train, x_test, y_train, y_test] = train_test_split(x, y, test_size = 0.2, random_state = 42)

    return [x_train, x_test, x_names, y_train, y_test]

def binarize(y_train, thresh):
    y_train_bin = y_train.copy()
    y_train_bin[y_train_bin <= thresh] = 0
    y_train_bin[y_train_bin > thresh] = 1
    return y_train_bin

def run_SVM(x_train, x_test, y_train, thresh):
    clf = svm.SVC(kernel='linear', probability = True) # Linear Kernel        

    clf.fit(x_train, y_train.ravel())
    y_score = clf.predict_proba(x_test) 
    df_coefs = clf.coef_[0]

    return [df_coefs, y_score]    

def run_SVM_classtest(df_merged, class_vals, runs):
    df_stats = pd.DataFrame(columns = ['accuracy', 'precision', 'recall'])
    df_clean = df_merged.copy()
    df_clean = df_clean.dropna()
    df_clean = df_clean.reset_index(drop=True)
    
    cols = np.array(df_clean.columns)
    feature_col_indxs = [x for x, s in enumerate(list(cols)) if '+' in s]
    feature_names = df_clean.columns[feature_col_indxs]

    df_coefs = pd.DataFrame()
    df_coefs['features'] = feature_names
    for j in class_vals:
        print(str(j))
        val = j
        df_clean['DK_class'] = 0
        indxs = np.where(df_clean['DK'] > val)[0]
        df_clean.loc[indxs, 'DK_class'] = 1
        
        df_data = df_clean.iloc[:, feature_col_indxs]
        df_data = df_data.dropna()
        data = df_data.to_numpy()
        feature_names = df_data.columns
        target = df_clean['DK_class'].to_numpy()
        target_names = np.array(['present', 'absent', 'cutoff_value'])
        
        dyskinesia = Bunch(data = data, feature_names = feature_names, target = target, target_names = target_names)
        
        df_features = pd.DataFrame([])
        #df_features['names'] = feature_names
        accuracy = np.array([])
        precision = np.array([])
        recall = np.array([])
        for i in range(runs):
            print(str(i))
            # Split dataset into training set and test set
            X_train, X_test, y_train, y_test = train_test_split(dyskinesia.data, dyskinesia.target, test_size=0.2)
            
            #Create classifier
            clf = svm.SVC(kernel='linear') # Linear Kernel
            
            #Train the model using the training sets
            clf.fit(X_train, y_train)
            
            #Predict the response for test dataset
            y_pred = clf.predict(X_test)
            accuracy = np.append(accuracy, metrics.accuracy_score(y_test, y_pred))
            precision = np.append(precision, metrics.precision_score(y_test, y_pred))
            recall = np.append(recall, metrics.recall_score(y_test, y_pred))
            
            #get feature importance
            df_features[str(i)] = clf.coef_[0]
        
        df_coefs[str(j)] = df_features.mean(axis=1)    
        data = [[accuracy.mean(), precision.mean(), recall.mean(), val]]
        df_ind = pd.DataFrame(data, columns =['accuracy', 'precision', 'recall', 'cutoff_value'])
        df_stats = pd.concat([df_stats, df_ind])
        
        print("Accuracy:", str(accuracy.mean()))
        print("Precision:", str(precision.mean()))
        print("Recall:", str(recall.mean()))
    return [df_stats, df_coefs]
    
def compute_ROC(y_test, y_score, n_classes):
    fpr= dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(n_classes):
        fpr, tpr, _ = roc_curve(y_test, y_score[:, i])
        roc_auc = auc(fpr, tpr)        
    return fpr, tpr, roc_auc
 
def plot_ROC(fpr, tpr, roc_auc, num_class):
    plt.close()
    plt.figure()
    lw = 2
    plt.rcParams["figure.figsize"] = (5,5)
    plt.tight_layout()
    plt.plot(fpr, tpr, color="darkorange", lw=lw,
             label = "ROC curve (area = %0.2f)" % roc_auc)
    plt.plot([0, 1], [0, 1], color="navy", lw=lw, linestyle="--")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Receiver operating characteristic")
    plt.legend(loc="lower right")
    plt.show()

def get_contacts(col_names):
    contacts = np.array([])
    for i in range(len(col_names)):
        x = re.findall("([^']*)", col_names[i])[6]
        contacts = np.append(contacts, x)    
    return np.unique(contacts)

def get_frequencies(col_names):
    frequencies = np.array([])
    for i in range(len(col_names)):
        x = col_names[i].split(',')[1]
        x = float(x)
        frequencies = np.append(frequencies, x)    
    return np.unique(frequencies)

def plot_SVM_coefs(df_coefs, coef):
    #assumes we are have data for 0.0Hz
    #find contact channels
    indices = [i for i, s in enumerate(list(df_coefs['features'])) if '(0.0' in s]
    col_names = np.array(df_coefs.loc[indices, 'features']) 
    contacts = get_contacts(col_names)
    colors = ['darkkhaki', 'darkorange', 'orchid', 'mediumpurple']

    plt.close()
    plt.rcParams["figure.figsize"] = (5,5)
    plt.tight_layout()

    for i in range(len(contacts)):
        channel = contacts[i]            
        indices = [j for j, s in enumerate(list(df_coefs['features'])) if channel in s]    
        col_names = np.array(df_coefs.loc[indices, 'features'])
        freqs = get_frequencies(col_names)
        plt.plot(freqs, df_coefs.loc[indices, str(coef)], alpha = 0.7, label = channel, color = colors[i])

    plt.title('SVM feature importance')
    plt.legend(ncol = 2, loc = 'upper right')
    plt.ylim(-1, 1)
    plt.ylabel('Permutation importance')
    plt.xlabel('Frequency (Hz)')
    
def add_classes(col, class_thresh, labels):
    #assumes data min val is 0
    temp = pd.DataFrame([])
    temp['score'] = col
    temp['class'] = 0
    
    for i in range(len(class_thresh)):
        if i == 0:
            min_val = 0
        else:
            min_val = class_thresh[i-1]
        max_val = class_thresh[i]
        indx = np.where((temp['score'] <= max_val) & (temp['score'] > min_val))[0]
        temp.loc[indx, 'class'] = labels[i]
    return temp['class'].values
    
def run_lr(lm_type, x_train, x_test, y_train, y_test):

    if lm_type == 'base':
        linear_regressor = sklearn.linear_model.LinearRegression()
        linear_regressor.fit(x_train, y_train) #perform linear regression

    elif lm_type == 'lasso':
        linear_regressor = sklearn.linear_model.Lasso()
        linear_regressor.fit(x_train, y_train) #perform linear regression
        
    y_pred = linear_regressor.predict(x_test) #make predictions

    #print("Coefficients: \n", linear_regressor.coef_)
    print("Mean squared error: %.2f" % mean_squared_error(y_test, y_pred))
    print("Coefficient of determination: %.2f" % r2_score(y_test, y_pred))

    coef_pvals = stats.coef_pval(linear_regressor, x_train, y_train)

    return [linear_regressor.coef_[0], coef_pvals[1:len(coef_pvals)], y_pred]
   
def col2mat(coefs, coef_pvals, feature_names):

    df_lm = pd.DataFrame()
    df_lm['features'] = feature_names
    df_lm['coefs'] = coefs
    df_lm['coefs'] = pd.to_numeric(df_lm.coefs, errors = 'coerce')
    df_lm['coef_pvals'] = coef_pvals
    #df_lm.dropna(inplace=True)
    
    contacts = get_contacts(feature_names)

#    indices = [j for j, s in enumerate(list(df_lm['features'])) if '+' in s]    
    
    indices = [j for j, s in enumerate(list(df_lm['features'])) if contacts[0] in s]    
    col_names = np.array(df_lm.loc[indices, 'features'])
    freqs = get_frequencies(col_names)
    
    df_lm_heat = pd.DataFrame(index=range(len(contacts)), columns = range(len(freqs)))
    df_lm_heat_p = pd.DataFrame(index=range(len(contacts)), columns = range(len(freqs)))
    df_lm_heat_sig = pd.DataFrame(index=range(len(contacts)), columns = range(len(freqs)))
    
    for i in range(len(contacts)):
        channel = contacts[i]
        indices = [j for j, s in enumerate(list(df_lm['features'])) if "'" + channel + "'" in s]    
        df_lm_heat.iloc[i, :] = df_lm.loc[indices, 'coefs'].values
        df_lm_heat_p.iloc[i, :] = df_lm.loc[indices, 'coef_pvals'].values
        
        df_lm_heat_sig.iloc[i,:] = 0
        indx = np.where(df_lm_heat_p.iloc[i,:] < 0.05)[0]
        df_lm_heat_sig.iloc[i,indx] = df_lm_heat.iloc[i,:][indx]

    return [df_lm_heat_sig, contacts]

def plot_heatmap(heatmap, freqs, contacts, title):
    max_val = max(abs(heatmap).max(axis=1))

    plt.close()
    fig, ax = plt.subplots(figsize=(40, 6))
    sb.heatmap((heatmap.astype(float)), cmap = 'PiYG', vmin = max_val *-1, vmax = max_val)
    #img = ax.imshow(df_lm_heat, cmap='copper', interpolation='nearest', origin = 'lower')
    
    ax.set_yticks(np.arange(0, len(contacts),1)+0.5)
    ax.set_yticklabels(contacts, rotation = 90)

    ax.set_xticks(np.arange(0, len(freqs),5) + 0.5) 
    ax.set_xticklabels(np.arange(0, len(freqs),5), rotation = 45) 
 
    ax.set_title(title)
    ax.set(xlabel = 'frequency band (Hz)')
    ax.set(ylabel = 'channel')
    plt.show()

def run_lm_wrapper(df, feature_key, target_key, lm_type, heatmap):
    [feature_i, target_i] = split(df, feature_key, target_key)
    [x_train, x_test, x_names, y_train, y_test] = shuffle(df, feature_i, target_i)

    #run linear regression
    [coefs, coef_pvals, y_pred] = run_lr(lm_type, x_train, x_test, y_train, y_test)
    df_coefs = pd.DataFrame([])
    df_coefs['coefs'] = coefs
    df_coefs['pvals'] = coef_pvals
    df_coefs['features'] = x_names
    
    [df_lm_heat_sig, contacts] = col2mat(df_coefs['coefs'], df_coefs['pvals'], df_coefs['features'])
    
    if heatmap == 2:
        freqs = np.linspace(0, 125, 126).astype(int)
        plot_heatmap(df_lm_heat_sig, freqs, contacts, 'linear regression: significant coefficients')

    return df_coefs

"""     
    feature_names = df.columns[feature_i]
    [df_lm_heat, df_lm_heat_p, df_lm_heat_sig] = col2mat(coefs, coef_pvals, feature_names)
    
    row_labels = get_contacts(feature_names)
   
    if (heatmap == 0):
        plot_heatmap(df_lm_heat, row_labels, "all beta values")
    if (heatmap == 1):
        plot_heatmap(df_lm_heat_p, row_labels, "all p-values")
    if (heatmap == 2):
        plot_heatmap(df_lm_heat_sig, row_labels, "all significant beta values")
"""

    
def run_pca(df, key, ncomponents, option):
    """
    Parameters
    ----------
    df : input merged dataframe
    key : partial mature from df column headers for feature input
    ncomponents : number of components in PCA analysis
    option: 0 = run PCA with current formatting, 1 = transform data for PCA
        
    Returns
    -------
    None.

    """
    
    df_temp = df.copy()
    df_temp = df.dropna().reset_index(drop=True)
    
    #get all features
    feature_col_indxs = [x for x, s in enumerate(list(df_temp.columns)) if key in s]

    df_preproc = df_temp.iloc[:, feature_col_indxs]
    
    if option == 1:
        df_preproc = df_preproc.T

    x = df_preproc.to_numpy()
    x_colnames = df_temp.columns[feature_col_indxs]

    pca = PCA(n_components=ncomponents)
    pcs = pca.fit_transform(x)    

    df_pcs = pd.DataFrame(data = pcs)
    df_pcs = df_pcs.add_prefix('PC')
        
    var_ratio = pca.explained_variance_ratio_
    return [df_pcs, var_ratio] 

def run_pca_wrapper(df, keys, ncomponents, option, out_dir):
    df_pcs = pd.DataFrame([])
    for i in range(len(keys)):
        [df_pc, test_vr] = run_pca(df, keys[i], ncomponents, option)
        df_pc = df_pc.add_suffix('_' + keys[i])
        df_pcs = pd.concat([df_pcs, df_pc], axis = 1)
        plot_pcs(df_pc.iloc[:, 0:ncomponents], keys[i], out_dir)
    return df_pcs

def plot_pcs(df_pcs, key, out_parent_dir):
    plt.close()
    nlen = df_pcs.shape[0]    
    npcs = df_pcs.shape[1]
    
    for i in range(npcs):
        label = 'PC' + str(i+1)
        plt.plot(np.linspace(0, nlen-1, nlen), df_pcs.iloc[:, i], label = label)
    if nlen < 1002:
        opt = 'freq'
        plt.xlabel('Frequency (Hz)')
    else:
        opt = 'td'
        plt.xlabel('Time (samples)')
    plt.ylabel('eigenvalues')
    plt.title(key)
    plt.legend()
    
    out_dir = out_parent_dir + 'plots'
    plt.savefig(out_dir + '/' + 'pca_' + opt + '_' + key + '.pdf')
    plt.savefig(out_dir + '/' + 'pca_' + opt + '_' + key + '.svg')

    
def run_svm_wrapper(df, feature_key, target_key, thresh):
    [feature_i, target_i] = split(df, feature_key, target_key)
    [x_train, x_test, x_names, y_train, y_test] = shuffle(df, feature_i, target_i)
    
    #run SVM
    y_train_bin = binarize(y_train, thresh)
    y_test_bin = binarize(y_test, thresh)
    [df_coefs, y_score] = run_SVM(x_train, x_test, y_train_bin, thresh)
    
    #compute & plot ROC curve
    n_classes = y_score.shape[1]
    [fpr, tpr, roc_auc] = compute_ROC(y_test_bin, y_score, n_classes)
    
    num_class = 1
    plot_ROC(fpr, tpr, roc_auc, num_class)
    
    return [df_coefs, x_names]

def get_top_features(coefs, x_names):
    df_coefs = pd.DataFrame([])
    df_coefs['coefs'] = coefs
    df_coefs['features'] = x_names
    df_coefs = df_coefs.sort_values(by = 'coefs').reset_index(drop=True)
    df_coefs_imp = df_coefs.drop(range(10, len(df_coefs)-10))
    df_top_pos = df_coefs_imp.iloc[::-1].iloc[0:10, :].reset_index(drop=True)
    df_top_neg = df_coefs_imp.iloc[0:10, :]
    return [df_top_pos, df_top_neg]
    
def add_classes_wrapper(df, min_thresh):
    df_lda = df.copy()
    df_lda = df_lda[df_lda['DK'] > min_thresh].reset_index(drop=True)
    df_lda['DK_log'] = np.log10(df_lda['DK'])
    df_lda['DK_log'] = df_lda['DK_log'] - df_lda['DK_log'].min()
    class_thresh = np.nanpercentile(df_lda['DK_log'], [20, 40, 60, 80, 100])
    labels = [1, 2, 3, 4, 5]
    df_lda['DK_class'] = add_classes(df_lda['DK_log'], class_thresh, labels)
    
    indx = df_lda[df_lda['DK_class'] == 0].index
    df_lda['DK_class'][indx] = 1

    return [df_lda, class_thresh]
       
def plot_classes(col, label, class_thresh):
    plt.close()
    plt.plot(col)
    
    for i in range(len(class_thresh)):
        plt.hlines(class_thresh[i], 0, len(col), alpha = 1, color = 'red')
    plt.ylabel(label)
    plt.xlabel('time (samples)')
    
def run_lda(df, feature_key, target):
    feature_i = [x for x, s in enumerate(list(df.columns)) if feature_key in s]
    X = df.iloc[:, feature_i].to_numpy() #assumes 10 PCs
    y = df.loc[:, target].to_numpy()
    #y_bi = df_temp.loc[:, 'DK_class_binary'].to_numpy()    

    out = sklearn.model_selection.cross_val_score(LinearDiscriminantAnalysis(), X, y, cv = 10)
    avg_acc = np.mean(out)
    sem = stat.sem(out)
    return [avg_acc, sem]

    