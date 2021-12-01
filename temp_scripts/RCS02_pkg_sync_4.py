#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 21 11:38:47 2021

@author: mariaolaru
exclude night-time data (10PM - 8AM)
"""

import pandas as pd
import numpy as np
import glob
import os
from matplotlib import pyplot as plt
pd.options.display.float_format = '{:.2f}'.format
#pd.reset_option('display.float_format')


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

    return [pkg_df, start_time, stop_time]

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
    
    phs_dfwg = phs_dfw[phs_dfw['contacts'] == '+11-9']
    phs_dfwg = pd.concat([phs_dfwg['timestamp'], phs_dfwg['max_amp_gamma']], axis = 1)

    #Merge phs tables
    phs_merged = pd.merge(phs_dfwb, phs_dfwg, how = 'inner', on = 'timestamp')
    return phs_merged

def preproc_psd(fp_psd, start_time, stop_time):
    psd_df = pd.read_csv(fp_psd)
    psd_df = psd_df.rename({'timestamp_end': 'timestamp'}, axis = 1)
#    psd_df['timestamp'] = psd_df['timestamp'].astype(int)*1000
    #subset data
    psd_df = psd_df[(psd_df['timestamp'] > start_time) & (psd_df['timestamp'] < stop_time)]
    
    #create wide phs df
    psd_dfw = psd_df.pivot_table(index = ['timestamp'], values = ['spectra'], columns = ['f_0', 'contacts'])
    psd_dfw = psd_dfw['spectra']
    

    return psd_dfw

def merge_pkg_df(df_pkg, df):
    #Both dfs must have 'timestamp' column to merge on
    df_merged = pd.merge(df_pkg, df, how = 'inner', on = 'timestamp')
    df_merged = df_merged.sort_values('timestamp')
    df_merged = df_merged.reset_index(drop=True)
    return df_merged

def process_pkg(df_merged):
    df_merged['BK_rev'] = df_merged['BK']*-1
    df_merged['BK_rev'] = df_merged['BK_rev'] + (df_merged['BK_rev'].min()*-1)
    
    #Remove outliers from each max_amp column separately
    df_merged['DK'] = remove_outliers(df_merged['DK'])
    df_merged['BK_rev'] = remove_outliers(df_merged['BK_rev'])
    
    #Normalize data from 0-1
    df_merged['BK_norm'] = normalize_data(df_merged['BK_rev'], 0, np.nanmax(df_merged['BK_rev']))
    df_merged['DK_norm'] = normalize_data(df_merged['DK'], 0, np.nanmax(df_merged['DK']))
    return df_merged

def find_nan_chunks(col, num_nans):
    #find nan values
    x = col.isnull().astype(int).groupby(col.notnull().astype(int).cumsum()).cumsum()
    #find indices of nan vals >= num_nan
    indx_remove = x[x >= num_nans].index
    return indx_remove    

def process_psd(df_merged):
    #df_merged['max_amp_beta_norm'] = normalize_data(df_merged['max_amp_beta'], np.nanmin(df_merged['max_amp_beta']), np.nanmax(df_merged['max_amp_beta']))
    #df_merged['max_amp_gamma_norm'] = normalize_data(df_merged['max_amp_gamma'], np.nanmin(df_merged['max_amp_gamma']), np.nanmax(df_merged['max_amp_gamma']))
    #df_merged['max_amp_diff_norm'] = df_merged['max_amp_beta_norm'] - df_merged['max_amp_gamma_norm']
    #df_merged['max_amp_diff_norm'] = normalize_data(df_merged['max_amp_diff'], 0, np.nanmax(df_merged['max_amp_diff']))
    cols = np.array(df_merged.columns)
    indices = [i for i, s in enumerate(list(cols)) if '+' in s]
    for i in indices:
        col_name = cols[i]
        df_merged[col_name] = normalize_data(df_merged[col_name], np.nanmin(df_merged[col_name]), np.nanmax(df_merged[col_name]))
    return df_merged

def process_dfs(df_pkg, df_lfp):
    df_merged = merge_pkg_df(df_pkg, df_lfp)

    df_merged = process_pkg(df_merged)
    df_merged = process_psd(df_merged)
    #Remove data with large nan blocks
    #indx_nan_chunks = find_nan_chunks(df_merged['+10-8'], 3)
    #indx_nan_chunks = find_nan_chunks(df_merged.max_amp_gamma, 3)
    #df_merged_trim = df_merged.drop(indx_nan_chunks, axis = 0)
     
    #indx_nan_chunks_2 = find_nan_chunks(df_merged_trim.DK, 6)
    #df_merged_trim = df_merged_trim.drop(indx_nan_chunks_2, axis = 0)
    #Fill remaining nan values
    #df_mergedf = df_merged_trim.ffill().reset_index(drop=True)
    #df_mergedfb = df_mergedf.bfill().reset_index(drop=True)
    
    return df_merged

def plot_pkg_sync(df_merged, freq_band):
    #Plotting
    #plt.plot(np.arange(1, len(df_mergedfb)+1, 1), df_mergedfb['BK_norm'], alpha = 0.7, label = 'PKG-BK', markersize = 1, color = 'indianred')
    
    #plt.plot(np.arange(1, len(df_mergedfb)+1, 1), df_mergedfb['max_amp_gamma_norm'], alpha = 0.7, label = 'RCS-gamma', markersize = 1, color = 'darkorange')
    #plt.plot(np.arange(1, len(df_mergedfb)+1, 1), df_mergedfb['max_amp_beta_norm'], alpha = 0.7, label = 'RCS-beta', markersize = 1, color = 'mediumpurple')
    title = ("freq_band: " + str(freq_band) + "Hz")
    plt.title(title)
    plt.rcParams["figure.figsize"] = (30,3)
    #plt.plot(np.arange(1, len(df_merged)+1, 1), df_merged[(contacts[0]+'_norm')], alpha = 0.7, label = contacts[0], markersize = 1, color = 'orchid')
    plt.plot(np.arange(1, len(df_merged)+1, 1), df_merged[(contacts[1] + '_norm')], alpha = 0.7, label = contacts[1], markersize = 1, color = 'mediumpurple')
    #plt.plot(np.arange(1, len(df_merged)+1, 1), df_merged[(contacts[2] + '_norm')], alpha = 0.7, label = contacts[2], markersize = 1, color = 'darkkhaki')
    plt.plot(np.arange(1, len(df_merged)+1, 1), df_merged[(contacts[3]+'_norm')], alpha = 0.7, label = contacts[3], markersize = 1, color = 'darkorange')

    plt.plot(np.arange(1, len(df_merged)+1, 1), df_merged['DK_norm'], alpha = 0.7, label = 'PKG-DK', markersize = 1, color = 'steelblue')
    
    #plt.plot(df_mergedfb['timestamp'], df_mergedfb['max_amp_diff_norm'], alpha = 0.7, markersize = 1, label = 'RCS (beta-gamma)')
    plt.legend(ncol = 5, loc = 'upper right')
    plt.ylabel('scores (normalized)')
    plt.xlabel('time (samples)')

    out_dir = '/Users/mariaolaru/Documents/temp/RCS02/RCS02L_pkg_rcs/plots'
    plt.savefig(out_dir + '/' + 'psd_' + 'freq' + str(freq_band) + '.pdf')
    plt.close()
 
def plot_corrs(df_corr, PKG_measure):
    df_corr_s = df_corr.loc[PKG_measure, :]
    plt.rcParams["figure.figsize"] = (5,5)
    plt.plot(df_corr_s['freq_band'], df_corr_s['+2-0'], label = '+2-0', color = 'orchid')
    plt.plot(df_corr_s['freq_band'], df_corr_s['+3-1'], label = '+3-1', color = 'mediumpurple')
    plt.plot(df_corr_s['freq_band'], df_corr_s['+10-8'], label = '+10-8', color = 'darkkhaki')
    plt.plot(df_corr_s['freq_band'], df_corr_s['+11-9'], label = '+11-9', color = 'darkorange')
    plt.title(PKG_measure)
    plt.legend(ncol = 2, loc = 'upper right')
    plt.ylim(-1, 1)
    plt.ylabel('Pearson coef (r)')
    plt.xlabel('Frequency (Hz)')
    
    out_dir = '/Users/mariaolaru/Documents/temp/RCS02/RCS02L_pkg_rcs/plots'
    plt.savefig(out_dir + '/' + 'psd_corr_' + 'PKG_' + PKG_measure + '.pdf')
    plt.close()    
pkg_dir = '/Users/mariaolaru/Documents/temp/RCS02/RCS02L/RCS02L_ge_onmed/RCS02R_pkg_data/'
[df_pkg, start_time, stop_time] = preproc_pkg(pkg_dir)

#fp_phs = '/Users/mariaolaru/Documents/temp/RCS02/RCS02L/RCS02L_pre-stim/RCS02L_pre-stim_phs.csv'
#df_phs = preproc_phs(fp_phs, start_time, stop_time)

fp_psd = '/Users/mariaolaru/Documents/temp/RCS02/RCS02L/RCS02L_pre-stim/RCS02L_pre-stim_psd_partial_2m_wholemin.csv'
df_psd = preproc_psd(fp_psd, start_time, stop_time)
df_psd.columns = [''.join(str(col)) for col in df_psd.columns]
df_psd = df_psd.reset_index()
df_merged = process_dfs(df_pkg, df_psd)
contacts = ['+2-0', '+3-1', '+10-8', '+11-9']

df_merged['timestamp_dt'] = pd.to_datetime(df_merged['Date_Time'])
df_merged['timestamp_dt_h'] = [i.hour for i in df_merged['timestamp_dt']]
df_merged['asleep'] = (df_merged['timestamp_dt_h'] >= 20) | (df_merged['timestamp_dt_h'] <= 8)
df_merged['asleep'] = df_merged['asleep'].astype(int)

col = "(77.0, '+10-8')"
title = ("Feature: " + col)
plt.title(title)
plt.rcParams["figure.figsize"] = (30,3)
plt.plot(np.arange(1, len(df_merged)+1, 1), df_merged['DK_norm'], alpha = 0.7, label = 'PKG-DK', markersize = 1, color = 'steelblue')
plt.plot(np.arange(1, len(df_merged)+1, 1), df_merged[col], alpha = 0.7, label = col, markersize = 1, color = 'mediumpurple')

plt.vlines(np.where(df_merged['asleep'] == 1)[0], 0, 1, alpha = 0.1, label = 'asleep', color = 'grey')
plt.hlines(0.03, 0, len(df_merged)-1, color = 'red')
#plt.plot(np.arange(1, len(df_merged)+1, 1), df_merged['asleep'], alpha = 0.1, label = 'asleep', markersize = 1, color = 'grey')
plt.plot()
plt.legend(ncol = 5, loc = 'upper right')
plt.ylabel('scores (normalized)')
plt.xlabel('time (samples)')


"""
#Processing
freq_bands = np.array(df_psd.columns.levels[0])
df_corr = pd.DataFrame([])
for i in range(len(freq_bands)):
    freq_band = freq_bands[i]
    print("freq band: " + str(freq_band) + "hz")

    df_merged = process_dfs(df_pkg, df_psd, freq_band)
    plot_pkg_sync(df_merged, freq_band)
    
    
    #get corr coefs
    df_vars = df_merged.loc[:, ['BK_norm', 'DK_norm', (contacts[3]), (contacts[2]), (contacts[1]), (contacts[0])]]
    
    #df_vars = df_mergedfb.loc[:, ['BK_norm', 'DK_norm', 'max_amp_beta', 'max_amp_gamma']]
    df_corr_ind = df_vars.corr()
    
    out_fp = '/Users/mariaolaru/Documents/temp/RCS02/RCS02L_pkg_rcs' + '/RCS02_corrs_' + str(freq_band) + 'hz' + '.csv'
    
    df_corr_ind = df_corr_ind.round(3)
    df_corr_ind['freq_band'] = freq_band
    df_corr = pd.concat([df_corr, df_corr_ind])
df_corr.to_csv(out_fp)
print(df_corr)
print('/n')

#get correlation plots
plot_corrs(df_corr, 'DK_norm')
plot_corrs(df_corr, 'BK_norm')
"""

#Run an SVM
from sklearn.utils import Bunch
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn import metrics

# Look at weights of each feature on training data
# ^ Feature selection 
# Try multiple classes
# Use top 10 features, or try top 25
# Use only features w/ certain weight threshold
# Try an LDA analysis w/ multiple categories
# If power bands don't make sense as most predictive weights, may be bad approach


# Normalized DK scores > 0.2 == Dyskinesia

class_vals = np.arange(0.00, 0.2, 0.005)
df_stats = pd.DataFrame(columns = ['accuracy', 'precision', 'recall'])
for j in class_vals:
    print(str(j))
    val = j
    df_merged['DK_class'] = 0
    indxs = np.where(df_merged['DK_norm'] > val)[0]
    df_merged.loc[indxs, 'DK_class'] = 1
    
    cols = np.array(df_merged.columns)
    feature_col_indxs = [x for x, s in enumerate(list(cols)) if '+' in s]
    df_data = df_merged.iloc[:, feature_col_indxs]
    
    data = df_data.to_numpy()
    feature_names = df_data.columns
    target = df_merged['DK_class'].to_numpy()
    target_names = np.array(['present', 'absent', 'cutoff_value'])
    
    dyskinesia = Bunch(data = data, feature_names = feature_names, target = target, target_names = target_names)
    
    accuracy = np.array([])
    precision = np.array([])
    recall = np.array([])
    for i in range(20):
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
        
    data = [[accuracy.mean(), precision.mean(), recall.mean(), val]]
    df_ind = pd.DataFrame(data, columns =['accuracy', 'precision', 'recall', 'cutoff_value'])
    df_stats = pd.concat([df_stats, df_ind])
        
    print("Accuracy:", str(accuracy.mean()))
    print("Precision:", str(precision.mean()))
    print("Recall:", str(recall.mean()))


plt.rcParams["figure.figsize"] = (10, 10)
plt.plot(df_stats['recall'], df_stats['precision'])
plt.ylabel('Precision')
plt.xlabel('Recall')


plt.scatter(np.arange(0,len(y_pred)), y_pred, alpha = 0.7, label = 'predict')
plt.scatter(np.arange(0,len(y_pred)), y_test, alpha = 0.7, label = 'test')
plt.legend()

"""
#Smooth data
df_mergedfb['DK_smoo'] = df_mergedfb['DK_norm'].rolling(window = 2, win_type = 'gaussian', center = True).mean(std=0.5)
df_mergedfb['max_amp_gamma_smoo'] = df_mergedfb['max_amp_gamma_norm'].rolling(window = 8, win_type = 'gaussian', center = True).mean(std=0.5)

plt.plot(np.arange(1, len(df_mergedfb)+1, 1), df_mergedfb['max_amp_gamma_norm'], alpha = 0.7, label = 'RCS-gamma', markersize = 1, color = 'darkorange')
plt.plot(np.arange(1, len(df_mergedfb)+1, 1), df_mergedfb['max_amp_gamma_smoo'], alpha = 0.7, label = 'smoo', markersize = 1, color = 'red')
plt.legend(ncol = 4, loc = 'upper right')
plt.ylabel('scores (normalized)')
plt.xlabel('time (samples)')
"""