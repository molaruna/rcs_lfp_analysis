#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 21 11:38:47 2021

@author: mariaolaru

Individual freq correlations
"""

import proc.rcs_pkg_sync_funcs as sync
import numpy as np
from matplotlib import pyplot as plt
    
pkg_dir = '/Users/mariaolaru/Documents/temp/RCS07/RCS07L/RCS07L_pkg_data/'
fp_phs = '/Users/mariaolaru/Documents/temp/RCS07/RCS07L/RCS07L_pre-stim/RCS07L_pre-stim_phs.csv'
fp_psd = '/Users/mariaolaru/Documents/temp/RCS07/RCS07L/RCS07L_pre-stim/RCS07L_pre-stim_psd.csv'
#fp_psd = '/Users/mariaolaru/Documents/temp/RCS07/RCS07L/RCS07L_pre-stim/RCS07L_pre-stim_psd_aperiodic.csv'
#fp_psd = '/Users/mariaolaru/Documents/temp/RCS07/RCS07L/RCS07L_pre-stim/RCS07L_pre-stim_psd_periodic.csv'
fp_coh = '/Users/mariaolaru/Documents/temp/RCS07/RCS07L/RCS07L_pre-stim/RCS07L_pre-stim_coh.csv'
fp_notes = '/Users/mariaolaru/Documents/temp/RCS07/RCS07L/RCS07L_pre-stim/RCS07L_pre-stim_meta_session_notes.csv'
sr = 250

[df_pkg, start_time, stop_time] = sync.preproc_pkg(pkg_dir)
df_phs = sync.preproc_phs(fp_phs, start_time, stop_time)

df_psd = sync.preproc_psd(fp_psd, start_time, stop_time)

df_coh = sync.preproc_coh(fp_coh, start_time, stop_time, sr)
df_notes = sync.preproc_notes(fp_notes, start_time, stop_time)
df_dys = sync.find_dyskinesia(df_notes)
df_meds = sync.get_med_times()

#Processing
df_merged = sync.process_dfs(df_pkg, df_phs, df_psd, df_coh, df_meds, df_dys)
df_merged = sync.add_sleep_col(df_merged)


#remove BK scores reflecting periods of inactivity
df_merged_corr = df_merged[df_merged['inactive'] == 0]

#correlate all scores
keyword = 'spectra'
#keyword = 'fooof_flat'
#keyword = 'fooof_peak_rm'
df_spectra_corr = sync.compute_correlation(df_merged, keyword)
out_fp = '/Users/mariaolaru/Documents/temp/RCS07/RCS07L/RCS07L_pkg_rcs' + '/RCS07_corrs' + '.csv'
df_spectra_corr.to_csv(out_fp)

#plot correlations for each frequency
sync.plot_corrs(df_spectra_corr, 'DK')
sync.plot_corrs(df_spectra_corr, 'BK')

#correlate coherence
df_coh_corr = sync.compute_correlation(df_merged_corr, 'Cxy')
sync.plot_corrs(df_coh_corr, 'DK')
sync.plot_corrs(df_coh_corr, 'BK')

####### Plotting timeseries data ####################
df = df_merged
freq = 13
plt.close()
contacts = np.array(['+2-0', '+3-1', '+10-8', '+11-9'])
breaks = sync.find_noncontinuous_seg(df_merged['timestamp'])
title = "RCS07 PKG-RCS pre-stim time-series sync"
#title = ("freq_band: " + str(freq_band) + "Hz")
plt.title(title)
plt.rcParams["figure.figsize"] = (30,3.5)

#plt.plot(np.arange(1, len(df)+1, 1), df['phs_gamma'], alpha = 0.7, label = 'phs-gamma', markersize = 1, color = 'slategray')
#plt.plot(np.arange(1, len(df)+1, 1), df['phs_beta'], alpha = 0.7, label = 'phs-beta', markersize = 1, color = 'olivedrab')

plt.plot(np.arange(1, len(df)+1, 1), df['DK'], alpha = 0.9, label = 'PKG-DK', markersize = 1, color = 'steelblue')
#plt.plot(np.arange(1, len(df)+1, 1), df['BK'], alpha = 0.7, label = 'PKG-BK', markersize = 1, color = 'indianred')

#plt.plot(np.arange(1, len(df)+1, 1), df["('" + keyword + "'," + str(freq) + ".0,'" + contacts[0] + "')"], alpha = 0.7, label = str(freq)+ "Hz " + contacts[0], markersize = 1, color = 'orchid')
#plt.plot(np.arange(1, len(df)+1, 1), df["('" + keyword + "'," + str(freq) + ".0,'" + contacts[1] + "')"], alpha = 0.9, label = str(freq)+ "Hz " + contacts[1], markersize = 1, color = 'mediumpurple')

#freq = 13
#plt.plot(np.arange(1, len(df)+1, 1), df["('" + keyword + "'," + str(freq) + ".0,'" + contacts[2] + "')"], alpha = 0.7, label = str(freq)+ "Hz " + contacts[2], markersize = 1, color = 'darkkhaki')
#plt.plot(np.arange(1, len(df)+1, 1), df["('" + keyword + "'," + str(freq) + ".0,'" + contacts[3] + "')"], alpha = 0.7, label = str(freq)+ "Hz " + contacts[3], markersize = 1, color = 'darkorange')

#plt.vlines(df_merged[df['dyskinesia'] == 1].index, 0, 1, color = 'black', label = 'dyskinesia')
#plt.vlines(df_merged[df['med_time'] == 1].index, 0, 1, color = 'green', label = 'meds taken')
#plt.vlines(np.where(df['asleep'] == 1)[0], 0, 1, alpha = 0.1, label = 'asleep', color = 'grey')
#plt.vlines(breaks, 0, 1, alpha = 0.7, label = 'break', color = 'red')

#plt.hlines(class_thresh[0], 0, len(df), alpha = 0.7, label = 'LDA thresh', color = 'red')
#plt.hlines(class_thresh[1], 0, len(df), alpha = 0.7, label = 'LDA thresh', color = 'red')
#plt.hlines(class_thresh[2], 0, len(df), alpha = 0.7, label = 'LDA thresh', color = 'red')
#plt.hlines(class_thresh[3], len(df), alpha = 0.7, label = 'LDA thresh', color = 'red')
#plt.hlines(class_thresh[4], 0, len(df), alpha = 0.7, label = 'LDA thresh', color = 'red')


plt.legend(ncol = 6, loc = 'upper right')
plt.ylabel('scores (normalized)')
plt.xlabel('time (samples)')
#####################################################

#create dataset with combo channels of spectra and coherence features
i_rm = [x for x, s in enumerate(list(df_merged.columns)) if '+2-0_+' in s]
i_rm2 = [x for x, s in enumerate(list(df_merged.columns)) if "'+2-0')" in s]
i_rm3 = [x for x, s in enumerate(list(df_merged.columns)) if "'+11-9')" in s]
i_rmt = np.concatenate([i_rm, i_rm2, i_rm3])
df_merged_combo = df_merged.drop(df_merged.columns[i_rmt], axis = 1)

irm = i_rm = [x for x, s in enumerate(list(df_merged.columns)) if 'Cxy' in s]
df_merged_spectra = df_merged.drop(df_merged.columns[i_rm], axis = 1)

#run SVM
sync.run_svm_wrapper(df_merged, 'spectra', 'DK', 0.03)
sync.run_svm_wrapper(df_merged_spectra, '+2-0', 'DK', 0.03)
sync.run_svm_wrapper(df_merged_spectra, '+3-1', 'DK', 0.03)
sync.run_svm_wrapper(df_merged_spectra, '+10-8', 'DK', 0.03)
sync.run_svm_wrapper(df_merged_spectra, '+11-9', 'DK', 0.03)
sync.run_svm_wrapper(df_merged, 'Cxy', 'DK', 0.03)
sync.run_svm_wrapper(df_merged_combo, '+', 'DK', 0.03)
[coefs, x_names] = sync.run_svm_wrapper(df_merged_combo, 'spectra', 'DK', 0.03)
sync.run_svm_wrapper(df_merged_combo, 'Cxy', 'DK', 0.03)
sync.run_svm_wrapper(df_merged_combo, '+3-1_+10-8', 'DK', 0.03)

#run linear regressions
sync.run_lm_wrapper(df_merged, 'spectra', 'DK', 2) #all spectra channels
sync.run_lm_wrapper(df_merged, 'Cxy', 'DK', 2) #all coherence combos
sync.run_lm_wrapper(df_merged_combo, '+', 'DK', 2) #2 spectra, 2 coh channels
sync.run_lm_wrapper(df_merged_combo, 'spectra', 'DK', 2) #2 spectra channels
sync.run_lm_wrapper(df_merged_combo, 'Cxy', 'DK', 2) #2 coh channels
sync.run_lm_wrapper(df_merged_combo, "'+3-1'", 'DK', 2) #1 spectra channels


#get top features

[df_top_pos, df_top_neg] = sync.get_top_features(coefs, x_names)
import pandas as pd
df_top_coefs = pd.concat([df_top_pos, df_top_neg])

## split DK data in 5 dyskinesia classes
class_thresh = np.nanpercentile(df_merged['DK'][df_merged['DK'] > 0], [20, 40, 60, 80, 100])
labels = [1, 2, 3, 4, 5]
df_merged['DK_class'] = sync.add_classes(df_merged['DK'], class_thresh, labels)
df_merged['DK_class_binary'] = 0

#indx = df_merged[df_merged['DK'] > 0.03].index
#df_merged.loc[indx, 'DK_class_binary'] = 1

#create LDA df using feature selection
features = df_top_coefs['features']
df_temp = df_merged.copy()
df_temp = df_temp.dropna().reset_index(drop=True)
X = df_temp.loc[:, features].to_numpy()
y_orig = df_temp.loc[:, 'DK'].values.reshape(-1, 1)
y = df_temp.loc[:, 'DK_class'].to_numpy()
#y_bi = df_temp.loc[:, 'DK_class_binary'].to_numpy()

import sklearn
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
out = sklearn.model_selection.cross_val_score(LinearDiscriminantAnalysis(), X, y, cv = 10)
np.mean(out)
import scipy.stats as stat
stat.mer(out)
#sync.run_LDA(df_merged) 

#fig.colorbar(img)
#temp = df_lm.sort_values('coefs').head(10)
#temp = df_lm.sort_values('coefs').tail(10)
#temp = temp.iloc[::-1]

#linear regression w/ top features
df_top = df_merged.loc[:, np.append(features, 'DK')]
sync.run_lm_wrapper(df_top, '+', 'DK', 2) #2 spectra, 2 coh channels

###Try PCA analysis
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
import pandas as pd

# Make an instance of the Model
pca = PCA(n_components=5) #minimum components to explain 95% of variance
pca.fit(x_train)
pcs = pca.fit_transform(x_train)
pcs = pd.DataFrame(data = pcs)
pca.n_components_

###Try NMF analysis
from sklearn.decomposition import NMF
nmf = NMF(n_components=5, init = 'random', random_state = 0, max_iter = 2000)
W = nmf.fit_transform(x_train)
H = nmf.components_

###Try PCA analysis Tanner's way
from sklearn.decomposition import PCA

import pandas as pd
df_temp = df_merged.copy()
df_temp = df_temp.dropna().reset_index(drop=True)

#get all features
feature_col_indxs = [x for x, s in enumerate(list(df_temp.columns)) if '+' in s]
addl_col = np.where(df_temp.columns == "phs_gamma")[0]
X = df_temp.iloc[:, np.append(feature_col_indxs, addl_col)].to_numpy()
X_colnames = df_temp.columns[np.append(feature_col_indxs, addl_col)]

df_lm = pd.DataFrame()
df_lm['features'] = X_colnames

feature_col_indxs = [x for x, s in enumerate(list(df_temp.columns)) if '+' in s]
addl_col = np.where(df_temp.columns == "phs_gamma")[0]

y_orig = df_temp.loc[:, 'DK'].values.reshape(-1, 1)
x_orig = df_temp.iloc[:, np.append(feature_col_indxs, addl_col)]

# Make an instance of the Model
contacts = np.array(['+2-0', '+3-1', '+10-8', '+11-9'])
i=3
contact = contacts[i]
indices = [j for j, s in enumerate(list(df_lm['features'])) if contacts[i] in s]    

x = x_orig.iloc[:, indices].to_numpy()
x_train, x_test, y_train, y_test = train_test_split(x, y_orig, test_size = 0.2, random_state = 42)

pca = PCA(n_components=5) #minimum components to explain 95% of variance
pca.fit(x_train.T)
pcs = pca.fit_transform(x_train.T)
pcs = pd.DataFrame(data = pcs)
pca.n_components_

plt.plot(np.linspace(0, 125, 126), pcs[0], label = 'PC1')
plt.plot(np.linspace(0, 125, 126), pcs[1], label = 'PC2')
plt.plot(np.linspace(0, 125, 126), pcs[2], label = 'PC3')
plt.plot(np.linspace(0, 125, 126), pcs[3], label = 'PC4')
plt.plot(np.linspace(0, 125, 126), pcs[4], label = 'PC5')
plt.xlabel('Frequency (Hz)')
plt.ylabel('PC importance')
plt.title(contacts[i])
plt.legend()
