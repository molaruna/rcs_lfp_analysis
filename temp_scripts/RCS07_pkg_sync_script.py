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
df_merged_rmrest = df_merged[df_merged['inactive'] == 0]

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
df_coh_corr = sync.compute_correlation(df_merged, 'Cxy')
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

i_rm4 = [x for x, s in enumerate(list(df_merged_combo.columns)) if '+3-1_+' in s]
df_merged_spectra_2ch = df_merged_combo.drop(df_merged_combo.columns[i_rm4], axis = 1)

irm = i_rm = [x for x, s in enumerate(list(df_merged.columns)) if 'Cxy' in s]
df_merged_spectra = df_merged.drop(df_merged.columns[i_rm], axis = 1)

#run PCA analysis
df = df_merged_spectra_2ch
[df_pcs, pcs_vr] = sync.run_pca(df, 'spectra', 10, 0)

#re-add DK data into pc dataframe
df_svm = df_pcs.copy()
df_svm['DK'] = df.dropna().reset_index(drop=True)['DK']

"""
keys = ['+2-0', '+3-1', '+10-8', '+11-9']
for i in range(len(keys)):
    [pcs, test_vr] = sync.run_pca(df, keys[i], 5, 1)
    sync.plot_pcs(pcs.iloc[:, 0:pcs.shape[1]-1], keys[i], pkg_dir)

keys = ['+3-1', '+10-8']
df_pcs = sync.run_pca_wrapper(df, keys, 5, 0, pkg_dir)

#run SVM with PCA feature selection
sync.run_svm_wrapper(df_pcs, 'PC', 'DK', 0.03)
"""

#run LDA with PCA features
"""
#get top features
[df_top_pos, df_top_neg] = sync.get_top_features(coefs, x_names)
import pandas as pd
df_top_coefs = pd.concat([df_top_pos, df_top_neg])
"""
## split DK data in 5 equal dyskinesia classes above SVM threshold for dyskinesia
df_lda = df_svm.copy()
df_lda = df_lda[df_lda['DK'] > 0.03].reset_index(drop=True)
df_lda['DK_log'] = np.log10(df_lda['DK'])
df_lda['DK_log'] = df_lda['DK_log'] - df_lda['DK_log'].min()
class_thresh = np.nanpercentile(df_lda['DK_log'], [20, 40, 60, 80, 100])
labels = [1, 2, 3, 4, 5]
df_lda['DK_class'] = sync.add_classes(df_lda['DK_log'], class_thresh, labels)

indx = df_lda[df_lda['DK_class'] == 0].index
df_lda['DK_class'][indx] = 1

plt.plot(df_lda['DK_log'])
plt.hlines(class_thresh[0], 0, len(df_lda), alpha = 1, label = '20th percentile', color = 'red')
plt.hlines(class_thresh[1], 0, len(df_lda), alpha = 1, label = '40th percentile', color = 'red')
plt.hlines(class_thresh[2], 0, len(df_lda), alpha = 1, label = '60th percentile', color = 'red')
plt.hlines(class_thresh[3], 0, len(df_lda), alpha = 1, label = '80th percentile', color = 'red')
plt.hlines(class_thresh[4], 0, len(df_lda), alpha = 1, label = '100th percentile', color = 'red')
plt.ylabel('log(normalized DK scores)')
plt.xlabel('time (samples)')
plt.legend()

#indx = df_merged[df_merged['DK'] > 0.03].index
#df_merged.loc[indx, 'DK_class_binary'] = 1

"""
#create LDA df using feature selection
features = df_top_coefs['features']
df_temp = df_merged.copy()
df_temp = df_temp.dropna().reset_index(drop=True)
"""
X = df_lda.iloc[:, range(10)].to_numpy() #assumes 10 PCs
y = df_lda.loc[:, 'DK_class'].to_numpy()
#y_bi = df_temp.loc[:, 'DK_class_binary'].to_numpy()

import sklearn
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
out = sklearn.model_selection.cross_val_score(LinearDiscriminantAnalysis(), X, y, cv = 10)
np.mean(out)
import scipy.stats as stat
stat.sem(out)
#sync.run_LDA(df_merged) 


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



#fig.colorbar(img)
#temp = df_lm.sort_values('coefs').head(10)
#temp = df_lm.sort_values('coefs').tail(10)
#temp = temp.iloc[::-1]

#linear regression w/ top features
df_top = df_merged.loc[:, np.append(features, 'DK')]
sync.run_lm_wrapper(df_top, '+', 'DK', 2) #2 spectra, 2 coh channels

###Try PCA analysis
#from sklearn.decomposition import PCA
#from sklearn.model_selection import train_test_split
#import pandas as pd

# Make an instance of the Model
#pca = PCA(n_components=5) #minimum components to explain 95% of variance
#pca.fit(x_train)
#pcs = pca.fit_transform(x_train)
#pcs = pd.DataFrame(data = pcs)
#pca.n_components_

###Try NMF analysis
from sklearn.decomposition import NMF
nmf = NMF(n_components=5, init = 'random', random_state = 0, max_iter = 2000)
W = nmf.fit_transform(x_train)
H = nmf.components_

