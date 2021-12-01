#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Process the outputs of Matlab's ProcessRCS() function
Outputs: power spectra plots

@author: mariaolaru
"""
from preproc.preprocess_funcs import *
from plts.plot_funcs import *
from proc.process_funcs import *

dir_name = '/Users/mariaolaru/Documents/temp/RCS14L/RCS14L_post-stim/'
[msc, df_notes, gp] = preprocess_settings(dir_name)
md = preprocess_data(dir_name, msc, gp) #separate fn b/c can take much longer time to process data

#Find data with 1000Hz sr

def find_fs_ranges_temp(msc, md, fs):
    """
    Find timestamp ranges that contain the desired/input fs
    """
    md_ts_tail = md['timestamp'].tail(1).values[0]

    indx = np.where(msc['sr'] == fs)[0]
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

ts_ranges = find_fs_ranges_temp(msc, md, 1000)

mds = md[(md['timestamp'] > ts_ranges.loc[0, 'timestamp_start']) & (md['timestamp'] < ts_ranges.loc[0, 'timestamp_stop'])]
df_linked = link_data(msc, mds, gp)

def convert_psd_temp(df, sr):
    """
    Parameters
    ----------
    df : linked dataframe

    Returns
    -------
    df_psd : dataframe, power spectra using Welch's in long-form

    """
    cols = df['sense_contacts'].unique()
    
    df_psd = pd.DataFrame(columns = ['f_0'])   
    for i in range(len(cols)):
        dfs = df[(df['sense_contacts'] == cols[i])] 
        df_curr = convert_psd(dfs['voltage'], sr, cols[i])
        df_psd = df_psd.merge(df_curr, how = 'outer')
    
    return df_psd

df_psd = convert_psd_temp(df_linked, 1000) #does this work?

def plot_PSD(df_psd, msc, sr, gp):
    if sr == 1000:
        xlim_max = 500
    else:
        xlim_max = 100

    out_name = os.path.basename(gp)
    plot_title = make_montage_title(out_name, msc)

    fig, axs = plt.subplots(nrows = 1, ncols = 4, figsize=(15, 5))
    fig.suptitle(plot_title)

    for ax in fig.get_axes():
        ax.label_outer()
        ax.set_ylim([-10, -2])
        ax.set_xlim([0, xlim_max])

    out_plot_fp = get_name(gp, out_name)
    print("Plotting: \n" + out_name + "\n")

    sense_contacts = df_linked['sense_contacts'].unique()

    for i in range(len(sense_contacts)):
        axs[i].set(xlabel = 'frequency (Hz)')

        if i == 0:
            axs[i].set(ylabel = 'log10(Power)')

        axs[i].set_title('sense contacts: ' + sense_contacts[i])
        axs[i].axvspan(4, 8, color = 'royalblue', alpha = 0.1)
        axs[i].axvspan(13, 30, color = 'indianred', alpha = 0.1)
        axs[i].axvspan(60, 90, color = 'olivedrab', alpha = 0.1)
        axs[i].plot(df_psd['f_0'], np.log10(df_psd[[sense_contacts[i]]]))
            
    fig.tight_layout()
    fig.savefig(out_plot_fp + "_PSD_pre-stim_" + str(sr) + "hz" + ".svg")
    fig.savefig(out_plot_fp + "_PSD_pre-stim_" + str(sr) + "hz" + ".pdf")

