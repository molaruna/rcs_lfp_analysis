# RCS14_entrainment_naive.py
# Generate timeseries analysis and power estimate
# Author: maria.olaru@

import os
from matplotlib import pyplot as plt
import numpy as np
import scipy.signal as signal
import pandas as pd

def get_name(gp, out_name_full):
    out_plot_dir = gp + '/' + 'plots/' 
    
    if not os.path.isdir(out_plot_dir):
        os.mkdir(out_plot_dir)
       
    out_plot_fp = out_plot_dir + out_name_full
    if os.path.isfile(out_plot_fp + '.svg'):
        count = 2
        out_plot_fp_count = out_plot_fp + '_v' + str(count)
        while os.path.isfile(out_plot_fp_count + '.svg'):
            count = count + 1
            out_plot_fp_count = out_plot_fp + '_v' + str(count)        
        out_plot_fp = out_plot_fp_count  

    return out_plot_fp

def make_montage_title(out_name, msc):
    #subj_id = mscs['subj_id'].iloc[0]
    mscs = msc.reset_index().iloc[0]
    stim_amp = str(mscs['stim_amp'])
    stim_contact = str(int(mscs['stim_contact']))
    stim_freq = str(mscs['stim_freq'])
    
    plot_title = out_name + '\nstim contact: ' + stim_contact + '; stim amp: ' + stim_amp + '; stim freq: ' + stim_freq
    return plot_title


def make_plot_title(out_name, step_size, mscs, tt):
    #subj_id = mscs['subj_id'].iloc[0]
    mscs = mscs.reset_index().iloc[0]
    stim_state = str(int(mscs['stim_status']))
    stim_amp = str(mscs['amplitude_ma'])
    stim_contact = str(int(mscs['stim_contact_cath']))
    stim_freq = str(int(mscs['stimfrequency_hz']))
    step_size = str(step_size)
    start_time = str(mscs['timestamp'])
    tt = str(round(tt, 2))
    
    plot_title = out_name + '\nstim state: ' +  stim_state + '; stim contact: ' + stim_contact + '; stim amp: ' + stim_amp + '; stim freq: ' + stim_freq + '\n' + start_time + '; time/PSD = ' + step_size + 's' + '; total time: ' + tt + 'm'
    return plot_title

def plot_PSD_wide(df_psds, gp):
    stim_freqs = np.array([*df_psds.keys()])
    stim_amps = np.array([*df_psds[stim_freqs[0]].keys()])
    num_sfs = len(stim_freqs)
    num_sense_contacts = len(df_psds[stim_freqs[0]][stim_amps[0]].columns)-1
    sense_contacts = df_psds[stim_freqs[0]][stim_amps[0]].columns[1:num_sense_contacts+1]
    out_name = os.path.basename(gp)
    plot_title = out_name + "test_RCS02_entrainment"

    fig, axs = plt.subplots(num_sfs, num_sense_contacts, figsize=(num_sense_contacts*4, num_sfs*4))
    fig.suptitle(plot_title, x=0.5, y=.999, verticalalignment='top')

    r = 0
    c = 0

    for ax in fig.get_axes():
        ax.label_outer()
        ax.set_ylim([-10, -2])
        ax.set_xlim([0, 100])

    out_plot_fp = get_name(gp, out_name)
    print("Plotting: \n" + out_name + "\n")

    for stim_freq in stim_freqs:
        df_sf = df_psds[stim_freq]
        stim_amps = np.array([*df_sf.keys()])
        #num_sas = len(stim_amps)
        c = 0
        if (c == 0):
            axs[r, c].set(ylabel = 'log10(Power)')

        for i in range(len(sense_contacts)):
            #if r == 0:
            axs[r, c].set_title('contacts: ' + sense_contacts[i] + ' | stim_freq: ' + str(stim_freq) + 'hz')
            axs[r, c].axvspan(4, 8, color = 'royalblue', alpha = 0.1)
            axs[r, c].axvspan(13, 30, color = 'indianred', alpha = 0.1)
            axs[r, c].axvspan(60, 90, color = 'olivedrab', alpha = 0.1)   
            axs[r, c].axvline(stim_freq/2, alpha = 0.2)

            for stim_amp in stim_amps:
                df_psd_sense = df_sf[stim_amp]
                spectra = df_psd_sense[sense_contacts[i]]
                axs[r, c].plot(df_psd_sense['f_0'], np.log10(spectra), alpha = 0.8, label = stim_amp)
            axs[r, c].legend(loc='upper right', ncol=3, title = "stim amp")
            c = c+1
        r = r+1
    fig.tight_layout(w_pad = 1, h_pad = 3)
    
    fig.savefig(out_plot_fp + "_psds" + ".svg")
    print(out_plot_fp + "_psds" + ".pdf")
    fig.savefig(out_plot_fp + "_psds"  + ".pdf")

"""
def plot_PSD(mscs, df_psd, fs, out_name, plot_title, gp, alpha = 0.1):
    mscs = mscs.reset_index().iloc[0]
    sense_contacts = [mscs['ch1_sense_contact_an'], mscs['ch1_sense_contact_cath'], mscs['ch2_sense_contact_an'], mscs['ch2_sense_contact_cath'], mscs['ch3_sense_contact_an'], mscs['ch3_sense_contact_cath'], mscs['ch4_sense_contact_an'], mscs['ch4_sense_contact_cath']]    

    out_plot_fp = get_name(gp, out_name)
    print("Plotting: \n" + out_name + "\n")
    
    fig, axs = plt.subplots(len(df_psd['channel'].unique()), figsize=(6.4*2, 4.8*2))
    fig.suptitle(plot_title)

    for i in range(len(df_psd['channel'].unique())):
        ax_title = 'ch' + str(i+1) + ': contacts ' + str(int(sense_contacts[i*2])) + '-' + str(int(sense_contacts[i*2+1]))
        axs[i].set_title(ax_title)
        
        for ax in fig.get_axes():
            ax.label_outer()
            
        axs[i].set(xlabel = 'frequency (Hz)', ylabel = 'mV**2/Hz')
        axs[i].axvspan(13, 30, color = 'indianred', alpha = 0.1, label = 'beta band')
        axs[i].axvspan(60, 90, color = 'olivedrab', alpha = 0.1, label = 'gamma band')
        #axs[i].set_xlim([0, 100])
        #axs[i].legend()
        
        for j in range(len(df_psd['step'].unique())):
            df_psds = df_psd.loc[df_psd['channel'] == df_psd['channel'].unique()[i]]
            df_psdss = df_psds.loc[df_psds['step'] == df_psds['step'].unique()[j]]
            #df_psdss = df_psdss[df_psdss['f_0'] <= 100]
            axs[i].semilogy(df_psdss['f_0'], df_psdss['Pxx_den'], color = 'blue', alpha = alpha)

    fig.savefig(out_plot_fp + ".svg")
"""

def plot_PSD_montage_channels(psds, msc, sr, gp):
    if sr == 1000:
        xlim_max = 500
    else:
        xlim_max = 100
    psds_keys = np.array([*psds.keys()])
    
    out_name = os.path.basename(gp)
    plot_title = make_montage_title(out_name, msc)

    fig, axs = plt.subplots(2, 6, figsize=(22, 7))
    fig.suptitle(plot_title)

    for ax in fig.get_axes():
        ax.label_outer()
        ax.set_ylim([-10, -2])
        ax.set_xlim([0, xlim_max])

    out_plot_fp = get_name(gp, out_name)
    print("Plotting: \n" + out_name + "\n")

    num_montages = len(psds[psds_keys[0]].columns)-1
    r = 0
    c = 0
    for i in range(num_montages):
        if (i == 6):
            c = 0
            r = 1
        sense_contacts = psds[psds_keys[0]].columns[i+1]
        axs[r, c].set_title('contacts: ' + sense_contacts)
        
        if (r == 1):    
            axs[r, c].set(xlabel = 'frequency (Hz)')
        if (c == 0):
            axs[r, c].set(ylabel = 'log10(Power)')
            
        axs[r, c].axvspan(4, 8, color = 'royalblue', alpha = 0.1)
        axs[r, c].axvspan(13, 30, color = 'indianred', alpha = 0.1)
        axs[r, c].axvspan(60, 90, color = 'olivedrab', alpha = 0.1)
#        df_psdss = df_psdss[df_psdss['f_0'] <= 100]
        for j in range(len(psds_keys)):
            condition = psds_keys[j]
            axs[r, c].plot(psds[condition]['f_0'], np.log10(psds[psds_keys[j]][[sense_contacts]]), label = condition)
            axs[r, c].legend()
        c = c + 1
            
    fig.tight_layout(w_pad=5)
    fig.savefig(out_plot_fp + "_contacts_sr" + str(sr) + ".svg")
    print(out_plot_fp + "_contacts_sr" + str(sr) + ".pdf")
    fig.savefig(out_plot_fp + "_contacts_sr"  + str(sr) + ".pdf")
    
def plot_PSD_montage_conditions(psds, msc, sr, gp):
    if sr == 1000:
        xlim_max = 500
    else:
        xlim_max = 100
    psds_keys = np.array([*psds.keys()])
    
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

    num_conditions = len(psds_keys)
    num_montages = len(psds[psds_keys[0]].columns)-1

    for i in range(num_conditions):
        condition = psds_keys[i]
        axs[i].set_title('condition: ' + condition)
        axs[i].set(xlabel = 'frequency (Hz)')

        if i == 0:
            axs[i].set(ylabel = 'log10(Power)')
            
        axs[i].axvspan(4, 8, color = 'royalblue', alpha = 0.1)
        axs[i].axvspan(13, 30, color = 'indianred', alpha = 0.1)
        axs[i].axvspan(60, 90, color = 'olivedrab', alpha = 0.1)

        for j in range(num_montages):
            sense_contacts = psds[psds_keys[0]].columns[j+1]
            axs[i].plot(psds[condition]['f_0'], np.log10(psds[psds_keys[i]][[sense_contacts]]), label = sense_contacts)
        if i == num_conditions-1:
            axs[i].legend(loc='lower right',
          ncol=3)
            
    fig.tight_layout()
    fig.savefig(out_plot_fp + "_conditions_" + str(sr) + ".svg")
    fig.savefig(out_plot_fp + "_conditions_" + str(sr) + ".pdf")


"""
def plot_spectrogram(md, msc, gp, i_int, padding, step_size, out_name):
    
    subj_id = msc['subj_id'].loc[i_int]    

    fs = msc['ch1_sr'].iloc[i_int]
    
    sense_contacts = [msc['ch1_sense_contact_an'].iloc[i_int], msc['ch1_sense_contact_cath'].iloc[i_int], msc['ch2_sense_contact_an'].iloc[i_int], msc['ch2_sense_contact_cath'].iloc[i_int], msc['ch3_sense_contact_an'].iloc[i_int], msc['ch3_sense_contact_cath'].iloc[i_int], msc['ch4_sense_contact_an'].iloc[i_int], msc['ch4_sense_contact_cath'].iloc[i_int]]    
    stim_freq = msc['stimfrequency_hz'].iloc[i_int]
    
    ss = msc.iloc[[i_int-1,i_int], :].reset_index()
    plot_title = get_plot_title(out_name, ss, padding, step_size)

    mds = subset_md(md, msc, i_int, padding)
    dfp = melt_mds(mds, step_size, fs)
    ch = 4
    [f, t, Sxx] = dfp_spect(dfp, fs, ch, padding)


    fig, axs = plt.subplots(2, 1)
    fig.suptitle(plot_title + "testing")
    
    i = ch-1
    ax_title = 'ch' + str(i+1) + ': contacts ' + str(int(sense_contacts[i*2])) + '-' + str(int(sense_contacts[i*2+1]))

    axs[1].set_title(ax_title)      
    axs[1].set(xlabel = 'Time (seconds)', ylabel = 'Frequency (Hz)')

    axs[1].axhline(stim_freq/2, 0, 1, c = 'indianred', alpha = 0.4, label = '1/2 stim freq')
    axs[1].set_ylim([stim_freq/2 - 10, stim_freq/2 + 10])
    
    im = axs[1].pcolormesh(t, f, np.log10(Sxx)) #frequencies are off b/c scaling
    fig.colorbar(im, ax=axs[1])
    im

    axs[1].legend()

    out_name_full = subj_id + "_" + out_name
    out_plot_fp = get_name(gp, out_name_full)
                 
    fig.savefig(out_plot_fp + ".svg")
    print("Plotting: \n" + out_name_full + "\n")
"""

def butter_lowpass_filtfilt(data, cutoff, fs, order=5):
    b, a = butter_lowpass(cutoff, fs, order=order)
    y = signal.filtfilt(b, a, data)
    return y    
    
def butter_lowpass(cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = signal.butter(order, normal_cutoff, btype = 'low', analog = False)
    
    return b, a

def diff(Pxx_den, f_0):
    d = np.diff(Pxx_den)
    freq_diff = f_0[1]-f_0[0]
    d_f0=f_0[0:len(f_0)-1] + freq_diff/2
    df_smooth_diff = pd.DataFrame({'f_0_diff':d_f0, 'Pxx_den_diff':d})
 
    return df_smooth_diff

def plot_phs(df_phs, gp, subj_id):

    out_name = os.path.basename(gp)

    channels = df_phs['contacts'].unique()
    plot_title = subj_id + ' pre-stim oscillation peak heights'
    fig, axs = plt.subplots(nrows = 1, ncols = len(channels), figsize=(len(channels)*4, 5))
    fig.suptitle(plot_title)

    for ax in fig.get_axes():
        ax.label_outer()
        ax.set_ylim([0, 2.75])
        ax.set_xlim([0, 100])

    out_plot_fp = get_name(gp, out_name)
    print("Plotting: \n" + out_name + "\n")

    for i in range(len(channels)):

        curr_ch = df_phs['contacts'].unique()[i]
        df_ch = df_phs[df_phs['contacts'] == curr_ch]

        axs[i].set_title('contacts: ' + curr_ch)
        axs[i].set(xlabel = 'frequency (Hz)')

        if i == 0:
            axs[i].set(ylabel = 'log10(Power)')
            
        axs[i].axvspan(4, 8, color = 'royalblue', alpha = 0.1)
        axs[i].axvspan(8, 12.9, color = 'orange', alpha = 0.1)
        axs[i].axvspan(13, 30, color = 'indianred', alpha = 0.1)
        axs[i].axvspan(30, 59.9, color = 'plum', alpha = 0.1)
        axs[i].axvspan(60, 90, color = 'olivedrab', alpha = 0.1)
        #ax.axhline(y=0.65, color='r')
        axs[i].scatter(df_ch['freq'], df_ch['max_amp'], alpha = 0.2)       

        fig.tight_layout()


    out_name = os.path.basename(gp)
    out_plot_fp = get_name(gp, out_name)

    plt.savefig(out_plot_fp + "_native_osc" + ".svg")
    plt.savefig(out_plot_fp + "_native_osc" + ".pdf")

def plot_entrainment(df_entraint, gp):
    x_labels = np.sort(df_entraint['stim_freq'].unique())
    y_labels = np.sort(df_entraint['stim_amp'].unique())
    
    x_str = ["%.1f" % x for x in x_labels]
    y_str = ["%.1f" % y for y in y_labels]
    print(y_str)
    
    df_mat = pd.DataFrame(data = -1, columns = x_str, index = range(len(y_labels)))
    
    for i in range(len(x_labels)):
        stim_freq = x_labels[i]
        for j in range(len(y_labels)):
            stim_amp = y_labels[j]
            max_amp = df_entraint[(df_entraint['stim_freq'] == stim_freq) & (df_entraint['stim_amp'] == stim_amp)]['max_amp']
            if (max_amp.empty):
                max_amp = -1
            df_mat.iloc[j, i] = max_amp
    
    fig, ax = plt.subplots(1,1)
    img = ax.imshow(df_mat, extent = [0,len(x_str)-1,0,len(y_str)-1], cmap='copper', interpolation='nearest', origin = 'lower')
    
    ax.set_xticks(np.arange(0, len(x_str),1))
    ax.set_yticks(np.arange(0, len(y_str),1))
    ax.set_xticklabels(x_str, rotation = -45)
    ax.set_yticklabels(y_str)
    ax.set_title('Entrainment scores')
    ax.set(xlabel = 'stimulation frequency (Hz)')
    ax.set(ylabel = 'stimulation amplitude (mA)')
    
    fig.colorbar(img)

    out_name = os.path.basename(gp)
    out_plot_fp = get_name(gp, out_name)

    plt.savefig(out_plot_fp + "_entrainment" + ".svg", bbox_inches="tight")
    plt.savefig(out_plot_fp + "_entrainment" + ".pdf", bbox_inches="tight")

    
