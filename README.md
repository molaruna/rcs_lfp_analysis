# rcs_lfp_analysis

This code uses CSV file inputs from [processrcs_wrapper()](https://github.com/molaruna/processrcs_wrapper) to run time-series and spectral analyses from chronic human brain recordings collected using the Summit RC+S system. 

## Getting started

This code uses Python 3.8.3

## Data
Data are collected from the Summit RC+S neurostimulator (Medtronic) and available on [UCSF Box](https://ucsf.app.box.com/folder/0) and [Dropbox](https://www.dropbox.com/work) as Session directories. You can request access for these data from me. Then, the data are preprocessed and formatted into CSV files for python compatibility using [processrcs_wrapper()](https://github.com/molaruna/processrcs_wrapper). Specifically, these are CSV files that are required in each Session directory:<br/>
* eventLogTable.csv
* metaData.csv (header info: subj_ID, implant_side)
* timeDomainData.csv 
* stimLogSettings.csv
* timeDomainSettings.csv

Unless otherwise specified, CSV file header information matches the respective MAT file header information. 

## Analysis
All CSV inputs are first preprocessed using ```preproc.preprocess_settings(session_parent_dir)```.

Here's an example of the Session parent directory:
```
(base) ➜  RCS07L_montage git:(master) ✗ pwd
/Users/mariaolaru/Documents/RCS07/RCS07L/RCS07L_montage
(base) ➜  RCS07L_montage git:(master) ✗ ls
Session1570875824700
Session1570924382813
Session1570925029056
Session1570937044332
```
Now, in the Python console, run preproc.preprocess_settings():
```
import preproc.preprocess_funcs as preproc
session_parent_dir = '/Users/mariaolaru/Documents/RCS07/RCS07L/RCS07L_montage/'
[df_settings, df_notes, gp] = preproc.preprocess_settings(session_parent_dir)
df_ts = preproc.preprocess_data(session_parent_dir, msc, gp) 
```
### Montage spectral analysis
Montages refer to automated recordings which alternate between 11 sensing electrode pairs & 2 sampling rates (500Hz, 1000Hz) to better spatially localize the neural signal. Typically, montage recordings are collected in all four combinations of medication state (ON/OFF) and stimulation state (ON/OFF). Montages are plotted as power spectra using plot.plot_montage(session_parent_dir, labels):
```
from plts.plot_montage import plot_montage
labels = ["medOFF_stimON", "medON_stimON", "medOFF_stimOFF", "medON_stimOFF"]
plot_montage(session_parent_dir, labels)
```
Note: ```labels``` is a vector that lists the various conditions of each Session directory in *chronological* order. <br/>
<br/>
All outputs will be generated in a `plots` directory of the input directory

### Creating overlaid power spectra
```python3
from preproc.preprocess_funcs import preproc
from proc.process_funcs import proc

dir_name = '/Users/mariaolaru/Documents/temp/RCS02/RCS02L/RCS02L_pre-stim/'

[msc, df_notes, gp] = preproc.preprocess_settings(dir_name)
md = preproc.preprocess_data(dir_name, msc, gp) #separate fn b/c can take much longer time to process data
subj_id = msc['subj_id'][0] + " " + msc['implant_side'][0]

sr = 250
[msc_ds, md_ds] = proc.downsample_data(msc, md, sr) #downsamples separately for each msc label of sr

contacts = [msc['ch0_sense_contacts'].unique()[0], msc['ch1_sense_contacts'].unique()[0], msc['ch2_sense_contacts'].unique()[0], msc['ch3_sense_contacts'].unique()[0]]
df_psd = proc.convert_psd_long_old(md_ds, gp, contacts, 120, 119, sr)
```

## License
This software is open source and under an MIT license.



