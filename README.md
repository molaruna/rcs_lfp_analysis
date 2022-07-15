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

## Preprocessing workflow
All CSV inputs are first preprocessed using ```preproc.preprocess_settings(session_parent_dir)```.

Here's an example of the Session parent directory:
```
(base) ➜  RCS02L_montage git:(master) ✗ pwd
~/Documents/RCS02/RCS02L/RCS02L_montage
(base) ➜  RCS02L_montage git:(master) ✗ ls
Session1570875824700
Session1570924382813
Session1570925029056
Session1570937044332
```
Now here's how you can process and format the Sessions of the Session parent directory:
```python3
import preproc.preprocess_funcs as preproc
session_parent_path = '~/Documents/RCS02/RCS02L/RCS02L_montage/'
[settings, notes, grandparent_path] = preproc.preprocess_settings(session_parent_path)
neural_data = preproc.preprocess_data(session_parent_path, settings, grandparent_path) 
```
### Montage spectral analysis
Montages refer to automated recordings which alternate between 11 sensing electrode pairs & 2 sampling rates (500Hz, 1000Hz) to better spatially localize the neural signal. Typically, montage recordings are collected in all four combinations of medication state (ON/OFF) and stimulation state (ON/OFF). Montages are plotted as power spectra using plot.plot_montage(session_parent_dir, labels):
```python3
from plts.plot_montage import plot_montage
labels = ["medOFF_stimON", "medON_stimON", "medOFF_stimOFF", "medON_stimOFF"]
plot_montage(session_parent_path, labels)
```
Note: ```labels``` is a vector that lists the various conditions of each Session directory in *chronological* order. <br/>
<br/>
All outputs will be generated in a `plots` directory of the input directory

### Creating overlaid power spectra
```python3

subj_id = settings['subj_id'][0] + " " + settings['implant_side'][0]

contacts = [settings['ch0_sense_contacts'].unique()[0], settings['ch1_sense_contacts'].unique()[0], settings['ch2_sense_contacts'].unique()[0], settings['ch3_sense_contacts'].unique()[0]]

sr = 250 #this information can be found in the settings table, but good to sanity check manually first
overlaid_psds = proc.convert_psd_long_old(neural_data, grandparent_path, contacts, 120, 119, sr) #create 2 minute intervals with 1 minute overlap
```

## License
This software is open source and under an MIT license.



