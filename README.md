# rcs_lfp_analysis

This code uses CSV file inputs from [processrcs_wrapper()](https://github.com/molaruna/processrcs_wrapper) to run time-series and spectral analyses from chronic human brain recordings collected using the Summit RC+S system. 

## Getting started

This code uses Python 3.8.3

## Data
Data are collected from the Summit RC+S neurostimulator (Medtronic) and available on [UCSF Box](https://ucsf.app.box.com/folder/0) and [Dropbox](https://www.dropbox.com/work) as Session directories, you can request access from me. Then, the data is preprocessed and formatted into CSV files for python compatibility using [processrcs_wrapper()](https://github.com/molaruna/processrcs_wrapper). Specifically, these CSV files are required in each Session directory:<br/>
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
(base) ➜  RCS07L_pre-stim git:(master) ✗ pwd
/Users/mariaolaru/Documents/RCS07/RCS07L/RCS07L_pre-stim
(base) ➜  RCS07L_pre-stim git:(master) ✗ ls
Session1570875824700
Session1570924382813
Session1570925029056
Session1570937044332
Session1570955823964
```
Now, in the Python console, input the parent directory of the Sessions folders:
```
import preproc.preprocess_funcs as preproc
session_parent_dir = '/Users/mariaolaru/Documents/RCS07/RCS07L/RCS07L_pre-stim/'
[df_settings, df_notes, gp] = preproc.preprocess_settings(session_parent_dir)
df_ts = preproc.preprocess_data(session_parent_dir, msc, gp) 
```
### Montage plots



