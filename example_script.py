#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Preprocesses, analyzes, and plots an entire directory of Session folders

Requirements – CSV files that have been restructed via ProcessRCS() JSON->MAT->CSV
these files can be created in bulk using processrcs_wrapper()

example
ls -d1 $PWD/Session*/DeviceNPC**/ > /Users/mariaolaru/my_list.txt
python main.py main /Users/mariaolaru/my_list.txt

@author: mariaolaru
"""
import sys
from preprocess_script import *
from plot_script import *

def main():
 
    args = sys.argv
#    script_name = globals()[args[1]]
#    path_list = args[2]

    #for testing purposes:
    path_list = "/Users/mariaolaru/google_drive/UCSF_Neuroscience/starr_lab_MO/studies/gamma_entrainment/meta_data/RCS10_ge_session_list.txt"

    [msc, gp] = preprocess_settings(path_list)
#    md = preprocess_data(path_list, msc) #separate fn b/c can take much longer time to process data

"""
    i_int = 13
    padding = [10, 120] #start and stop padding in seconds      
    step_size = 10 #units = seconds
    out_name = 'ge_transition'

    plot_PSD_long(md, msc, gp, i_int, padding, step_size, out_name)
"""

if __name__ == '__main__': 
    main()