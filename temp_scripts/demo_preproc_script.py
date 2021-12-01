#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 18 20:06:59 2021

@author: mariaolaru
"""
from preproc.preprocess_funcs import *

def combine_data(dir_name):
    [df_settings, df_notes, gp] = preprocess_settings(dir_name)
    md = preprocess_data(dir_name, df_settings, gp) #separate fn b/c can take much longer time to process data
    
    df = link_data(msc, md) #create combined dataframe
    return [df, df_notes]

dir_name = '/Users/mariaolaru/Documents/temp/RCS10L/RCS10L_post-stim/'

[df, df_notes] = combine_data(dir_name)