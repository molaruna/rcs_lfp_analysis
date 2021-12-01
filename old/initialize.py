#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 25 16:31:01 2021

@author: mariaolaru
"""
import numpy as np
import os

def find_file(file_name, parent_dir):
    
    #STEP 1: Get all files in all subdirectories of parent_dir
    array_all = np.array([])

    for root, subdirectories, files in os.walk(parent_dir):
        if file_name in files:
            file_match = os.path.join(root, file_name)
            array_all = np.append(array_all, file_match)
    
    return array_all