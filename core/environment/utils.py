#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 13 14:59:54 2022

@author: luke
"""

#import sys
#sys.path.append('/home/luke/Desktop/semi_autonomous/py')
import os, sys
sys.path.append(os.path.join(os.environ["ITERATIVE_PLACER_REPO"], "py"))

import pcb as pcb

# utility function
def get_pcb_num(pcb_file: str):
    pv = pcb.vptr_pcbs()
    pcb.read_pcb_file(pcb_file, pv)      # Read pcb file
    return len(pv)    
