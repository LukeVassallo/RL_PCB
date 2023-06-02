"""
This module provides functions for collecting and displaying machine and
library information.

Usage:

1. Call `machine_info_in_paragraphs(style)` to retrieve machine information
   in the form of paragraphs.
   - `style`: The style of the paragraphs (e.g., font, size, color).

2. Call `lib_info_in_paragraphs(style)` to retrieve library information
   in the form of paragraphs.
   - `style`: The style of the paragraphs (e.g., font, size, color).

Note: Make sure to install the required dependencies and import the necessary
modules before using this code.
"""


from cpuinfo import get_cpu_info
import psutil
import numpy as np
import optuna
import pandas
import matplotlib
import seaborn

import os

from pynvml import nvmlSystemGetDriverVersion
from pynvml import nvmlDeviceGetCount
from pynvml import nvmlDeviceGetName
from pynvml import nvmlDeviceGetHandleByIndex
from pynvml import nvmlInit, nvmlShutdown
from pynvml import nvmlDeviceGetMemoryInfo

import sys
import torch

from pcb import pcb
from graph import graph     # Necessary for graph related methods

from reportlab.platypus import Paragraph

def machine_info_in_paragraphs(style):
    data = []

    data.append(Paragraph(f'sysname={os.uname()[0]}',style))
    data.append(Paragraph(f'nodename={os.uname()[1]}',style))
    data.append(Paragraph(f'release={os.uname()[2]}',style))
    data.append(Paragraph(f'version={os.uname()[3]}',style))
    data.append(Paragraph(f'machine={os.uname()[4]}',style))
    data.append(Paragraph('<br />',style))

    info = get_cpu_info()
    data.append(Paragraph(f'CPU arch                : {info["arch"]}',style))
    data.append(Paragraph(f'CPU bits                : {info["bits"]}',style))
    data.append(Paragraph(
        f'CPU brand               : {info["brand_raw"]}',style))
    data.append(Paragraph(f'CPU cores               : {info["count"]}',style))
    data.append(Paragraph(
        f'CPU base clock          : {info["hz_advertised_friendly"]}',style))
    data.append(Paragraph(
        f'CPU boost clock         : {info["hz_actual_friendly"]}',style))
    #print(f'CPU L1 I/D cache       : {info["l1_instruction_cache_size"]/1024}kB / {info["l1_data_cache_size"]/1024}kB')
    #print(f'CPU L2 cache           : {info["l2_cache_size"]/1024}kB ')
    #print(f'CPU L3 cache           : {info["l3_cache_size"]/(1024*1024)}MB')
    #print('')
    data.append(Paragraph(f'System Memory           : {np.round(psutil.virtual_memory().total / (1024**3),2)}GB',style))
    nvmlInit()
    data.append(Paragraph(
        f'Nvidia driver version   : {str(nvmlSystemGetDriverVersion())}',style))
    deviceCount = nvmlDeviceGetCount()
    for i in range(deviceCount):
        handle = nvmlDeviceGetHandleByIndex(i)
        data.append(Paragraph(f'Device {i}                : {str(nvmlDeviceGetName(handle))}',style))
        data.append(Paragraph(f'Device {i}                : {np.round(nvmlDeviceGetMemoryInfo(handle).total / 1024**3,2)}GB',style))

    nvmlShutdown()

    return data

def lib_info_in_paragraphs(style):
    data = []
    major=sys.version_info.major
    minor=sys.version_info.minor
    micro=sys.version_info.micro
    data.append(Paragraph(
        f'python                  : {major}.{minor}.{micro}',style))
    data.append(Paragraph(
        f'torch                   : {torch.__version__}',style))
    data.append(Paragraph(
        f'optuna                  : {optuna.__version__}',style))
    data.append(Paragraph(
        f'numpy                   : {np.__version__}',style))
    data.append(Paragraph(
        f'pandas                  : {pandas.__version__}',style))
    data.append(Paragraph(
        f'matplotlib              : {matplotlib.__version__}',style))
    data.append(Paragraph(
        f'seaborn                 : {seaborn.__version__}',style))

    data.append(Paragraph(
        pcb.build_info_as_string().replace('\n','<br />')[:-6],style))
    data.append(Paragraph(
        graph.build_info_as_string().replace('\n','<br />'),style))

    return data
