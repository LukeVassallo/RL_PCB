import pcb.pcb as pcb 
import graph.graph as graph
import torch 
from datetime import datetime

from cpuinfo import get_cpu_info
import psutil
import numpy as np
import optuna
import pandas
import matplotlib
import seaborn

import os, sys

print("Program started at " + datetime.now().strftime("%Y-%m-%d %H:%M:%S:%f")+"\n")

print(f'sysname            : {os.uname()[0]}')
print(f'nodename           : {os.uname()[1]}')
print(f'release            : {os.uname()[2]}')
print(f'version            : {os.uname()[3]}')
print(f'machine            : {os.uname()[4]}')
print("")

info = get_cpu_info()
print(f'CPU arch           : {info["arch"]}')
print(f'CPU bits           : {info["bits"]}')
print(f'CPU brand          : {info["brand_raw"]}')
print(f'CPU cores          : {info["count"]}')
print(f'CPU base clock     : {info["hz_advertised_friendly"]}')
print(f'CPU boost clock    : {info["hz_actual_friendly"]}')
print(f'System Memory      : {np.round(psutil.virtual_memory().total / (1024**3),2)}GB')
print("")

#nvmlInit()
#data.append(Paragraph(f'Nvidia driver version   : {str(nvmlSystemGetDriverVersion())}',style))
#deviceCount = nvmlDeviceGetCount()
#for i in range(deviceCount):
    #handle = nvmlDeviceGetHandleByIndex(i)
    #data.append(Paragraph(f'Device {i}                : {str(nvmlDeviceGetName(handle))}',style))
    #data.append(Paragraph(f'Device {i}                : {np.round(nvmlDeviceGetMemoryInfo(handle).total / 1024**3,2)}GB',style))

#nvmlShutdown()

    #return data


print(f'python             : {sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}')
print(f'torch              : {torch.__version__}')
print(f'optuna             : {optuna.__version__}')
print(f'numpy              : {np.__version__}')
print(f'pandas             : {pandas.__version__}')
print(f'matplotlib         : {matplotlib.__version__}')
print(f'seaborn            : {seaborn.__version__}')
print("")

if torch.cuda.is_available():
    print(f"Using CUDA {torch.version.cuda}")
    print("Available devices:")
    for dev in range(torch.cuda.device_count()):
        print(    torch.cuda.get_device_name(dev))
    print()    
     
    device = torch.device("cuda:0")
    print(f"Running on {device} - {torch.cuda.get_device_name(device)}\n")
    
    print('Memory Usage:')
    print('Allocated:', round(torch.cuda.memory_allocated(0)/1024**3,1), 'GB')
    print('Cached:   ', round(torch.cuda.memory_reserved(0)/1024**3,1), 'GB')
    
else:
    device = torch.device("cpu")    
    print("Running on CPU")

graph.build_info()
pcb.build_info()

print('')
print('Testing place and route binaries')
print(f"Testing kicadParser ... {'OK' if os.system('${RL_PCB}/bin/kicadParser --help > /dev/null') == 0 else 'Failed'}")
print(f"Testing sa ... {'OK' if os.system('${RL_PCB}/bin/sa --help > /dev/null') == 0 else 'Failed'}")
print(f"Testing pcb_router ... {'OK' if os.system('${RL_PCB}/bin/pcb_router --help > /dev/null') == 0 else 'Failed'}")
print('')
