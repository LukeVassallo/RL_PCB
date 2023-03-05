import pcb.pcb as pcb 
import graph.graph as graph
import torch 
from datetime import datetime

print("Program started at " + datetime.now().strftime("%Y-%m-%d %H:%M:%S:%f")+"\n")

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
