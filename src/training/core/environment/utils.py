"""
This module provides a utility function to retrieve the number of PCBs
(Printed Circuit Boards) from a given PCB file.

Module: pcb

Functions:

    get_pcb_num(pcb_file: str) -> int: Retrieves the number of PCBs from the
    specified PCB file.

Usage example:
from pcb import pcb

pcb_file = "example.pcb"
num_pcbs = pcb.get_pcb_num(pcb_file)
print("Number of PCBs:", num_pcbs)

"""
from pcb import pcb

# utility function
def get_pcb_num(pcb_file: str):
    pv = pcb.vptr_pcbs()
    pcb.read_pcb_file(pcb_file, pv)      # Read pcb file
    return len(pv)
