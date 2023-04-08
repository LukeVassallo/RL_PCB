#!/bin/bash

RL_PCB=${PWD}
export RL_PCB
echo "RL_PCB=${RL_PCB}"

export PATH="/usr/local/cuda-11.7/bin:$PATH"
export LD_LIBRARY_PATH="/usr/local/cuda-11.7/lib64:$LD_LIBRARY_PATH"

echo "LD_LIBRARY_PATH=${LD_LIBRARY_PATH}"
echo "PATH=${PATH}"

conda activate rl_pcb		# Used for quickly setting up alternative version of python
source venv/bin/activate	# True virutal environment
