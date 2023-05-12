#!/bin/bash

CUDA_VERSION=11.7

# Setup repository environment variable
RL_PCB=${PWD}
export RL_PCB
echo "RL_PCB=${RL_PCB}"

# Make CUDA available
if [ -d "/usr/local/cuda-${CUDA_VERSION}" ]; then
	export PATH="/usr/local/cuda-${CUDA_VERSION}/bin:$PATH"
	export LD_LIBRARY_PATH="/usr/local/cuda-${CUDA_VERSION}/lib64:$LD_LIBRARY_PATH"
else
	echo "Could not find ${CUDA} on system."
fi

echo "LD_LIBRARY_PATH=${LD_LIBRARY_PATH}"
echo "PATH=${PATH}"

# Activate virtual environment
echo -n "Attempting to activate virtual environment ${RL_PCB}/venv ... "
if [ -d "${RL_PCB}/venv" ]; then
	source ${RL_PCB}/venv/bin/activate	# True virutal environment
	echo "OK"
else
	echo "Failed, venv does not exist."
	echo "Please use script 'create_venv.sh' to automatically setup the environment."
fi
