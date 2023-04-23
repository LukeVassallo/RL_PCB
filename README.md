- [Installation Guide](#installation-guide)
    - [GPU Setup](#gpu-setup)
        - [Install Nvidia GPU Driver](#install-nvidia-gpu-driver)
        - [Download and install CUDA toolkit](#download-and-install-cuda-toolkit)
    - [Setup Virtual Envrionment](#setup-virtual-envrionment)

# Installation Guide

Install Pre-requisites

```
sudo apt install python3-virtualenv build-essential libboost-dev libboost-filesystem-dev
```

## GPU Setup

1.  Check Nvidia compatability list
2.  Remove CUDA if instealled `sudo apt-get --purge remove cuda* *cublas* *cufft* *curand* *cusolver* *cusparse* *npp* *nvjpeg* *nsight*`
3.  Check if driver is installed, and if it is remove it with `sudo apt-get --purge remove *nvidia*`
4.  Remove cuddnn `sudo apt remove libcudnn* libcudnn*-dev`

### Install Nvidia GPU Driver

To install the driver, you can start by issuing the command `ubuntu-drivers` devices and identifying the latest third party non-free version. Once you have identified the appropriate version, use `apt` to install the driver. After installing the driver, reboot your system and issue the command `nvidia-smi` to identify the full driver version. You will need this information in the upcoming section to determine which CUDA version is supported.

### Download and install CUDA toolkit

To ensure that your device driver is compatible with CUDA, you'll need to check the compatibility using the following link: https://docs.nvidia.com/deploy/cuda-compatibility/. Once you've confirmed the compatibility, you can proceed to the CUDA Toolkit Archive at https://developer.nvidia.com/cuda-toolkit-archive. From there, select version 11.7 and then choose the appropriate platform parameters from the "Select Target Platform" section. Next, download the runfile (local) and proceed with the installation process. Keep in mind that CUDA 11.7 is supported for the GTX1080 with driver version 525.89.02. Finally, make sure to follow the installation instructions carefully and avoid installing the driver when prompted.

```
wget https://developer.download.nvidia.com/compute/cuda/11.7.1/local_installers/cuda_11.7.1_515.65.01_linux.run
sudo sh cuda_11.7.1_515.65.01_linux.run
```

Update the setup.sh script as necessary. The default contents for `PATH` and `LD_LIBRARY_PATH` are:

```
export PATH="/usr/local/cuda-11.7/bin:$PATH"
export LD_LIBRARY_PATH="/usr/local/cuda-11.7/lib64:$LD_LIBRARY_PATH"
```

## Setup Virtual Environment

Clone the repository

```
git clone https://gitlab.lukevassallo.com/luke/rl_pcb && cd rl_pcb
```

Before proceeding with the virtual environment, ensure that python 3.8.x is available. Create a virtual environment and setup.

```
python3 -m virtualenv venv --python=python3.8
source venv/bin/activate 

which python
python -c "import sys; print(sys.path)"
python -V

python -m pip install --upgrade pip
python -m pip install --upgrade setuptools==65.5.0	# See: https://github.com/openai/gym/issues/3176

# Refer to pytorch and update according to your cuda version
python -m pip install torch==1.13.1+cu117 torchvision==0.14.1+cu117 torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cu117

python -m pip install matplotlib numpy==1.23.3 opencv-python gym pyglet optuna tensorboard reportlab py-cpuinfo psutil pandas seaborn pynvml plotly moviepy

python -m pip install traceback-with-variables

python -m pip install -U kaleido

python -m pip install ./lib/pcb_netlist_graph-0.0.1-py3-none-any.whl
python -m pip install ./lib/pcb_file_io-0.0.1-py3-none-any.whl
```

The `setup.sh` script is already setup to activate the virtual environment. No further configuration is necessary.

## Installation of python3.8

```
sudo add-apt-repository ppa:deadsnakes/ppa -y
sudo apt update
sudo apt install python3.8
```
