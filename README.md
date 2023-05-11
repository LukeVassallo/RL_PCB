- [Installation Guide](#installation-guide)
    - [GPU Setup](#gpu-setup)
        - [Install Nvidia GPU Driver](#install-nvidia-gpu-driver)
        - [Download and install CUDA toolkit](#download-and-install-cuda-toolkit)
    - [Setup Virtual Envrionment](#setup-virtual-envrionment)

RL\_PCB is a novel learning based method for optimising the placement of circuit comoponents on a Printed Circuit Board (PCB). 

Main contribution of this work
1. The policies learn the fundamental rules of the task and demonstrate an understanding of the problem dynamics. The agent is observed taking actions that **in the long term** minimise overlap-free wirelength, while the components naturally fall in place resulting in a coherent layout. 
2. Since the agent represents a component, emergent behaviours are observered as a result of each component interacting with its neighbours. When we emphase HPWL in the reward function we observe collaboration, on ther other hand when we emphasise EW we observe competition. 
3. The learned behaviour is robust because training data is diverse and consistent with the evaluative feedback assigned. Consistency is achieved by exensive normalisation and eliminating all potential sources that introduce bias. Similarly goes for the reward. Diversity is obtained by allowing every agent to contribute to training samples with diffferent perspectives. 

|     |     |     |
| --- | --- | --- |
| ![055_15.gif](.figs/055_15.gif) <br /> (EW=0, HPWL=5, Overlap=5) | ![055_15.gif](.figs/802_14.gif) <br /> (EW=8, HPWL=0, Overlap=2) | ![055_15.gif](.figs/082_14.gif) <br /> (EW=0, HPWL=8, Overlap=2)     |
| ![policy.gif](.figs/policy.gif) <br /> (EW=0, HPWL=8, Overlap=2)  | ![policy_802_td3.gif](.figs/policy_802_td3.gif) <br /> (EW=8, HPWL=0, Overlap=2)| ![policy_802_sac.gif](.figs/policy_802_sac.gif) <br /> (EW=8, HPWL=0, Overlap=2)|
| ![policy_802_sac_b.gif](.figs/policy_802_sac_b.gif) <br /> (EW=8, HPWL=0, Overlap=2)  | ![policy_sac_226.gif](.figs/policy_sac_226.gif) <br /> (EW=2, HPWL=2, Overlap=6) | ![policy_td3_226.gif](.figs/policy_td3_226.gif)  <br /> (EW=2, HPWL=2, Overlap=6)   |
# Installation Guide
**It is very important that the installation procedure is carried out while being in the root of the repository (i.e. the same location as the script install_tools_and_virtual_environment.sh)**

## Install Pre-requisites
```
sudo apt install build-essential libboost-dev libboost-filesystem-dev
```

All python code uses python version 3.8. Additionally python virtual envrionments are needed to install dependencies in a contained environment without altering the system configuration. 
```
sudo add-apt-repository ppa:deadsnakes/ppa -y
sudo apt update
sudo apt install python3.8 python3.8-venv
```

## Run automated installation script
The automated installation procedure makes the following changes to the local repository:
- Create a directory bin and installs the KiCad parsing utility, and place and route tools
- Creates a environment using python3.8, installs pytorch 1.13 with CUDA 11.7 and all necessary python packages
- Installs the wheel libraries in the lib folder

```
./install_tools_and_virtual_environment.sh
```

# Run tests and experiments
Always source the envrionment setup script before running any tests or experiments. **The script should be run from the root of the repository**
```
cd <path-to-rl_pcb>
source setup.sh 
```

Run an experiment
```
cd experiments/00_parameter_exeperiments
./run.sh 	
```

Run a test
```
cd tests/01_training_td3_cpu
./run.sh
```

The script `run.sh` will perform the following: 
1. Carry out the training run(s) by following the instructions in `run_config.txt` that is located within the same directory
2. Generates an experiment report that processes the experimental data and presents the results in tables and figures. All experiment metadata is also reported and customisation is possible through `report_config.py` that is location within the same directory.
3. Perform evaluation of all policies alongside simulated annealing baseline. All optimised placements are subsequently routed using an A\* based algorithm. 
4. Generate and evaluation report that processes all evaluation data and tabulates HPWL and routed wirelength metrics. All experiment metadata is also reported.

The generated files can be cleaned by running
```
./clean.sh
```

Every test and experiment contains a directory `expected results` that contains pre-generated reports. Should you run the experiments as provided, identical results are to be expected.

# GPU Setup (Optional)
**The commands in this section do big changes to your system. Please read carefully before running commands**

1.  Check Nvidia compatability list
2.  Remove CUDA if instealled `sudo apt-get --purge remove cuda* *cublas* *cufft* *curand* *cusolver* *cusparse* *npp* *nvjpeg* *nsight*`
3.  Check if driver is installed, and if it is remove it with `sudo apt-get --purge remove *nvidia*`
4.  Remove cuddnn `sudo apt remove libcudnn* libcudnn*-dev`

### Install Nvidia GPU Driver

To install the driver, you can start by issuing the command `ubuntu-drivers` devices and identifying the latest third party non-free version. Once you have identified the appropriate version, use `apt` to install the driver. After installing the driver, reboot your system and issue the command `nvidia-smi` to identify the full driver version. You will need this information in the upcoming section to determine which CUDA version is supported.

### Download and install CUDA toolkit
To ensure that your device driver is compatible with CUDA, you'll need to check the compatibility using the following link: https://docs.nvidia.com/deploy/cuda-compatibility/. Once you've confirmed the compatibility, you can proceed to the CUDA Toolkit Archive at https://developer.nvidia.com/cuda-toolkit-archive. From there, select version 11.7 and then choose the appropriate platform parameters from the "Select Target Platform" section. Next, download the runfile (local) and proceed with the installation process. Finally, make sure to follow the installation instructions carefully and avoid installing the driver when prompted.

```
wget https://developer.download.nvidia.com/compute/cuda/11.7.1/local_installers/cuda_11.7.1_515.65.01_linux.run
sudo sh cuda_11.7.1_515.65.01_linux.run
``

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

python -m pip install -U kaleido

python -m pip install ./lib/pcb_netlist_graph-0.0.1-py3-none-any.whl
python -m pip install ./lib/pcb_file_io-0.0.1-py3-none-any.whl
```

The `setup.sh` script is already setup to activate the virtual environment. No further configuration is necessary.
