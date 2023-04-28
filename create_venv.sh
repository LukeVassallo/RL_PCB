#!/bin/bash

CPU_ONLY=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --cpu_only)
            CPU_ONLY=true
            shift
            ;;
    esac
done

if [ -d "venv" ]; then
	echo "Virtual environment already exists ... Program terminating."
	exit
fi

python3.8 -m venv venv
source venv/bin/activate

which python
python -c "import sys; print(sys.path)"
python -V

python -m pip install --upgrade pip
python -m pip install --upgrade setuptools==65.5.0	# See: https://github.com/openai/gym/issues/3176

# Refer to pytorch and update according to your cuda version
if [ "$CPU_ONLY" == true ]; then
	python -m pip install torch==1.13.1+cpu torchvision==0.14.1+cpu torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cpu
else
	python -m pip install torch==1.13.1+cu117 torchvision==0.14.1+cu117 torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cu117
fi

python -m pip install matplotlib numpy==1.23.3 opencv-python gym pyglet optuna tensorboard reportlab py-cpuinfo psutil pandas seaborn pynvml plotly moviepy

python -m pip install -U kaleido

python -m pip install ./lib/pcb_netlist_graph-0.0.1-py3-none-any.whl
python -m pip install ./lib/pcb_file_io-0.0.1-py3-none-any.whl

python tests/00_verify_machine_setup/test_setup.py
