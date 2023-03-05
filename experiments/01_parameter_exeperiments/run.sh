#!/bin/bash

EXP_DIR=${PWD}
export EXP_DIR=${EXP_DIR}
echo "Script launched from ${EXP_DIR}"
echo "RL_PCB repository root is ${RL_PCB}"

cd ${RL_PCB}

./scheduler.sh --run_config ${EXP_DIR}/run_config.txt --yes --instances 4

cd ${EXP_DIR}


python report_config.py 

cd ${RL_PCB}
python generate_experiment_report.py --dir ${EXP_DIR}/work --hyperparameters ${EXP_DIR}/hyperparameters/hp_td3.json,${EXP_DIR}/hyperparameters/hp_sac.json --report_config ${EXP_DIR}/report_config.json --output ${EXP_DIR}/experiment_report.pdf -y --tmp_dir ${EXP_DIR}/tmp
cd ${EXP_DIR}
