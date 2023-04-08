#!/bin/bash

EXP_DIR=${PWD}
export EXP_DIR=${EXP_DIR}
echo "Script launched from ${EXP_DIR}"
echo "RL_PCB repository root is ${RL_PCB}"

mkdir -p work
echo "Starting tensorboard ... "
tensorboard --logdir ./work/ --host 0.0.0.0 &
sleep 2

cd ${RL_PCB}/src/training

./scheduler.sh --run_config ${EXP_DIR}/run_config_a.txt --logfile $EXP_DIR/scheduler_a.log --instances 6 --yes 
./scheduler.sh --run_config ${EXP_DIR}/run_config_b.txt --logfile $EXP_DIR/scheduler_b.log --instances 6 --yes 

cd ${EXP_DIR}

python report_config.py 

cd ${RL_PCB}/src/report_generation
python generate_experiment_report.py --dir ${EXP_DIR}/work --hyperparameters ${EXP_DIR}/hyperparameters/hp_td3.json ${EXP_DIR}/hyperparameters/hp_sac.json --report_config ${EXP_DIR}/report_config.json --output ${EXP_DIR}/experiment_report.pdf -y --tmp_dir ${EXP_DIR}/tmp
cd ${EXP_DIR}
