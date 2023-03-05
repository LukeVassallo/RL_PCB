#!/bin/bash

EXP_DIR=${PWD}
export EXP_DIR=${EXP_DIR}
echo "Script launched from ${EXP_DIR}"
echo "RL_PCB repository root is ${RL_PCB}"

cd ${RL_PCB}

./scheduler.sh --run_config ${EXP_DIR}/run_config.txt --logfile $EXP_DIR/scheduler.log --instances 4 --yes 

cd ${EXP_DIR}


python report_config.py 

cd ${RL_PCB}
python generate_experiment_report.py --dir ${EXP_DIR}/work --hyperparameters ${EXP_DIR}/hyperparameters/hp_td3.json,${EXP_DIR}/hyperparameters/hp_sac.json --report_config ${EXP_DIR}/report_config.json --output ${EXP_DIR}/experiment_report.pdf -y --tmp_dir ${EXP_DIR}/tmp
cd ${EXP_DIR}


cd ${RL_PCB}
TD3_EVAL_TESTING_DIR=${EXP_DIR}/work/eval_testing_set
SAC_EVAL_TESTING_DIR=${EXP_DIR}/work/eval_testing_set

./eval_just_do_it.sh -p ${RL_PCB}/dataset/base/evaluation.pcb -b ${RL_PCB}/dataset/base_raw --bin_dir ${RL_PCB}/bin --path_prefix "" -d ${EXP_DIR}/work -e parameter_experiment_262,parameter_experiment_622 --report_type both,mean -o ${TD3_EVAL_TESTING_DIR} --runs 2 --max_steps 200 --report_type both,mean --skip_simulated_annealing --workers 6

cd ${EXP_DIR}
