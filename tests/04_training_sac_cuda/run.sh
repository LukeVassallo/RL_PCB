#!/bin/bash

TEST_DIR=${PWD}
export TEST_DIR=${TEST_DIR}
echo "Script launched from ${TEST_DIR}"
echo "RL_PCB repository root is ${RL_PCB}"

cd ${RL_PCB}/src/training
./scheduler.sh --run_config ${TEST_DIR}/run_config.txt --logfile $TEST_DIR/scheduler.log --instances 4 --yes 
cd ${TEST_DIR}

python report_config.py 

cd ${RL_PCB}/src/report_generation
python generate_experiment_report.py --dir ${TEST_DIR}/work --hyperparameters ${TEST_DIR}/hyperparameters/hp_sac.json --report_config ${TEST_DIR}/report_config.json --output ${TEST_DIR}/experiment_report.pdf -y --tmp_dir ${TEST_DIR}/tmp
cd ${TEST_DIR}

cd ${RL_PCB}/src/evaluation_scripts
TD3_EVAL_TESTING_DIR=${TEST_DIR}/work/eval_testing_set
SAC_EVAL_TESTING_DIR=${TEST_DIR}/work/eval_testing_set

./eval_just_do_it.sh -p ${RL_PCB}/dataset/base/evaluation.pcb -b ${RL_PCB}/dataset/base_raw --bin_dir ${RL_PCB}/bin --path_prefix "" -d ${TEST_DIR}/work -e training_sac_cuda_622 --report_type both,mean -o ${TD3_EVAL_TESTING_DIR} --runs 2 --max_steps 200 --report_type both,mean --skip_simulated_annealing --workers 6

cd ${TEST_DIR}
