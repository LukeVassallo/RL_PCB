#!/bin/bash

# Place and route binaries
KICAD_PARSER=${RL_PCB}/bin/kicadParser
SA_PCB=${RL_PCB}/bin/sa
PCB_ROUTER=${RL_PCB}/bin/pcb_router

TEST_DIR=${PWD}
export TEST_DIR=${TEST_DIR}
echo "Script launched from ${TEST_DIR}"
echo "RL_PCB repository root is ${RL_PCB}"

mkdir -p work

cd ${RL_PCB}/src/training
./scheduler.sh --run_config ${TEST_DIR}/run_config.txt --logfile $TEST_DIR/scheduler.log --instances 4 --yes 
cd ${TEST_DIR}

python report_config.py 

cd ${RL_PCB}/src/report_generation
python generate_experiment_report.py --dir ${TEST_DIR}/work --hyperparameters ${TEST_DIR}/hyperparameters/hp_td3.json --report_config ${TEST_DIR}/report_config.json --output ${TEST_DIR}/experiment_report.pdf -y --tmp_dir ${TEST_DIR}/tmp
cd ${TEST_DIR}

# Check if all place and route binaries exist
if [ -e "$KICAD_PARSER" ] && [ -e "$SA_PCB" ] && [ -e "$PCB_ROUTER" ]; then
    echo "Starting evaluation ..."
    cd ${RL_PCB}/src/evaluation_scripts
    TD3_EVAL_TESTING_DIR=${TEST_DIR}/work/eval_testing_set
    SAC_EVAL_TESTING_DIR=${TEST_DIR}/work/eval_testing_set

    ./eval_just_do_it.sh -p ${RL_PCB}/dataset/base/evaluation.pcb -b ${RL_PCB}/dataset/base_raw --bin_dir ${RL_PCB}/bin --path_prefix "" -d ${TEST_DIR}/work -e training_td3_cuda_262 --report_type both,mean -o ${TD3_EVAL_TESTING_DIR} --runs 4 --max_steps 600

    cd ${TEST_DIR}
else
    echo "One or more place and route binaries expected at ${RL_PCB}/bin were not found."
fi
