#!/bin/bash 
BIN=../../../../bin             # Default binary directory
BOARDS=../../../../boards       # Default boards directory
PCB=/home/luke/Desktop/semi_autonomous/boards/05_3_multi_agent/no_power_components/05_3_multi_agent_no_power_0.pcb
#PCB=/home/luke/Desktop/semi_autonomous/boards/05_3_multi_agent/no_power_components/05_3_multi_agent_no_power_eval_0.pcb
RUNS=4
PATH_PREFIX="."
REPORT_TYPE="mean"
SKIP_SA=false
SHUFFLE=false
MAX_STEPS=600
DEVICE="cuda"

print_help() {
    echo "-d, --dir                 directory containing experiment runs."
    echo "-e, --experiment          name of experiment whose agents to evaluate."
    echo "-o, --output              directory output."
    echo "--help                    print this help and exit."
}

while [[ $# -gt 0 ]]; do
    case $1 in
        -b|--boards_dir)
            BOARDS="${2%/}"             # strip trailing forward slash if any.
            shift
            shift
            ;;
        --bin_dir)
            BIN="${2%/}"                # strip trailing forward slash if any.
            shift
            shift
            ;;
#         -d|--dir)
#             EVALUATION_DIR="$2"
#             shift                       # past argument
#             shift                       # past value
#             ;;
        -d|--dir)
            EXP_DIR="${2%/}"
            shift                       # past argument
            shift                       # past value
            ;;
        --device)
            DEVICE="$2"
            shift                       # past argument
            shift                       # past value
            ;;            
        -e|--experiment)                # comma seperated and no spaces.
            ALL_EXP="$2"
            shift                       # past argument
            shift                       # past value
            ;;      
        --max_steps)
            MAX_STEPS="$2"
            shift
            shift
            ;;
        -o|--output)
            OUTPUT="${2%/}"             # strip trailing / if any
            shift
            shift
            ;;      
        --path_prefix)
            PATH_PREFIX="${2%/}"
            shift
            shift
            ;;
        -p|--pcb)
            PCB="$2"                    # strip trailing / if any
            shift
            shift
            ;;            
        -r|--runs)
            RUNS="$2"
            shift
            shift
            ;;
        --report_type)
            REPORT_TYPE="$2"
            shift
            shift
            ;;
        --shuffle)
            SHUFFLE=true
            shift
            ;;
        --skip_simulated_annealing)
            SKIP_SA=true
            shift   # past argument
            ;;  
        --help)
            print_help
            exit 0
            ;;
        -*|--*)
            echo "Unknown option $1"
            exit 1
            ;;
        *)
            POSITIONAL_ARGS+=($1)
            shift
            ;;
    esac
done

# TODO: run checks
BIN="${BIN%/}"                  # JIC defaults are provided with trailing forwardslash
BOARDS="${BOARDS%/}"            # ditto
PCB="${PCB%/}"            # ditto

EXP_DIR=$PATH_PREFIX/$EXP_DIR
OUTPUT=$PATH_PREFIX/$OUTPUT

echo ""
echo "bin (BIN)                          $BIN"
echo "board (BOARD)                      $BOARDS"
echo "Experiment directory (EXP_DIR)     $EXP_DIR"
echo "Experiment (EXP)                   $ALL_EXP"
echo "Output (OUTPUT)                    $OUTPUT"
echo "Pcb file (PCB)                     $PCB"
echo "Skip simulated annealing (SKIP_SA) $SKIP_SA"
echo ""
sleep 1

allRuns=()
allExps=()

IFS=',' 
for EXP in ${ALL_EXP}; do
    for dir in $EXP_DIR/*; do 
        echo "Searching for '$EXP' in '$dir'"
        for file in $dir/*; do
            if [[ ${file: -9} == "_desc.log" ]]; then
                while read -r line;
                do 
                    tmp=$(echo "$line" | awk '{split($0,a," -> "); if(a[1] == "experiment") {print a[2];} }')
                    if [ ! -z "$tmp" ];
                    then
                        EXPERIMENT=$(echo $tmp | tr -d '\r\n')
                        if [[ $EXPERIMENT != $EXP ]]; then
                            break;
                        fi
                    fi
                    
                    tmp=$(echo "$line" | awk '{split($0,a," -> "); if(a[1] == "run_name") {print a[2];} }')
                    if [ ! -z "$tmp" ];
                    then
                        RUN_NAME=$(echo $tmp | tr -d '\r\n')
                    fi
                    
                    tmp=$(echo "$line" | awk '{split($0,a," -> "); if(a[1] == "run") {print a[2];} }')
                    if [ ! -z "$tmp" ];
                    then
                        RUN=$(echo $tmp | tr -d '\r\n')
                    fi
                    
                    tmp=$(echo "$line" | awk '{split($0,a," -> "); if(a[1] == "rl_model_type") {print a[2];} }')
                    if [ ! -z "$tmp" ];
                    then
                        RL_MODEL_TYPE=$(echo $tmp | tr -d '\r\n')
                    fi
                    
                    tmp=$(echo "$line" | awk '{split($0,a," -> "); if(a[1] == "hyperparameters") {print a[2];} }')
                    if [ ! -z "$tmp" ];
                    then
                        HYPERPARAMETERS=$(echo $tmp | tr -d '\r\n')
                    fi
                    
                    tmp=$(echo "$line" | awk '{split($0,a," -> "); if(a[1] == "w") {print a[2];} }')
                    if [ ! -z "$tmp" ];
                    then
                        EUCLIDEAN_WIRELENGTH=$(echo $tmp | tr -d '\r\n')
                    fi
                    
                    tmp=$(echo "$line" | awk '{split($0,a," -> "); if(a[1] == "hpwl") {print a[2];} }')
                    if [ ! -z "$tmp" ];
                    then
                        HPWL=$(echo $tmp | tr -d '\r\n')
                    fi
                    
                    tmp=$(echo "$line" | awk '{split($0,a," -> "); if(a[1] == "o") {print a[2];} }')
                    if [ ! -z "$tmp" ];
                    then
                        OVERLAP=$(echo $tmp | tr -d '\r\n')
                    fi
                                        
                    
                done < $file
                
                if [[ $EXPERIMENT == $EXP ]]; then
    #                 echo ${RUN_NAME}
    #                 echo $RUN
                    REWARD_PARAMS=${EUCLIDEAN_WIRELENGTH}:${HPWL}:${OVERLAP}

                    allRuns+=( $OUTPUT/${RUN_NAME}_${RUN} )
                    allExps+=( $EXP )
                    allRewardParms+=( $REWARD_PARAMS ) 
                    
#                     echo "[DEBUG] RL_MODEL_TYPE=${RL_MODEL_TYPE}"
#                     echo "[DEBUG] model=${EXP_DIR}/${RUN_NAME}_${RUN}_${RL_MODEL_TYPE}/models/best_mean"
#                     echo "[DEBUG] PCB=$PCB"
#                     echo "[DEBUG] HYPERPARAMETERS=$PATH_PREFIX/$HYPERPARAMETERS"
# #                     echo "[DEBUG] RL_MODEL_TYPE=${RL_MODEL_TYPE}"
# #                     echo "[DEBUG] RL_MODEL_TYPE=${RL_MODEL_TYPE}"
#                     echo "[DEBUG] REWARD_PARAMS=$REWARD_PARAMS"

                    if [ "$SHUFFLE" = false ]; then
                        python eval_run_rl_policy.py --policy ${RL_MODEL_TYPE} --model="${EXP_DIR}/${RUN_NAME}_${RUN}_${RL_MODEL_TYPE}/models/best_mean" --pcb_file $PCB --hyperparameters $PATH_PREFIX/$HYPERPARAMETERS --max_steps $MAX_STEPS --runs $RUNS --reward_params $REWARD_PARAMS --output $OUTPUT/${RUN_NAME}_${RUN} --quick_eval --device $DEVICE #&> /dev/null   
                    else
                        python eval_run_rl_policy.py --policy ${RL_MODEL_TYPE} --model="${EXP_DIR}/${RUN_NAME}_${RUN}_${RL_MODEL_TYPE}/models/best_mean" --pcb_file $PCB --hyperparameters $PATH_PREFIX/$HYPERPARAMETERS --max_steps $MAX_STEPS --runs $RUNS --reward_params $REWARD_PARAMS --output $OUTPUT/${RUN_NAME}_${RUN} --quick_eval --device $DEVICE --shuffle_idxs #&> /dev/null   
                    fi
                 
                    
                    if [ "$SKIP_SA" = false ]; then
                        ./eval_place_and_route.sh -d $OUTPUT/${RUN_NAME}_${RUN} -b ${BOARDS} --bin_dir ${BIN} 
                    else
                        ./eval_place_and_route.sh -d $OUTPUT/${RUN_NAME}_${RUN} -b ${BOARDS} --bin_dir ${BIN} --skip_simulated_annealing
                    fi
                    
                    ./eval_generate_results_file.sh -d ${OUTPUT}/${RUN_NAME}_${RUN}
                
                fi
                
            fi
        done
    done
done

# Plot all reports
IFS=',' 
for RT in ${REPORT_TYPE}; do
    echo $RT
    if [ "$SKIP_SA" = false ]; then
        python eval_report_generator.py --run_dirs ${allRuns[*]} --experiments ${allExps[*]} --reward_params ${allRewardParms[*]} --max_steps $MAX_STEPS --report_type=${RT} --output ${OUTPUT}/evaluation_report_${RT}.pdf &> /dev/null
    else
        python eval_report_generator.py --run_dirs ${allRuns[*]} --experiments ${allExps[*]} --reward_params ${allRewardParms[*]} --max_steps $MAX_STEPS --report_type=${RT} --output ${OUTPUT}/evaluation_report_${RT}.pdf --skip_simulated_annealing &> /dev/null
    fi
done
