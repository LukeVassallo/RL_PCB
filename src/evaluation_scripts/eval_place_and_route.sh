#!/bin/bash

BIN=../../../../bin             # Default binary directory
BOARDS=../../../boards       # Default boards directory

KICAD_PARSER=kicadParser
SA_PLACER=sa
PCB_ROUTER=pcb_router
SKIP_SA=false

kicad_pcb_from_pcb()
{
    echo $1
    while read -r line;
    do
        kicad_pcb=$(echo "$line" | awk '{split($0,a,"="); if(a[1] == ".kicad_pcb") {print a[2];} }')
    if [ ! -z "$kicad_pcb" ];
    then
        return $kicad_pcb
    fi
        #echo $line
    done < $1
}

route_rl_generated_pcb()
{
    # $1 - output directory
    # $2 - original kicad pcb
    # $3 - FILENAME without extension to derivde KICAD_PCB, ROUTED_FILENAME and LOGFILE
    # $4 - $PCB_FILE
    OUTPUT_DIR=$1       
    ORIGINAL_KICAD_PCB=$2
    NEW_FILENAME=$3
    PCB_FILE=$4

    KICAD_PCB=$NEW_FILENAME.kicad_pcb
    ROUTED_FILENAME=${NEW_FILENAME}_routed.kicad_pcb
    LOGFILE=$NEW_FILENAME.log

    cp -v $BOARDS/$ORIGINAL_KICAD_PCB ${OUTPUT_DIR}/${KICAD_PCB}

    $BIN/${KICAD_PARSER} --kicad_pcb ${OUTPUT_DIR}/${KICAD_PCB} --pcb $PCB_FILE --update -o $OUTPUT_DIR --skip_kicad_pcb_validation &> /dev/null # skip .kicad_pcb name validation since the file is being renamed. Otherwise no parsing/updating will occur.
    $BIN/${SA_PLACER} --kicad_pcb $OUTPUT_DIR/$KICAD_PCB --logfile $OUTPUT_DIR/$LOGFILE --print_cost_and_exit --ignore_power_nets &> /dev/null
    $BIN/${PCB_ROUTER} --kicad_pcb $OUTPUT_DIR/$KICAD_PCB --output_dir $OUTPUT_DIR --name $OUTPUT_DIR/$ROUTED_FILENAME --logfile $OUTPUT_DIR/$LOGFILE --rip_up_reroute_iterations=0 --layer_change_cost=100 --ignore_power_nets &> /dev/null
}

print_help() {
    echo "-d, --dir                 directory containing evaluation runs."
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
            BIN="${2%/}"             # strip trailing forward slash if any.
            shift
            shift
            ;;
        -d|--dir)
            EVALUATION_DIR="$2"
            shift   # past argument
            shift   # past value
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

if [ ! -f "${BIN}/${KICAD_PARSER}" ]; then
    echo "Could not find ${BIN}/${KICAD_PARSER} ... Program terminating."
    exit -1
fi

if [ ! -f "${BIN}/${SA_PLACER}" ]; then
    echo "Could not find ${BIN}/${SA_PLACER} ... Program terminating."
    exit -1
fi

if [ ! -f "${BIN}/${PCB_ROUTER}" ]; then
    echo "Could not find ${BIN}/${PCB_ROUTER} ... Program terminating."
    exit -1
fi

CNTR=0
for board in ${EVALUATION_DIR}/* ; do
    echo $board
    CNTR=0
    if [ "$SKIP_SA" = false ]; then
        for pcb in $board/sa_pcb/*.pcb; do

            while read -r line;
            do
                ORIGINAL_KICAD_PCB=$(echo "$line" | awk '{split($0,a,"="); if(a[1] == ".kicad_pcb") {print a[2];} }')
                if [ ! -z "$ORIGINAL_KICAD_PCB" ];
                then
                    break
                fi
                #echo $line
            done < $pcb
            
            OUTPUT_DIR=${board}/sa_pcb
            
            FILE=$(cut -d '.' -f 1 <<< "$ORIGINAL_KICAD_PCB")  # returns the first field in a string seperated with underscores '_'
            NEW_FILENAME=${FILE}_${CNTR}
            
            KICAD_PCB=$NEW_FILENAME.kicad_pcb
            ROUTED_FILENAME=${NEW_FILENAME}_routed.kicad_pcb
            LOGFILE=$NEW_FILENAME.log
            
            cp -v $BOARDS/$ORIGINAL_KICAD_PCB ${OUTPUT_DIR}/${KICAD_PCB}
            
            $BIN/${KICAD_PARSER} --kicad_pcb ${OUTPUT_DIR}/${KICAD_PCB} --pcb $pcb --update -o $OUTPUT_DIR --skip_kicad_pcb_validation &> /dev/null # skip .kicad_pcb name validation since the file is being renamed. Otherwise no parsing/updating will occur.
            $BIN/${SA_PLACER} --kicad_pcb $OUTPUT_DIR/$KICAD_PCB -i 500 --output_dir $OUTPUT_DIR --name $OUTPUT_DIR/$KICAD_PCB --logfile $OUTPUT_DIR/$LOGFILE --ignore_power_nets &> /dev/null
#             $BIN/${SA_PLACER} --kicad_pcb $OUTPUT_DIR/$KICAD_PCB -i 500 --output_dir $OUTPUT_DIR --name $OUTPUT_DIR/$KICAD_PCB --logfile /dev/null &> /dev/null   # if logfile is omitted, one will be created having the same URI as the output file albeit with a .log extension
#             $BIN/${SA_PLACER} --kicad_pcb $OUTPUT_DIR/$KICAD_PCB -i 500 --output_dir $OUTPUT_DIR --name $OUTPUT_DIR/$KICAD_PCB --logfile $OUTPUT_DIR/$LOGFILE --ignore_power_nets --print_cost_and_exit &> /dev/null

            # Convert the optimised layout into a .pcb file and draw it as a .png image.
            PLACED_PCB=${NEW_FILENAME}_placed.pcb
            PLACED_PNG=${NEW_FILENAME}_placed.png
            $BIN/${KICAD_PARSER} --kicad_pcb ${OUTPUT_DIR}/${KICAD_PCB} --generate_pcb --pcb ${OUTPUT_DIR}/${PLACED_PCB} &> /dev/null # skip .kicad_pcb name validation since the file is being renamed. Otherwise no parsing/updating will occur.
            python pcb2png.py --pcb ${OUTPUT_DIR}/${PLACED_PCB} --output ${OUTPUT_DIR}/${PLACED_PNG}
            
            # Remove temporary files generated during the creation of .pcb file.
            if [ -d "tmp" ]; then
            	rm -fr tmp
            fi

            $BIN/${PCB_ROUTER} --kicad_pcb $OUTPUT_DIR/$KICAD_PCB --output_dir $OUTPUT_DIR --name $OUTPUT_DIR/$ROUTED_FILENAME --logfile $OUTPUT_DIR/$LOGFILE --rip_up_reroute_iterations=0 --layer_change_cost=100 --ignore_power_nets &> /dev/null
            
            ((CNTR+=1))
        
        done
    fi        

    for trial in $board/trial_*; do
        if [ -f "$trial/best_hpwl_00_overlap.pcb" ]; then
#             echo "Found $trial/best_hpwl_zero_overlap.pcb"
            PCB_FILE=$trial/best_hpwl_00_overlap.pcb
            SUFFIX=best_hpwl_00_overlap
            
            while read -r line;
            do
                ORIGINAL_KICAD_PCB=$(echo "$line" | awk '{split($0,a,"="); if(a[1] == ".kicad_pcb") {print a[2];} }')
                if [ ! -z "$ORIGINAL_KICAD_PCB" ];
                then
                    break
                fi
                #echo $line
            done < $trial/best_hpwl_00_overlap.pcb
            
            OUTPUT_DIR=${trial} 
            FILE=$(cut -d '.' -f 1 <<< "$ORIGINAL_KICAD_PCB")  # returns the first field in a string seperated with underscores '_'

            NEW_FILENAME=${FILE}_${SUFFIX}
            route_rl_generated_pcb $OUTPUT_DIR $ORIGINAL_KICAD_PCB $NEW_FILENAME $PCB_FILE
        fi
        
        if [ -f "$trial/best_hpwl_10_overlap.pcb" ]; then
#             echo "Found $trial/best_hpwl_10_overlap.pcb"
            PCB_FILE=$trial/best_hpwl_10_overlap.pcb
            SUFFIX=best_hpwl_10_overlap
            
            while read -r line;
            do
                ORIGINAL_KICAD_PCB=$(echo "$line" | awk '{split($0,a,"="); if(a[1] == ".kicad_pcb") {print a[2];} }')
                if [ ! -z "$ORIGINAL_KICAD_PCB" ];
                then
                    break
                fi
                #echo $line
            done < $trial/best_hpwl_10_overlap.pcb   
            
            OUTPUT_DIR=${trial} 
            FILE=$(cut -d '.' -f 1 <<< "$ORIGINAL_KICAD_PCB")  # returns the first field in a string seperated with underscores '_'            
            
            NEW_FILENAME=${FILE}_${SUFFIX}
            route_rl_generated_pcb $OUTPUT_DIR $ORIGINAL_KICAD_PCB $NEW_FILENAME $PCB_FILE
        fi
        
        if [ -f "$trial/best_hpwl_20_overlap.pcb" ]; then
#             echo "Found $trial/best_hpwl_20_overlap.pcb"
            PCB_FILE=$trial/best_hpwl_20_overlap.pcb
            SUFFIX=best_hpwl_20_overlap
            
            while read -r line;
            do
                ORIGINAL_KICAD_PCB=$(echo "$line" | awk '{split($0,a,"="); if(a[1] == ".kicad_pcb") {print a[2];} }')
                if [ ! -z "$ORIGINAL_KICAD_PCB" ];
                then
                    break
                fi
                #echo $line
            done < $trial/best_hpwl_20_overlap.pcb         
            
            OUTPUT_DIR=${trial} 
            FILE=$(cut -d '.' -f 1 <<< "$ORIGINAL_KICAD_PCB")  # returns the first field in a string seperated with underscores '_'            
            
            NEW_FILENAME=${FILE}_${SUFFIX}
            route_rl_generated_pcb $OUTPUT_DIR $ORIGINAL_KICAD_PCB $NEW_FILENAME $PCB_FILE
        fi
            
    done
    
done
    
