#!/bin/bash

BIN=./bin
BOARDS=./boards

HPWL="None"
OL="None"
RWL="None"

extract_parameters()
{
    # $1 file to parse
    sa_pcb=false
    pcbRouter=false

    HPWL="None"
    OL="None"
    RWL="None"
    
    while read -r line;
    do 
        if [[ $line == "begin sa_pcb" ]]; then
            sa_pcb=true
        fi
        if [[ $line == "end sa_pcb" ]]; then
            sa_pcb=false
        fi
        
        if [[ $line == "begin pcbRouter" ]]; then
            pcbRouter=true
        fi
        if [[ $line == "end pcbRouter" ]]; then
            pcbRouter=false
        fi
        
        
        if [ $sa_pcb ]; then
            tmp=$(echo "$line" | awk '{split($0,a,"="); if(a[1] == "wirelength") {print a[2];} }')
            if [ ! -z "$tmp" ];
            then
                #echo "wirelength=$tmp"
                HPWL=$tmp
            fi
            tmp=$(echo "$line" | awk '{split($0,a,"="); if(a[1] == "overlap") {print a[2];} }')
            if [ ! -z "$tmp" ];
            then
                #echo "overlap=$tmp"
                OL=$tmp
            fi
        fi

        if [ $pcbRouter ]; then
            tmp=$(echo "$line" | awk '{split($0,a,"="); if(a[1] == "routed_wirelength") {print a[2];} }')
            if [ ! -z "$tmp" ];
            then
                #echo "routed_wirelength=$tmp"
                RWL=$tmp
            fi
        fi
        
        
    done < $1
}

while [[ $# -gt 0 ]]; do
    case $1 in
        -d|--dir)
            EVALUATION_DIR="$2"
            shift   # past argument
            shift   # past value
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

# run checks

echo -n "" > ${EVALUATION_DIR}/results.txt

for board in ${EVALUATION_DIR}/* ; do
    brd=$(echo ${board} | rev | cut -d '/' -f 1 | rev)      # extract board name from name
    echo $brd
    
    for logfile in $board/sa_pcb/*.log; do
        if [ ! -f $logfile ]; then
            continue
        fi
#         while read -r line;
#         do
#             ORIGINAL_KICAD_PCB=$(echo "$line" | awk '{split($0,a,"="); if(a[1] == ".kicad_pcb") {print a[2];} }')
#             if [ ! -z "$ORIGINAL_KICAD_PCB" ];
#             then
#                 break
#             fi
#             #echo $line
#         done < $pcb
#         echo $logfile    
        num=$(echo ${logfile: 0:-4} | rev | cut -d '_' -f 1 | rev)      # extract trial from name
        extract_parameters $logfile
        echo "$brd,trial_$num,SA_PCB,$HPWL,$OL,$RWL" >> ${EVALUATION_DIR}/results.txt

    done
    
    for trial in $board/trial_*; do
    
        if [ ! -d $trial ]; then
            continue
        fi

        trial_name=$(echo $trial | rev | cut -d '/' -f 1 | rev) # extract last field.
        for logfile in $trial/*.log; do
            if [ ! -f $logfile ]; then
                continue
            fi
            echo "[DEBUG - eval_generate_results_file] - Using file -> $logfile"	    
            if [ ${logfile: -24:-4} == "best_hpwl_00_overlap" ]; then
                extract_parameters $logfile
                echo "$brd,$trial_name,${logfile: -24:-4},$HPWL,$OL,$RWL" >> ${EVALUATION_DIR}/results.txt                
            fi
    
            if [ ${logfile: -24:-4} == "best_hpwl_10_overlap" ]; then
                extract_parameters $logfile
                echo "$brd,$trial_name,${logfile: -24:-4},$HPWL,$OL,$RWL" >> ${EVALUATION_DIR}/results.txt
            fi
    
            if [ ${logfile: -24:-4} == "best_hpwl_20_overlap" ]; then
                extract_parameters $logfile
                echo "$brd,$trial_name,${logfile: -24:-4},$HPWL,$OL,$RWL" >> ${EVALUATION_DIR}/results.txt
            fi
        done
        
         
            
    done
    echo ""
    
done
