#!/bin/bash

POSITIONAL_ARGS=()
RUN_CONFIG=""
LOGFILE=""
N=2
ASSUME_YES=false

draw_progress_bar()
{
	echo -ne '\r'
	for j in {1..100}
	do
		if (($1 >= $j)); then
			echo -ne '#' 
		else
			echo -ne ' '
		fi
	done
}

task() {
    eval $1 #> /dev/null
}

# initialize a semaphore with a given number of tokens
open_sem(){
    mkfifo pipe-$$
    exec 3<>pipe-$$
    rm pipe-$$
    local i=$1
    for((;i>0;i--)); do
        printf %s 000 >&3
    done
}

# run the given command asynchronously and pop/push tokens
run_with_lock(){
    local x
    # this read waits until there is something to read
    read -u 3 -n 3 x && ((0==x)) || exit $x
    (
     ( "$@"; )
    # push the return code of the command to the semaphore
    printf '%.3d' $? >&3
    )&
}

print_help() {
    echo "-i | --instances     number of concurrent runs."
    echo "--run_config         name of pickle file used to save the dataset. Must have a \'.pickle\' extension."
    echo "--help               print this help and exit."
}

while [[ $# -gt 0 ]]; do
    case $1 in
        -i|--instances)
            N="$2"
            shift   # past argument
            shift   # past value
            ;; 
        -c|--run_config)
            RUN_CONFIG="$2"
            shift   # past argument
            shift   # past value
            ;; 
        -l|--logfile)
            LOGFILE="$2"
            shift   # past argument
            shift   # past value
            ;;             
        -y|--yes)
            ASSUME_YES=true
            shift
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


echo "Run configuration file         : $RUN_CONFIG"
echo "Concurrent runs                : $N"
echo "Logfile                        : $LOGFILE"

echo ""
if [ "$ASSUME_YES" = false ]; then
	read -p "Do you want to continue?[Y/n] " -n 1 -r
	printf "\n\n"   # (optional) move to a new line
else
	REPLY=Y
fi

if [[ $REPLY =~ ^[Yy]$ ]]
then
    echo "Starting ... "
    V=$(date +%s) && RANDOM=$((V%32768))
    # Clean
    if [ "$LOGFILE" != "" ] 
    then
        if [ ! -d "dirname $LOGFILE" ]
        then
            mkdir -p $(dirname $LOGFILE)
        fi
    fi
    
    printf "\nStarting $(date +"%Y-%m-%dT%H:%M:%S%:z")\n\n" >> $LOGFILE

    # The following ensures that at any given moment N items are executing.
    #N=3
    open_sem $N
    ITERS=0
    echo ""     # newline for visual segmentation
    echo ""     # newline for visual segmentation

    while read -r line; 
    do
        run_with_lock task "$line"
        
        if [ "$LOGFILE" != "" ] 
        then
            echo $line >> $LOGFILE  # append
        fi
            
        sleep 1
        echo $line

    done  < $RUN_CONFIG

    wait # Waits on all processes to finish before moving on
    echo ""
    
    printf "\nExiting $(date +"%Y-%m-%dT%H:%M:%S%:z")\n\n" >> $LOGFILE
    
else
	echo "Terminating on user's request."
fi



