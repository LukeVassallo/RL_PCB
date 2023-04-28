#!/bin/bash

date="2023/04/28"

kicadParser_branch=parsing_and_plotting
SA_PCB_branch=crtYrd_bbox
pcbRouter_branch=updating_dependencies

#GIT=https://www.github.com/
#GIT_USER=lukevassallo
GIT=git@gitlab.lukevassallo.com:
GIT_USER=luke

CLEAN_ONLY=false
CLEAN_BEFORE_BUILD=false
RUN_PLACER_TESTS=false
RUN_ROUTER_TESTS=false

printf "\n"
printf "  **** Luke Vassallo M.Sc - 02_update_utility_binaries.sh\n"
printf "   *** Program to to update kicad parsing utility and place and route tols.\n"
printf "    ** Last modification time %s\n" $date
printf "\n"
sleep 1

print_help() {
    echo "--clean_only              removes the git repositories and exits."
    echo "--clean_before_build      removes the git repositories then clones and builds binaries."
    echo "--run_placer_tests	    runs automated tests to verify placer."
    echo "--run_router_tests        runs automated tests to verify router."
    echo "--help                    print this help and exit."
}

while [[ $# -gt 0 ]]; do
    case $1 in
        --clean_only)
            CLEAN_ONLY=true
            shift   # past argument
            ;;
	--clean_before_build)
	    CLEAN_BEFORE_BUILD=true
	    shift
	    ;;
	--run_placer_tests)
	    RUN_PLACER_TESTS=true
	    shift
	    ;;
	--run_router_tests)
	    RUN_ROUTER_TESTS=true
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

if [ -d "bin" ]; then
	cd bin
else
	mkdir bin && cd bin
fi

if [ "$CLEAN_ONLY" = true ] || [ "$CLEAN_BEFORE_BUILD" = true ]; then
	echo -n "Attempting to clean the KicadParser repository ... "
	if [ ! -d "KicadParser" ]; then
		echo "Not found, therefore nothing to clean.";
	else
		echo "Found, deleting."
		rm -fr KicadParser
	fi

	echo -n "Attempting to clean the SA_PCB repository ... "
	if [ ! -d "SA_PCB" ]; then
		echo "Not found, therefore nothing to clean.";
	else
		echo "Found, deleting."
		rm -fr SA_PCB
	fi
	
	echo -n "Attempting to clean the pcbRouter repository ... "
	if [ ! -d "pcbRouter" ]; then
		echo "Not found, therefore nothing to clean.";
	else
		echo "Found, deleting."
		rm -fr pcbRouter
	fi

	if [ "$CLEAN_ONLY" = true ]; then
		exit 0
	fi
fi

echo -n "Building kicad pcb parsing utility. Checking for repository ... "
ORIGIN=${GIT}${GIT_USER}/kicadParser
if [ -d "KicadParser" ]; then
	echo "Found, cleaning"
	cd KicadParser
	make clean
    	git pull $ORIGIN ${kicadParser_branch}
    #git submodule update --remote --recursive
else
	echo "Not found, cloning."
	git clone --branch ${kicadParser_branch} ${ORIGIN} --recurse-submodules KicadParser
	cd KicadParser
fi 	
make -j$(nproc)
cp -v build/kicadParser_test ../kicadParser
cd ..

echo -n "Building simulated annealing pcb placer. Checking for repository ... "
ORIGIN=${GIT}${GIT_USER}/SA_PCB
if [ -d "SA_PCB" ]; then
	echo "Found, cleaning"
	cd SA_PCB
	make clean
	git pull ${ORIGIN} ${SA_PCB_branch}
    #git submodule update --remote --recursive
else
	echo "Not found, cloning."
	git clone --branch ${SA_PCB_branch} ${ORIGIN} --recurse-submodules
	cd SA_PCB
fi 	
make -j$(nproc)
if [ "$RUN_PLACER_TESTS" = true ]; then
	make test_place_excl_power
	make test_place_incl_power
fi

#cp -v ./build/sa_placer_test ../bin/sa_placer
cp -v ./build/sa_placer_test ../sa
cd ..

echo -n "Building pcbRouter binary. Checking for repository ... "
ORIGIN=${GIT}${GIT_USER}/pcbRouter
if [ -d "pcbRouter" ]; then
	echo "Found, cleaning"
	cd pcbRouter
	make clean
    	git pull ${ORIGIN} ${pcbRouter_branch}
    #git submodule update --remote --recursive
else
	echo "Not found, cloning."
	git clone --branch ${pcbRouter_branch} ${ORIGIN} --recurse-submodules
	cd pcbRouter
fi 	
make -j$(nproc)
if [ "$RUN_ROUTER_TESTS" = true ]; then
	make test_route_excl_power
	make test_route_incl_power
fi

cp -v build/pcbRouter_test ../pcb_router
cd ..

cd .. 
