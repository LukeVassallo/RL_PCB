#!/bin/bash

date="2022/11/09"

kicadParser_branch=parsing_and_plotting
SA_PCB_branch=crtYrd_bbox
pcbRouter_branch=updating_dependencies

#GIT=https://www.github.com/
#GIT_USER=lukevassallo
GIT=git@gitlab.lukevassallo.com:
GIT_USER=luke

printf "\n"
printf "  **** Luke Vassallo M.Sc - 02_update_utility_binaries.sh\n"
printf "   *** Program to to update kicad parsing utility and place and route tols.\n"
printf "    ** Last modification time %s\n" $date
printf "\n"
sleep 1

if [ -d "bin" ]; then
	cd bin
else
	mkdir bin && cd bin
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
cp -v build/pcbRouter_test ../pcb_router
cd ..

cd .. 
