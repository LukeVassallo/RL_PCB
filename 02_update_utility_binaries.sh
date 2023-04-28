#!/bin/bash

date="2022/11/09"

KicadParser_branch=parsing_and_plotting
SA_PCB_branch=crtYrd_bbox
pcbRouter_branch=updating_dependencies

#GIT=https://www.github.com
#GIT_USER=lukevassallo
GIT=https://gitlab.lukevassallo.com
GIT_USER=luke

printf "\n"
printf "  **** Luke Vassallo M.Sc - 02_update_utility_binaries.sh\n"
printf "   *** Program to to update kicad parsing utility and place and route tols.\n"
printf "    ** Last modification time %s\n" $date
printf "\n"
sleep 1

cd bin
echo -n "Building kicad pcb parsing utility. Checking for repository ... "
if [ -d "KicadParser" ]; then
	echo "Found, cleaning"
	cd KicadParser
	make clean
    git pull origin ${KicadParser_branch}
    #git submodule update --remote --recursive
else
	echo "Not found, cloning."
	git clone --branch ${KicadParser_branch} ${GIT}/${GIT_USER}/KicadParser --recurse-submodules
	cd KicadParser
fi 	
make -j8
cp -v build/kicadParser_test ../kicadParser
cd ..

echo -n "Building simulated annealing pcb placer. Checking for repository ... "
if [ -d "SA_PCB" ]; then
	echo "Found, cleaning"
	cd SA_PCB
	make clean
    git pull origin ${SA_PCB_branch}
    #git submodule update --remote --recursive
else
	echo "Not found, cloning."
	git clone --branch ${SA_PCB_branch} ${GIT}/${GIT_USER}/SA_PCB --recurse-submodules
	cd SA_PCB
fi 	
make -j8
#cp -v ./build/sa_placer_test ../bin/sa_placer
cp -v ./build/sa_placer_test ../sa
cd ..

echo -n "Building pcbRouter binary. Checking for repository ... "
if [ -d "pcbRouter" ]; then
	echo "Found, cleaning"
	cd pcbRouter
	make clean
    git pull origin ${pcbRouter_branch}
    #git submodule update --remote --recursive
else
	echo "Not found, cloning."
	git clone --branch ${pcbRouter_branch} ${GIT}/${GIT_USER}/pcbRouter --recurse-submodules
	cd pcbRouter
fi 	
make -j8
cp -v build/pcbRouter_test ../pcb_router
cd ..

cd .. 
