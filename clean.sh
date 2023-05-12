#!/bin/bash

if [ -z ${RL_PCB} ]; then
	source setup.sh
fi

DIR=${RL_PCB}/venv
echo -n "Attempting to clean ${DIR} ... "
if [ -d "${DIR}" ]; then
	echo "Found, deleting."
	rm -fr ${DIR} 
else
	echo "Not found, therefore nothing to clean."
fi

DIR=${RL_PCB}/bin
echo -n "Attempting to clean ${DIR} ... "
if [ -d "${DIR}" ]; then
	echo "Found, deleting."
	rm -fr ${DIR} 
else
	echo "Not found, therefore nothing to clean."
fi
