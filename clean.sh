#!/bin/bash

if [ -d "${RL_PCB}/venv" ]; then
	echo "Cleaning ${RL_PCB}/venv"
	rm -fr ${RL_PCB}/venv
fi

if [ -d "${RL_PCB}/bin" ]; then
	echo "Cleaning ${RL_PCB}/bin"
	rm -fr ${RL_PCB}/bin
fi
