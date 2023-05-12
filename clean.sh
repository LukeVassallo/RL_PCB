#!/bin/bash

if [ -d "${RL_PCB}/venv" ]; then
	echo "Cleaning venv"
	rm -fr venv
fi

if [ -d "${RL_PCB}/bin" ]; then
	echo "Cleaning bin"
	rm -fr bin
fi
