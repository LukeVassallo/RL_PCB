#!/bin/bash

echo "Cleaning ... "

echo -n "Attempting to clean the work directory ... "
if [ -d "work" ]; then
    echo "Found, deleting."
    rm -fr work 
else
    echo "Not found, therefore nothing to clean."
fi

echo -n "Attempting to clean the tmp directory ... "
if [ -d "tmp" ]; then
    echo "Found, deleting."
    rm -fr tmp 
else
    echo "Not found, therefore nothing to clean."
fi

echo -n "Attempting to clean the report_config.json ... "
if [ -f "report_config.json" ]; then
    echo "Found, deleting."
    rm -fr report_config.json 
else
    echo "Not found, therefore nothing to clean."
fi

echo -n "Attempting to clean the experiment_report.pdf ... "
if [ -f "experiment_report.pdf" ]; then
    echo "Found, deleting."
    rm -fr experiment_report.pdf 
else
    echo "Not found, therefore nothing to clean."
fi

echo -n "Attempting to clean the scheduler.log ... "
if [ -f "scheduler.log" ]; then
    echo "Found, deleting."
    rm -fr scheduler.log 
else
    echo "Not found, therefore nothing to clean."
fi
