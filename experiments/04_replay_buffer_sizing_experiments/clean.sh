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

echo -n "Attempting to clean the scheduler_a.log ... "
if [ -f "scheduler_a.log" ]; then
    echo "Found, deleting."
    rm -fr scheduler_a.log 
else
    echo "Not found, therefore nothing to clean."
fi

echo -n "Attempting to clean the scheduler_b.log ... "
if [ -f "scheduler_b.log" ]; then
    echo "Found, deleting."
    rm -fr scheduler_b.log 
else
    echo "Not found, therefore nothing to clean."
fi
