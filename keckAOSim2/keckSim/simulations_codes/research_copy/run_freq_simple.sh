#!/bin/bash

# Path to your Python script
SCRIPT="stat_alpha.py"

echo "Running:--Atm only"
python $SCRIPT 
pkill -f "$SCRIPT"
sleep 2

echo "Running:--M1 only"
python $SCRIPT --M1
pkill -f "$SCRIPT"
sleep 2

# 2?? M1+ NCPA
echo "Running:--M1 + NCPA"
python $SCRIPT --M1 --NCPA
pkill -f "$SCRIPT"
sleep 2
echo "All runs completed."
