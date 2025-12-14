#!/bin/bash

# Path to your Python script
SCRIPT="SH_ZWFS_stat.py"

# 1?? Neither M1 nor NCPA
echo "Running: no M1, no NCPA"
python $SCRIPT
pkill -f "$SCRIPT"
sleep 2

# 2?? M1 only
echo "Running: M1 only"
python $SCRIPT --M1
pkill -f "$SCRIPT"
sleep 2

# 3?? M1 and NCPA
echo "Running: M1 and NCPA"
python $SCRIPT --M1 --NCPA
pkill -f "$SCRIPT"
sleep 2

echo "All runs completed."
