#!/bin/bash

# Path to your Python script

SCRIPT="SH_ZWFS_stat.py"

# 1?? M1 only
echo "Running: atm only"
python $SCRIPT --M1 --offset 40
pkill -f "$SCRIPT"
sleep 2


echo "All runs completed."

