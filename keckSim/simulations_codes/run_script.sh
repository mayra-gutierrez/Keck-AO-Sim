#!/bin/bash

# Path to your Python script
SCRIPT="KAO_statistics_loop.py"

# 1?? Neither M1 nor NCPA
echo "Running: frequency vs magnitude"
python $SCRIPT
pkill -f "$SCRIPT"
sleep 2
echo "All runs completed."
