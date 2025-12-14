#!/bin/bash

# Path to your Python script
SCRIPT="KAO_statistics_studies.py"

# 1?? Neither M1 nor NCPA
echo "Running: Alpha study"
python $SCRIPT
pkill -f "$SCRIPT"
sleep 2
echo "All runs completed."
