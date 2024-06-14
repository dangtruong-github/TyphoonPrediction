#!/bin/bash

cd /N/u/tnn3/BigRed200/truongchu

# Default values for arguments
DATA_PATH=""
EVAL_FOLDER=""
MODEL_PATH=""

# Function to display help message
display_help() {
    echo "Usage: $0 -data /data/path -eval /eval/folder/path -model /model/path"
    exit 1
}

# Parse command line arguments
while [[ "$#" -gt 0 ]]; do
    case $1 in
        -data) DATA_PATH="$2"; shift ;;
        -eval) EVAL_FOLDER="$2"; shift ;;
        -model) MODEL_PATH="$2"; shift ;;
        *) echo "Unknown parameter passed: $1"; display_help ;;
    esac
    shift
done

# Convert paths to absolute paths based on the current working directory
if [ -n "$DATA_PATH" ]; then
    DATA_PATH=$(realpath "$DATA_PATH")
fi
if [ -n "$EVAL_FOLDER" ]; then
    EVAL_FOLDER=$(realpath "$EVAL_FOLDER")
fi
if [ -n "$MODEL_PATH" ]; then
    MODEL_PATH=$(realpath "$MODEL_PATH")
fi

# Check if some but not all arguments are provided
if [ -n "$DATA_PATH" ] || [ -n "$EVAL_FOLDER" ] || [ -n "$MODEL_PATH" ]; then
    if [ -z "$DATA_PATH" ] || [ -z "$EVAL_FOLDER" ] || [ -z "$MODEL_PATH" ]; then
        echo "Error: All three arguments -data, -eval, and -model must be provided."
        display_help
    fi
fi

# Run the Python script with the provided arguments, or without arguments if none are provided
if [ -n "$DATA_PATH" ] && [ -n "$EVAL_FOLDER" ] && [ -n "$MODEL_PATH" ]; then
    python3 train/eval.py -data "$DATA_PATH" -eval "$EVAL_FOLDER" -model "$MODEL_PATH"
else
    python3 train/eval.py
fi
