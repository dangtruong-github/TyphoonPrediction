#!/bin/bash

# Function to check if an element exists in an array
contains_element () {
    local e match="$1"
    shift
    for e; do [[ "$e" == "$match" ]] && return 0; done
    return 1
}

# Declare an array to store the arguments
declare -a args=("$@")

# List of valid commands
valid_commands=("preprocess" "train" "eval")

# Iterate over the arguments and run the corresponding scripts
for arg in "${args[@]}"; do
    if contains_element "$arg" "${valid_commands[@]}"; then
        case $arg in
        preprocess)
            ./bash_run/preprocess.sh
            ;;
        train)
            ./bash_run/train.sh
            ;;
        eval)
            ./eval.sh
            ;;
        esac
    else
        echo "Invalid argument: $arg"
        echo "Usage: ./run.sh [preprocess] [train] [eval]"
        exit 1
    fi
done
