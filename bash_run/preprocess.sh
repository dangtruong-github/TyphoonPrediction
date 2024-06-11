#!/bin/bash

cd /N/u/tnn3/BigRed200/truongchu
rm -rf data_total
mkdir data_total
echo "Running Python code:"
python data_preprocess/pipeline.py
echo "Finish preprocess"

