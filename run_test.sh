#!/bin/bash

schemes=("Sqrt2_17" "BitLog2_Single_17" "BitLog2_Half_16" "BitLog2_Half_17")
models=("deit_tiny" "deit_small" "deit_base" "vit_small" "vit_base") #"swin_small" "swin_base" "swin_tiny")

for scheme in "${schemes[@]}"
do
    for model in "${models[@]}"
    do
        log_file="logs/${scheme}/${model}.log"
        echo "Running experiment with model: $model"
        python -u test_quant.py --model $model --log_quant_scheme $scheme | tee $log_file
        echo "Logs saved to $log_file"
        echo ""
        echo ""
    done
done