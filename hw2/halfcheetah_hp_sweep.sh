#!/bin/bash

BS=(10000 30000 50000)
LR=(.005 .01 .02)

for bs in ${BS[@]} ; do
    for lr in ${LR[@]} ; do
      python cs285/scripts/run_hw2.py --env_name HalfCheetah-v2 --ep_len 150 \
        --discount 0.95 -n 100 -l 2 -s 32 -b ${bs} -lr ${lr} -rtg --nn_baseline \
        --exp_name q4_search_b${bs}_lr${lr}_rtg_nnbaseline
    done
done
