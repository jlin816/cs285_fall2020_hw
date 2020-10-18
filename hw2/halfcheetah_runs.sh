#!/bin/bash

python cs285/scripts/run_hw2.py --env_name HalfCheetah-v2 --ep_len 150 \
--discount 0.95 -n 100 -l 2 -s 32 -b 50000 -lr .01 \
--exp_name q4_b50000_r.01 &

python cs285/scripts/run_hw2.py --env_name HalfCheetah-v2 --ep_len 150 \
--discount 0.95 -n 100 -l 2 -s 32 -b 50000 -lr .01 -rtg \
--exp_name q4_b50000_r.01_rtg &

python cs285/scripts/run_hw2.py --env_name HalfCheetah-v2 --ep_len 150 \
--discount 0.95 -n 100 -l 2 -s 32 -b 50000 -lr .01 --nn_baseline \
--exp_name q4_b50000_r.01_nnbaseline &

python cs285/scripts/run_hw2.py --env_name HalfCheetah-v2 --ep_len 150 \
--discount 0.95 -n 100 -l 2 -s 32 -b 50000 -lr .01 -rtg --nn_baseline \
--exp_name q4_b50000_r.01_rtg_nnbaseline &
