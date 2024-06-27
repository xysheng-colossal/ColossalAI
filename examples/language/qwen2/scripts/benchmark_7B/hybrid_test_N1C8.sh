#!/bin/bash

################
#Load your environments and modules here
################

hostname -i > hosts.txt
HOSTFILE=$(realpath hosts.txt)

cd ../..
export OMP_NUM_THREADS=8

colossalai run --nproc_per_node 8 --hostfile $HOSTFILE benchmark.py -p "3d" -g -x --zero 2 -b 4 -s 12
