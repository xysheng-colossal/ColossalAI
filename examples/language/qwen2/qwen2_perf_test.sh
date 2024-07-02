#!/bin/bash

################
#Load your environments and modules here
################

hostname -i > hosts.txt
HOSTFILE=$(realpath hosts.txt)

export OMP_NUM_THREADS=8

##geminie:zero3+batch size=16, OOM
#colossalai run --nproc_per_node 8 --hostfile $HOSTFILE benchmark.py -g -x -b 16
##geminie:zero3
#colossalai run --nproc_per_node 8 --hostfile $HOSTFILE benchmark.py -g -x -b 4
##geminie:zero3 + offload
#colossalai run --nproc_per_node 8 --hostfile $HOSTFILE benchmark.py -p gemini_auto -g -x -b 4
##hybird: zero2 + no grad ckpt, OOM
#colossalai run --nproc_per_node 8 --hostfile $HOSTFILE benchmark.py -p "3d" -x --zero 2 -b 1

#hybird: zero2+flash_atten+grad_ckpt+bs4
colossalai run --nproc_per_node 8 --hostfile $HOSTFILE benchmark.py -p "3d" -x -g --zero 2 -b 4
