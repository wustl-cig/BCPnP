#!/bin/bash

if [ $# -eq 0 ];then
  echo "no argument or the argument"

else

export CUDA_VISIBLE_DEVICES=$1

python main.py config/config_pmri.yaml

fi
