#!/bin/bash

# 循环p_bias从6.75到6.95，步长为0.02
for p_bias in $(seq 6.75 0.02 6.95)
do
  python process_realdata.py --model heightposer_local_h --p_bias $p_bias

  python evaluate_pose.py --model heightposer_local_h

  echo "Finished for p_bias=$p_bias"
  echo "-----------------------------------------"
done

