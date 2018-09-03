#!/usr/bin/env bash
touch /mpi_is_running
/home/ec2-user/anaconda3/envs/python2/bin/python train.py --epochs 12 --learning-rate 0.0001 --training /opt/ml/input/data/training
EXIT_CODE=$?
touch /mpi_is_finished
exit ${EXIT_CODE}
