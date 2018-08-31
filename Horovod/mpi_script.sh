#!/usr/bin/env bash
touch /mpi_is_running
/home/ec2-user/anaconda3/envs/python2/bin/python train.py --epochs 20 --learning_rate 0.01
EXIT_CODE=$?
touch /mpi_is_finished
exit ${EXIT_CODE}
