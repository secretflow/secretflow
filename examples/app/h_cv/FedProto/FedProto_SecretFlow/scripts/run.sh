#!/bin/bash
# bash ./scripts/baseline.sh
#echo script name: $0

python exps/federated_main.py --mode task_heter --dataset mnist --num_classes 10 --num_users 20 --ways 5 --stdev 2 --rounds 200

python exps/federated_main.py --mode model_heter --dataset mnist --num_classes 10 --num_users 20 --ways 5 --stdev 2 --rounds 200

python exps/federated_main.py --mode task_heter --dataset cifar10 --num_classes 10 --rounds 30 --train_ep 8 --ways 4 --stdev 1