#!/usr/bin/env bash

#
# Scripts for reproducing the ColoredMNIST experiments in the paper.
#

################################################################################

# Default ERM
python main.py -v --penalty_anneal_iters 0 --penalty_weight 0.0

# Default IRM
python main.py -v

# IRM with hyperparameter search (official code ver.)
python main.py -v --hidden_dim 390 --l2_regularizer_weight 0.00110794568 --lr 0.0004898536566546834 --penalty_anneal_iters 190 --penalty_weight 91257.18613115903

################################################################################

# Q1 (plot)
python make_plots.py --experiment q1


# Q2
## 75% oracle
python main.py -v --train_probs 0.45 0.35 --train_label_probs 0.25 --penalty_anneal_iters 0 --penalty_weight 0.0  # ERM
python main.py -v --train_probs 0.45 0.35 --train_label_probs 0.25  # IRM
python main.py -v --train_probs 0.45 0.35 --train_label_probs 0.25 --penalty_anneal_iters 0 --penalty_weight 0.0 --grayscale_model # ERM-grayscale
## 100% oracle
python main.py -v --train_probs 0.2 0.1 --train_label_probs 0.0 --penalty_anneal_iters 0 --penalty_weight 0.0  # ERM
python main.py -v --train_probs 0.2 0.1 --train_label_probs 0.0  # IRM
python main.py -v --train_probs 0.2 0.1 --train_label_probs 0.0 --steps 10001  # IRM until convergence
python main.py -v --train_probs 0.2 0.1 --train_label_probs 0.0 --penalty_anneal_iters 0 --penalty_weight 0.0 --grayscale_model # ERM-grayscale


# Q3 (plot)
python make_plots.py --experiment q3


# Q4
python main.py -v --n_envs 2 --train_probs 0.3 0.1 --penalty_anneal_iters 0 --penalty_weight 0.0
python main.py -v --n_envs 2 --train_probs 0.3 0.1
python main.py -v --n_envs 3 --train_probs 0.3 0.1
python main.py -v --n_envs 5 --train_probs 0.3 0.1
python main.py -v --n_envs 5 --train_probs 0.3 0.2 0.19 0.12 0.1
python main.py -v --n_envs 10 --train_probs 0.3 0.1

# Q5
python main.py -v --n_classes 5 --steps 5001 --penalty_anneal_iters 0 --penalty_weight 0.0
python main.py -v --n_classes 5
python main.py -v --n_classes 10 --steps 5001 --penalty_anneal_iters 0 --penalty_weight 0.0
python main.py -v --n_classes 10 --steps 1001
