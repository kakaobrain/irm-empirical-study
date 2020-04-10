#!/usr/bin/env bash

#
# Scripts for reproducing PunctuatedSST-2 experiments in the paper.
#

# Create environments
python make_environments.py --datadir GLUE/SST-2 --destdir data/PunctuatedSST-2 --version default
python make_environments.py --datadir GLUE/SST-2 --destdir data/GrayscaleSST-2 --version grayscale

# Default ERM with grayscale data (with hyperparameter search)
python main.py -v --datadir data/GrayscaleSST-2 --penalty_anneal_iters 0 --penalty_weight 0.0 --lr 0.01 --l2_regularizer_weight 0.0005

# Default ERM (with hyperparameter search)
python main.py -v --penalty_anneal_iters 0 --penalty_weight 0.0 --lr 0.01 --l2_regularizer_weight 0.01

# Default IRM (with hyperparameter search)
python main.py -v --lr 0.001 --l2_regularizer_weight 0.0001 --penalty_weight 7500
