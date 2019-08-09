import os
import sys
import time
import math
import copy
import random
import torch
import numpy as np
import pandas as pd
import argparse
from sklearn import linear_model

from module_x import generate_x
from module_y import generate_y
from module_load import load_data_wrapper
from module_time import time_since
from module_sdl import Network_SDL


# Create Parser
parser = argparse.ArgumentParser(description="generate_x.py")


# Simulation Data options
parser.add_argument("-n", type=int, default=200, help='sample size')
parser.add_argument("-m", type=int, default=1000, help='variable size')
parser.add_argument("-structure", default='Top-Down', help="Tree direction: Top-Down or Bottom-Up") 
parser.add_argument("-gamma", type=float, default=0.2, help="x-x generation relation additional random effect parameter")
parser.add_argument("-relation", default='Sin', help='x-x generation relation: Sin or Linear') 
parser.add_argument("-s", type=int, default=5, help='tree division number: 2, 5, 10')  
parser.add_argument("-layers", type=int, default=5, help='hierarchy layers of x : 10, 5, 3')
parser.add_argument('-nonzero_ratio', type=int, default=0.01, help='number of non-zero predictors') 
parser.add_argument('-residual_set', type=bool, default=True, help='whether use residual') # True - residual or False-N

# Optimization options
parser.add_argument("-end_epoch", type=int, default=50, help="Total Epoch")
parser.add_argument('-lr', type=float, default=0.1, help="learning rate")
parser.add_argument("-train_ratio", type=float, default=0.8, help="train_set_ratio")
parser.add_argument("-val_ratio", type=float, default=0.1, help="val_set_ratio")
parser.add_argument("-test_ratio", type=float, default=0.1, help="test_ratio")

# Random
parser.add_argument("-np_random_seed", type=int, default=12345, help="random seed")

opt = parser.parse_args()
print(opt)

if __name__ == '__main__':
    np.random.seed(opt.np_random_seed)

    # Working directory
    dirpath = os.getcwd()
    print("Current Directory is: " + dirpath)
    os.chdir(dirpath)
    sys.path.append(dirpath)


    sample_size = opt.n
    predictor_size = opt.m
    structure_type = opt.structure
    random_effect = opt.gamma
    x_x_relation = opt.relation
    folds = opt.s
    hierarchy_layers = opt.layers
    nonzero_ratio = opt.nonzero_ratio
    residual_set = opt.residual_set
    lr = opt.lr
    end_epoch = opt.end_epoch
    mini_batch_size = int(opt.n * opt.train_ratio)
    n_train, n_val, n_test = int(opt.n * opt.train_ratio), \
                             int(opt.n * opt.val_ratio), \
                             int(opt.n * opt.test_ratio)
    val_mini_batch_size = int(opt.n * opt.val_ratio)

    # Generate X and corresponding adjacent matrix
    x, epsilon, adjacent_matrix, child_parent, parent_child, layer_predictor_number, filename_x = \
        generate_x(predictor_size, sample_size, structure_type, random_effect, x_x_relation, folds, hierarchy_layers)

    # Generate Y
    y, sample, dataname_y = generate_y(x, epsilon, nonzero_ratio,
                                       structure_type, x_x_relation, folds, hierarchy_layers)
    # print(sample)

    # Load data
    training_data, validation_data, test_data = load_data_wrapper(filename_x, dataname_y, n_train, n_val, n_test)

    # Create SDL object
    net_SDL = Network_SDL(layer_predictor_number, adjacent_matrix, parent_child, child_parent, residual_set)

    # Train SDL
    loss_validation = net_SDL.SGD(training_data, validation_data, test_data,
                                  n_train, n_val, n_test,
                                  end_epoch,
                                  mini_batch_size, val_mini_batch_size, lr)

    # Test
    optimal_epoch_given = False
    test_loss = net_SDL.test_main(test_data, loss_validation, layer_predictor_number, optimal_epoch_given, LOAD_CHECKPOINT=True)
