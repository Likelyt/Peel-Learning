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

# Pls add the code directory here 
os.chdir('/home/li3551/stat/bioinformatics/code/')
sys.path.append('/home/li3551/stat/bioinformatics/code/')

from module_x import generate_x
from module_y import generate_y
from module_PL import Network_PL  #
from pathway import *
from module_info import get_info_x
from module_info import GetInfoRelation


# Create Parser
parser = argparse.ArgumentParser(description="generate_x.py")

parser.add_argument("-out_dir",
                    default=os.path.join(os.getcwd(), "data"),
                    help="directory for output files (will be created if it doesnt exist)")
parser.add_argument("-iteration",
                    type=int,
                    default=50,
                    help="count iteration for set of parameters")


parser.add_argument("-task",
                    default="real-data",
                    help="real-data or simulation")
########################################################################
#### Simulation Part: that is the arguments you need to specify    ####
########################################################################
parser.add_argument("-structure",
                    default="Top-Down",
                    help="Tree direction: Top-Down or Bottom-Up")
parser.add_argument("-n",
                    type=int,
                    default=1158,
                    help='sample size: 200,500,1000')
parser.add_argument("-m",
                    type=int,
                    default=91,
                    help='predictor_size: 500,1000,5000')
parser.add_argument("-s",
                    type=int,
                    default=2,
                    help="tree division number: 2,5,10")
parser.add_argument("-gamma",
                    type=float,
                    default=0.5,
                    help='parameter of random effect: 0.5,0.7,0.9')
parser.add_argument("-objective",
                    type=str,
                    default='classification',
                    help="Task: regression or classification")
parser.add_argument("-eval_metric",
                    type=str,
                    default='cross_entropy',
                    help="loss function: mse for regression, cross_entropy for classification")
parser.add_argument("-xx_relation",
                    default="Linear",
                    help='x-x generation relation: "Linear" or "Sin"')

########################################################################
####                         Optimization options                   ####
########################################################################

parser.add_argument("-end_epoch",
                    type=int,
                    default=1,
                    help="Epoch to stop training")
parser.add_argument('-lr', 
                    type=float, 
                    default=1, 
                    help="learning rate")
parser.add_argument("-layers",
                    type=int,
                    default=7,
                    help='hierarchy layers of x')
parser.add_argument('-balance_data',
                    type=str,
                    default='balance',
                    help="data is balance or not")
parser.add_argument('-balance_weight',
                    type=float,
                    default=0.25,
                    help="data is balance or not")
parser.add_argument('-format_type',
                    type=str,
                    default='residual',
                    help='residual version or original')
parser.add_argument("-coefficient_share",
                    type=str,
                    default='True',
                    help='whether use training estimated coefficients to share with testing data')
parser.add_argument('-optimizer',
                    type=str,
                    default='SGD',
                    help='optimizer is SDG or Adam')
parser.add_argument('-momentum',
                    type=float,
                    default=0.9,
                    help='momentum value')
parser.add_argument('-beta_1', 
                    type=float, 
                    default=0.9, 
                    help='beta_1 value')
parser.add_argument('-beta_2', 
                    type=float, 
                    default=0.999, 
                    help='beta_2 value')
parser.add_argument('-eps', 
                    type=float, 
                    default=1e-8, 
                    help='epsilon value')
parser.add_argument('-decay_rate',
                    type=float,
                    default=0.975,
                    help='decay rate for learning rate')

########################################################################
#### Real Data Part: Pls Specify the data filepath and na,e    ####
########################################################################
parser.add_argument('-data_x_path',
                    type=str,
                    default='/home/li3551/stat/bioinformatics/data/data1/x_brca.txt',
                    help='feature matrix, p * n')
parser.add_argument('-data_y_path',
                    type=str,
                    default='/home/li3551/stat/bioinformatics/data/data1/y_brca.txt',
                    help='responce vector')
parser.add_argument('-adj_matrix_path',
                    type=str,
                    default='/home/li3551/stat/bioinformatics/data/data1/adj_brca.txt',
                    help='adjacent matrix')                    



########################################################################
####                     Evaluation Criteria                        ####
########################################################################

parser.add_argument('-criteria',
                    type=str,
                    default='C_E',
                    help='classification: AUC, ACC, C_E; regression: MSE')

########################################################################
####                     Random Seed Setup                          ####
########################################################################
parser.add_argument("-np_random_seed",
                    type=int,
                    default=12345,
                    help="training_ratio")

########################################################################
####                   Print Out the Options                        ####
########################################################################
opt = parser.parse_args()
print(opt)

if __name__ == '__main__':    
    # if simulation, that is the configuration
    if opt.task == 'simulation':
        xx_relation = opt.xx_relation
        gamma = opt.gamma
        s = opt.s
        m = opt.m
        n = opt.n
        structure_type = opt.structure
    # if real data, that is the configuation
    elif opt.task == 'real-data':
        x_path =  opt.data_x_path
        y_path =  opt.data_y_path
        adjacent_path = opt.adj_matrix_path

        x = np.loadtxt(x_path)    
        y = np.loadtxt(y_path)
        adj = np.loadtxt(adjacent_path)
        print('The data format is p x n; The real data p: %d and n is: %d' % (x.shape[0], x.shape[1]))

        m = x.shape[0] # feature numbers, 91
        n = x.shape[1] # sample size, 1158

        # Extract the Adjacent matrix information
        info_relation = GetInfoRelation(adj)
        parent_child = info_relation.parent_child_relation(adj)
        child_parent = info_relation.child_parent_relation(adj)

        # Pre-Calculate each layer neuron numbers, node index
        layer_neuron_number, list_of_node = get_info_x(adj, opt.layers)
        layer_neuron_number = layer_neuron_number.tolist()

    # optimization configurations
    lr = opt.lr
    end_epoch = opt.end_epoch
    iteration = opt.iteration

    # training settings
    mini_batch_size = 50
    n_train, n_val, n_test = int(7 * n / 10), int(1 * n / 20), int(n / 4)
    val_mini_batch_size = int(n / 20)
    
    # PL algorithm residual format
    format_type = opt.format_type # residual type
    eps = opt.eps

    # regression or classification task
    if opt.objective == 'classification':
        eval_metric = opt.eval_metric
        criteria = opt.criteria
    elif opt.objective == 'regression':
        eval_metric = opt.eval_metric
        criteria = opt.criteria

    # optimizer selection: SGD or Adam
    if opt.optimizer == 'SGD':
        beta_1 = opt.beta_1
        beta_2 = opt.beta_2
        momentum = opt.momentum
    elif opt.optimizer == 'Adam':
        beta_1 = opt.beta_1
        beta_2 = opt.beta_2
        momentum = opt.momentum

    # Result Saving list
    save_result = []
    save_scores = []    
    
    # Split data x
    train_x = x[:, range(0, n - n_val - n_test)]
    validation_x = x[:, range(n - n_val - n_test, n - n_test)]
    test_x = x[:, range(n - n_test, n)]

    # Split data y
    y = np.reshape(y, (1, len(y)))
    train_y = y[:, range(0, n - n_val - n_test)]
    validation_y = y[:, range(n - n_val - n_test, n - n_test)]
    test_y = y[:, range(n - n_test, n)]
    
    # Reformat training, validation, test
    training_data = list(zip(np.transpose(train_x), np.transpose(train_y)))
    validation_data = list(
        zip(np.transpose(validation_x), np.transpose(validation_y)))
    test_data = list(zip(np.transpose(test_x), np.transpose(test_y)))

    # Configuration PL Algorithm
    net_PL = Network_PL(layer_neuron_number,
                         adj,
                         parent_child,
                         child_parent,
                         list_of_node,
                         format_type,
                         opt.optimizer,
                         momentum,
                         beta_1,
                         beta_2,
                         eps,
                         criteria,
                         opt.objective,
                         eval_metric,
                         coefficient_share='False')
    for it in range(iteration):
        if opt.task == 'simulation':
            # Generate the adjacent matrix, based on the simulation settings
            adj, L, layer_nodes, child_parent, parent_child = sim_setup.structure_gen(
                s, m, structure_type)
            # Transform the adjacent list to each layers list of node index
            layer_neuron_number, list_of_node, delete_nodes = sim_setup.layer_of_node(
                parent_child, child_parent, m, L)
            print("layer_neuron_number", layer_neuron_number)

            # Generate X and Y in simulations
            x, y, p, pick_nodes = sim_setup.xy_gen_bi(n, adj, child_parent,
                                                    parent_child, s, m, L,
                                                    layer_nodes, structure_type,
                                                    xx_relation, gamma)
        elif opt.task == 'real-data':
            # Train PL
            loss_validation = net_PL.train(training_data, validation_data,
                                            n_train, n_val, end_epoch,
                                            mini_batch_size, val_mini_batch_size,
                                            lr)

            print('epoch: %d/%d; loss_validation: %.3f' %(it+1, iteration, loss_validation[0]))

            # Test
            optimal_epoch_given = False
            true_test_y, test_y_hat, auc, acc, c_e = net_PL.test_main(
                test_data,
                loss_validation,
                layer_neuron_number,
                optimal_epoch_given,
                LOAD_CHECKPOINT=True)

            print('auc', auc, 'acc', acc, 'c_e', c_e)
            it_scores = np.array([auc, acc, c_e])
            save_scores.append(it_scores)
            it_result = np.array([true_test_y, test_y_hat])
            save_result.append(it_result)

    print('save_result', np.array(save_result))

    save_result_path = 'yhat_n%d_m%d_s%d_gamma%r_xx%s_iter%d.npy' % (n, m, s, gamma, xx_relation, opt.iteration)
    np.save(os.path.join(opt.out_dir, save_result_path), np.array(save_result))  # need to add para to the name
    
    print('save_scores', np.array(save_scores))
    scores_path = 'scores_n%d_m%d_s%d_gamma%r_xx%s_iter%d.csv' % (n, m, s, gamma, xx_relation, opt.iteration)
    np.savetxt(os.path.join(opt.out_dir, scores_path), np.array(save_scores),delimiter=',')
