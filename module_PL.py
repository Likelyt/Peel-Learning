import os
import sys
#from ggplot import *
import matplotlib.pyplot as plt

#os.chdir('/home/li3551/stat/bioinformatics/code/')
#sys.path.append('/home/li3551/stat/bioinformatics/code/')

from module_time import time_since

import time
import math
import random
import torch
import numpy as np
import pandas as pd
import argparse
from sklearn import linear_model
from sklearn.metrics import roc_auc_score
from sklearn.metrics import average_precision_score
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import roc_curve, auc, roc_auc_score
from sklearn.metrics import accuracy_score


class Network_PL(object):
    def __init__(self, sizes, adj_matrix, parent_child, child_parent,
                 list_of_node, 
                 format_type, optimizer, momentum, beta_1, beta_2, 
                 eps, criteria, objective, eval_metric,
                 coefficient_share):

        # layer_predictor_number/sizes: such like [7, 3, 1]
        # adj_matrix: adjacent matrix
        # parent_child: parent child dict
        # chile_parent: child parent dict
        # Format_type: residual or original
        # objective = 'regression' or 'classification'
        # eval_metric = 'mse' or 'cross_entropy' (for two classes)
        # selection_option = 'max' or 'min' for 'MSE' or 'AUC', 'ACC', or 'C_E'

        np.random.seed(12345)

        self.sizes = sizes
        self.sizes.append(1)
        # add the output yhat
        #print('self.sizes', self.sizes, '\n')

        self.num_layers = len(sizes)
        self.format_type = format_type
        self.parent_child = parent_child
        self.child_parent = child_parent
        self.optimizer = optimizer
        self.momentum = momentum
        self.objective = objective
        self.eval_metric = eval_metric
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.eps = eps
        self.criteria = criteria
        self.coefficient_share = coefficient_share

        if self.criteria == 'AUC' or self.criteria == 'ACC':
            self.aim = np.max
        elif self.criteria == 'MSE' or self.criteria == 'C_E':
            self.aim = np.min

        # biased: sizes[i]*1
        self.biases = [np.random.randn(i, 1) for i in self.sizes[1:]]

        # adjacent matrix
        self.adj_matrix = adj_matrix

        # calculate each layer which nodes will be left.
        #self.list_of_node = self.layer_node_number(parent_child, child_parent, sizes)

        #list_of_node += [[0,1]]
        self.list_of_node = list_of_node
        #print('self.list_of_node',type(list_of_node))

        # Initialization weight matrix
        self.weights_ur, self.weights = self.weight_matrix_init(
            adj_matrix, self.sizes, self.list_of_node)

        if self.optimizer == 'SGD':
            # v_t_prev initialization : weights
            self.v_t_prev = [np.zeros(w.shape) for w in self.weights]
            # u_t_prev initialization : biases
            self.u_t_prev = [np.zeros((i, 1)) for i in self.sizes[1:]]
        elif self.optimizer == 'Adam':
            # m_t_prev initialization: weights - first moment
            self.m_t_prev = [np.zeros(w.shape) for w in self.weights]
            # n_t_prev initialization: biases - first moment
            self.n_t_prev = [np.zeros((i, 1)) for i in self.sizes[1:]]
            # m_t_sq_prev initialization: weights - second moment
            self.m_t_sq_prev = [np.zeros(w.shape) for w in self.weights]
            # n_t_sq_prev initialization: biases - second moment
            self.n_t_sq_prev = [np.zeros((i, 1)) for i in self.sizes[1:]]

        # Calculate each layer's child and parent dictionary
        self.index_child, self.index_parent = self.pc_dict(
            self.weights_ur, self.sizes)

    def train(self, training_data, test_data, n_train, n_test, end_epoch,
              mini_batch_size, test_mini_batch_size, eta):
        #print('Enter train\n')
        # use stochastic gradient descent method to optimize the objective function

        # training_data: training data
        # epochs: learning epochs
        # mini_batch_size
        # eta: learning rate, lr
        # validation_data
        # test_data

        # loss dict
        loss_training = []
        loss_test = []
        self.eta = eta

        # temp_w = [np.zeros([y, x]) for x, y in zip(self.sizes[:-1], self.sizes[1:])]
        # temp_b = [np.zeros([y, 1]) for y in self.sizes[1:]]

        # calculate time
        start = time.time()

        group = n_train / mini_batch_size * end_epoch
        # random select training data into mini batch

        #nn = training_data[0].shape[0]
        for epoch in range(end_epoch):
            step = 0
            #np.random.seed(123)
            #temp = np.random.choice(nn, nn, replace=False)
            #training_data[0] = training_data[0][temp,]
            #training_data[1] = training_data[1][0,temp].reshape(1,-1) # to 1*n

            random.shuffle(training_data)

            # create mini_batches
            mini_batches = [
                list(training_data)[k:k + mini_batch_size]
                for k in range(0, n_train, mini_batch_size)
            ]
            '''
            # create mini_batches
            mini_batches = [[]
                            for _ in range(int(np.ceil(nn / mini_batch_size)))]
            for k in range(0, n_train, mini_batch_size):
                mini_batches[k] = [
                    training_data[0][k:k + mini_batch_size, ],
                    training_data[1][0, k:k + mini_batch_size]
                ]
                #print('\nmini_batches',k,len(mini_batches),'\n')
            '''
            # use mini_batch to calculate loss
            for mini_batch in mini_batches:
                step += 1
                # Organize data
                mini_batch = self.data_organize(mini_batch)

                self.weights, self.biases = self.update_weight(
                    mini_batch, eta, epoch)
                #print('self.biases',self.biases[1])
                #print('updated weights',self.weights,'\n')

                # Calculate loss
                if self.objective == 'classification':
                    if self.eval_metric == 'cross_entropy':
                        #print('mini_batch',mini_batch[0])
                        c_e, auc, acc, y_prob, print_prsa = self.evaluate(
                            mini_batch)
                elif self.objective == 'regression':
                    if self.eval_metric == 'mse':
                        mse, y_prob = self.evaluate(mini_batch)

                # Set the selection criteria
                if self.criteria == 'AUC':
                    loss_training.append(auc)
                elif self.criteria == 'ACC':
                    loss_training.append(acc)
                elif self.criteria == 'C_E':
                    loss_training.append(c_e)
                    #print('loss_training minibatch', loss_training, '\n')
                elif self.criteria == 'MSE':
                    loss_training.append(mse)

                # Print training loss
                if self.objective == 'classification':
                    if self.eval_metric == 'cross_entropy':
                        print_training = '\nTraining: %d / %d, Accumulate Time to End: %s, ' \
                                         '(Batch: %d / Batches num: %d, Percent Run: %.2f%%), '\
                                         'Cross Entropy: %.2f' % (epoch+1, end_epoch, time_since(start,(step+epoch*n_train/mini_batch_size)/group),
                                                                           step, n_train / mini_batch_size,
                                                                           (epoch * (n_train / mini_batch_size) + step)/group * 100, c_e)

                elif self.objective == 'regression':
                    if self.eval_metric == 'mse':
                        print_training = 'Epoch: %d / %d, Accumulate Time to End: %s, '\
                                         '(Batch: %d / Batches num: %d, Percent Run: %.2f%%), '\
                                         'Training MSE Loss: %.2f' % (epoch+1, end_epoch, time_since(start,(step+epoch*n_train/mini_batch_size)/group),
                                                                      step, n_train / mini_batch_size,
                                                                      (epoch * (n_train / mini_batch_size) + step)/group * 100,
                                                                      mse)

                #print(print_training)
                #print(print_prsa)
                #print('training y_prob', np.around(y_prob, 3))
                sys.stdout.flush()
                # print(self.weights[-2], self.biases[-2])

            ######################################################################
            # Use validation method to select the best parameters
            ######################################################################

            # Create mini_batches
            test_mini_batches = [
                list(test_data)[k:k + test_mini_batch_size]
                for k in range(0, n_test, test_mini_batch_size)
            ]

            # use mini_batch to calculate loss
            loss_test_all = 0
            loss_test_part = 0
            loss_test_average = 0

            for test_mini_batch in test_mini_batches:
                # transform data format_type
                test_mini_batch = self.data_organize(test_mini_batch)

                # Keep loss_validation
                if self.objective == 'classification':
                    if self.eval_metric == 'cross_entropy':
                        c_e, auc, acc, y_prob, print_prsa = self.evaluate(
                            test_mini_batch)
                elif self.objective == 'regression':
                    mse, prob = self.evaluate(test_mini_batch)

                #print('c_e in validation minibatch', c_e)
                loss_test_part = c_e
                loss_test_all += loss_test_part

            # average loss of val
            loss_test_average = loss_test_all / len(test_mini_batches)

            # Save validation_loss
            loss_test.append(loss_test_average)

            # Print validation loss
            if self.objective == 'classification':
                if self.eval_metric == 'cross_entropy':
                    print_test = '\nValidation: %d / %d, Cross_Entropy: %.3f' % (
                        epoch + 1, end_epoch, loss_test_average)
                elif self.eval_metric == 'error':
                    print_test = '\nValidation: %d / %d, Validation Miss Classification Rate : %.3f' % (
                        epoch + 1, end_epoch, loss_test_average)

            elif self.objective == 'regression':
                if self.eval_metric == 'mse':
                    print_test = '\nValidation: %d / %d, Validation MSE Loss: %.3f' % (
                        epoch + 1, end_epoch, loss_test_average)

            #print(print_test)
            #print(print_prsa)
            sys.stdout.flush()

            # check points
            # First create a model check point folder
            if not os.path.exists('model_checkpoint_train'):
                os.makedirs('model_checkpoint_train')

            checkpoint = {
                'epoch': epoch + 1,
                'learning_rate': eta,
                'w': self.weights,
                'biases': self.biases,
                'validation_loss': loss_test_average,
                'h_hat': self.h_hat,
                'b_intercept': self.b_intercept
            }

            # Need change the name of saving the model parameter name
            save_epoch = epoch + 1
            model_name = os.path.join('model_checkpoint_train/',
                                      "SDL_epoch_%d.pt" % save_epoch)

            # Save model parameters
            torch.save(checkpoint, model_name)
            print("Save model as %s \n" % model_name)
            '''
            # we can select the minimal of validation then use if for test result.
            ########################################################################################
            #                                        Test                                          #
            ########################################################################################

            optimal_epoch_given = loss_training.index(
                self.aim(loss_training)) + 1
            #optimal_epoch_given = epoch+1
            print('optimal_epoch_given', optimal_epoch_given)

            optimal_AUC, optimal_ACC = self.test_main(test_data,
                                                      loss_test,
                                                      self.sizes,
                                                      optimal_epoch_given,
                                                      pathway_num,
                                                      cv_num,
                                                      LOAD_CHECKPOINT=True)

            # convergence
            if epoch > 0:
                if abs(loss_training[-1] - loss_training[-2]) < 0.000001:
                    break
            '''

        return loss_test

    '''
    def data_organize(self, mini_batch):
        # transform data from [[x1,y1], [x2,y2]] to [[x1, x2], [y1, y2]]

        # mini_batch_size
        n = mini_batch[0].shape[0]

        # predictor size
        m = mini_batch[0].shape[1]

        x = mini_batch[0]
        y = mini_batch[1]
        x = np.reshape(x, (n, m))
        y = np.reshape(y, (n, 1))

        return [x, y]
    '''

    def data_organize(self, mini_batch):
        # transform data from [[x1,y1], [x2,y2]] to [[x1, x2], [y1, y2]]

        # mini_batch_size
        n = len(mini_batch)

        # predictor size
        m = mini_batch[0][0].shape[0]

        x = []
        y = []
        #print(mini_batch[0])
        for i in range(n):
            x.append(mini_batch[i][0])
            y.append(mini_batch[i][1])

        x = np.reshape(x, (n, m))
        y = np.reshape(y, (n, 1))
        return [x, y]

    def update_weight(self, mini_batch, eta, epoch):
        #print('Enter update_weight\n')
        # update data weights and bias

        # get X design matrix and response y vector
        x_matrix = mini_batch[0]
        y_vector = mini_batch[1]

        # get delta variable;  x: m*n,  y: 1*n

        #delta_nabla_b, delta_nabla_w, nabla_b_cov, nabla_w_cov = self.backprop(np.transpose(x_matrix), np.transpose(y_vector))
        delta_nabla_b, delta_nabla_w = self.backprop(np.transpose(x_matrix),
                                                     np.transpose(y_vector))

        # update weights and biases
        weights = [
            w - (eta * nw) for w, nw in zip(self.weights, delta_nabla_w)
        ]
        biases = [b - (eta * nb) for b, nb in zip(self.biases, delta_nabla_b)]

        return weights, biases
        '''
        # update weights and biases
       
        if self.optimizer == 'SGD':
            # Nesterov Momentum for SGD
            if self.momentum != 'null':
                # updates weights
                v_t_now = [
                    self.momentum * v_t_prev_layer +
                    np.power(self.decay_rate, epoch) * eta *
                    (nw - self.momentum * v_t_prev_layer)
                    for v_t_prev_layer, nw in zip(self.v_t_prev, delta_nabla_w)
                ]
                #v_t_now = [nw for nw in delta_nabla_w]
                #print('eta',np.power(self.decay_rate, epoch)*eta)
                weights = [
                    w - v_t_now_layer
                    for w, v_t_now_layer in zip(self.weights, v_t_now)
                ]
                #w_cov = self.w_cov - eta * nabla_w_cov

                # updates biases
                u_t_now = [
                    self.momentum * u_t_prev_layer +
                    np.power(self.decay_rate, epoch) * eta *
                    (nb - self.momentum * u_t_prev_layer)
                    for u_t_prev_layer, nb in zip(self.u_t_prev, delta_nabla_b)
                ]
                #u_t_now = [eta * nb for nb in delta_nabla_b]

                biases = [
                    b - u_t_now_layer
                    for b, u_t_now_layer in zip(self.biases, u_t_now)
                ]

                # let v_t = v_t-1
                self.v_t_prev = v_t_now
                # let u_t = u_t-1
                self.u_t_prev = u_t_now

        elif self.optimizer == 'Adam':
            if self.beta_1 != 0.0 and self.beta_2 != 0.0:
                m_t_now = [
                    self.beta_1 * m_t_prev_layer + (1 - self.beta_1) * nw
                    for m_t_prev_layer, nw in zip(self.m_t_prev, delta_nabla_w)
                ]
                m_t_sq_now = [
                    self.beta_1 * m_t_sq_prev_layer +
                    (1 - self.beta_1) * np.power(nw, 2)
                    for m_t_sq_prev_layer, nw in zip(self.m_t_sq_prev,
                                                     delta_nabla_w)
                ]

                n_t_now = [
                    self.beta_1 * n_t_prev_layer + (1 - self.beta_1) * nb
                    for n_t_prev_layer, nb in zip(self.n_t_prev, delta_nabla_b)
                ]
                n_t_sq_now = [
                    self.beta_1 * n_t_sq_prev_layer +
                    (1 - self.beta_1) * np.power(nb, 2)
                    for n_t_sq_prev_layer, nb in zip(self.n_t_sq_prev,
                                                     delta_nabla_b)
                ]

                m_t_now_hat = [
                    m_t_now_part / (1 - self.beta_1)
                    for m_t_now_part in m_t_now
                ]
                m_t_sq_now_hat = [
                    m_t_sq_now_part / (1 - self.beta_2)
                    for m_t_sq_now_part in m_t_sq_now
                ]

                n_t_now_hat = [
                    n_t_now_part / (1 - self.beta_1)
                    for n_t_now_part in n_t_now
                ]
                n_t_sq_now_hat = [
                    n_t_sq_now_part / (1 - self.beta_2)
                    for n_t_sq_now_part in n_t_sq_now
                ]

                weights = [
                    w - np.power(self.decay_rate, epoch) * eta *
                    m_t_now_hat_layer /
                    (np.sqrt(m_t_sq_now_hat_layer) + self.eps)
                    for w, m_t_now_hat_layer, m_t_sq_now_hat_layer in zip(
                        self.weights, m_t_now_hat, m_t_sq_now_hat)
                ]
                biases = [
                    b - np.power(self.decay_rate, epoch) * eta *
                    n_t_now_hat_layer /
                    (np.sqrt(n_t_sq_now_hat_layer) + self.eps)
                    for b, n_t_now_hat_layer, n_t_sq_now_hat_layer in zip(
                        self.biases, n_t_now_hat, n_t_sq_now_hat)
                ]

                self.m_t_prev = m_t_now
                self.n_t_prev = n_t_now
                self.m_t_sq_prev = m_t_sq_now
                self.n_t_sq_prev = n_t_sq_now
        print('weights[-1]', weights[-1])
        '''

    '''
    def weight_matrix_init(self, adj_matrix, sizes, list_of_node, scale=False):
            # Initialization weight matrix, created w_final
            # return w_final
    
            # Define fixed random effect
            np.random.seed(123)
    
            # how many weight matrix needed
            weight_mat_number = len(sizes) - 1
    
            # double sizes for [x, epsilon] use
            sizes_double = []
            for i in range(weight_mat_number + 1):
                sizes_double.append(sizes[i] * 2)
    
            # weights
            w_mat = [np.random.randn(y, x) for x, y in zip(sizes_double[:-1], sizes_double[1:])]
    
            # Scale weight parameters smaller or larger
            if scale == True:
                for i in range(weight_mat_number):
                    w_mat[i] = w_mat[i]/3
    
            # list_of_node from [0,1,2] change to [0,1,2,0,1,2], leave the last to 1 * 6 (1*3|1*3)
            for i in range(self.num_layers):
                print('i',i,list_of_node)
                list_of_node[i].extend(list_of_node[i])
    
            # keep each layer node name for W_hat
            list_of_node_name = [[] for _ in range(weight_mat_number + 1)]
            list_of_node_name[-1] = map(lambda i: 'x' + str(i), list_of_node[-1])
            for j in range(len(list_of_node)):
                list_of_node_name[j] = list(map(lambda i: 'x_' + str(i), list_of_node[j][:int(len(list_of_node[j]) / 2)]))
                list_of_node_name[j].extend(list(map(lambda i: 'epsilon_' + str(i),
                                                list_of_node[j][int(len(list_of_node[j]) / 2):])))
    
            # Give Adjacent column and row name,
            # adj_mat_NAME [      epsilon_0, epsilon_1, ...,
            #                 x_0,
            #                 x_1,
            #                 ...
            #              ]
            # and is DataFrame
            adj_mat_NAME = [[] for _ in range(weight_mat_number + 1)]
            adj_mat_NAME[0] = pd.DataFrame(adj_matrix, index=list_of_node_name[0][: int(len(list_of_node_name[0])/2)],
                                columns=list_of_node_name[0][int(len(list_of_node_name[0])/2):])
    
            # Create W_hat matrix for each layer
            W_hat = [np.zeros([y, x]) for x, y in zip(sizes_double[:-1], sizes_double[1:])]
            W_hat_copy = [np.zeros([y, x]) for x, y in zip(sizes_double[:-1], sizes_double[1:])]
    
            # Give W_hat matrix column and row name:
            for i in range(weight_mat_number):
                w_mat[i] = pd.DataFrame(w_mat[i], index=list_of_node_name[i + 1], columns=list_of_node_name[i])
                W_hat[i] = pd.DataFrame(W_hat[i], index=list_of_node_name[i + 1], columns=list_of_node_name[i])
                W_hat_copy[i] = pd.DataFrame(W_hat_copy[i], index=list_of_node_name[i + 1], columns=list_of_node_name[i])
    
            # Initialization W_hat matrix: left and right
            # Layer i
            for i in range(weight_mat_number):
                # Right half of W
                for j in list_of_node_name[i][int(len(list_of_node_name[i])/2):]:
                    # Upper half of W
                    for k in list_of_node_name[i + 1][:int(len(list_of_node_name[i + 1])/2)]:
                        if adj_mat_NAME[i][j][k] != 0:
                            W_hat[i][j][k] = w_mat[i][j][k]
                            W_hat_copy[i][j][k] = w_mat[i][j][k]
    
                # delete x_variable: row
                delete_number_of_row = list(set(list_of_node_name[i][:int(len(list_of_node_name[i])/2)]).symmetric_difference(
                    set(list_of_node_name[i+1][:int(len(list_of_node_name[i+1])/2)])))
                # delete epsilon_variable: column
                delete_number_of_col = list(set(list_of_node_name[i][int(len(list_of_node_name[i])/2):]).symmetric_difference(
                    set(list_of_node_name[i + 1][int(len(list_of_node_name[i + 1])/2):])))
                # Update adjacent matrix layer by layer
                adj_mat_NAME[i + 1] = adj_mat_NAME[i].drop(delete_number_of_row)  # row
                adj_mat_NAME[i + 1] = adj_mat_NAME[i + 1].drop(delete_number_of_col, axis=1)  # column
    
    
            # Then W_hat will be transformed from DataFrame to ndarray matrix: w_nda, W_HAT_copy ndarray
            w_nda = [np.zeros([y, x]) for x, y in zip(sizes_double[:-1], sizes_double[1:])]
            w_nda_u = [np.zeros([y, x]) for x, y in zip(sizes_double[:-1], sizes_double[1:])]
    
            for i in range(weight_mat_number):
                w_nda[i] = W_hat[i].values
                w_nda_u[i] = W_hat_copy[i].values
            
            #print('w_mat[-1]',w_mat[-1])
            # Delete the last row: [x_0, epsilon_0] -> [x_0]
            #w_mat[-1] = w_mat[-1].drop(['epsilon_0'], axis=0)
            print('w_mat[-1]',w_mat[-1])
    
            # w_nda, w_nda_u is the normal matrix
            # w_nda_u to become double matrix
            for i in range(len(sizes)-1):
                next_layer_size_by_two = w_nda_u[i].shape[0]
                # Assign upper, right half of w_mat to upper left of w_nda_u
                for j in range(int(next_layer_size_by_two/2)):
                    w_nda_u[i][j][j] = (w_mat[i].values)[j][j]
            
            print('w_nda_u[-1]',w_nda_u[-1])
            #w_nda_u[-1] = np.delete(w_nda_u[-1], obj=[1,3], axis=0)

            w_nda_u[-1] = np.delete(w_nda_u[-1], [e for e in range(int(len(w_nda_u[-1]) / 2), int(len(w_nda_u[-1])))], 0)
            print('w_nda_u[-1]',w_nda_u[-1])

            # w_nda_u right part of the W
            w_nda_ur = [np.zeros([y, x]) for x, y in zip(sizes[:-1], sizes[1:])]
            # layer i, upper + right
            for i in range(len(w_nda)):
                rown = np.shape(w_nda[i])[0]
                coln = np.shape(w_nda[i])[1]
                for j in range(0, int(rown/2)):
                    for k in range(int(coln/2), coln):
                        w_nda_ur[i][j][k - int(coln/2)] = w_nda[i][j][k]
    
            # Finally, w_final version: take upper level of the w_nda_u row (500*2000)
            w_final = [[] for _ in range(len(w_mat))]
            for i in range(weight_mat_number - 1):
                w_final[i] = w_nda_u[i][:int(len(w_nda_u[i])/2)]
                print('w_final[i]',i,w_final[i].shape,'\n')
            w_final[-1] = w_nda_u[-1]
            print('w_final[-1]',w_final[-1],'\n')
    
            # W_hat, W_hat_copy: Data_frame version
            # w_nda_ur: is the w_final [upper and right part]
            # w_final: is the final w
            return w_nda_ur, w_final
    '''

    def weight_matrix_init(self, adj_matrix, sizes, list_of_node, scale=False):
        #print('Enter weight_matrix_init\n')
        ### (YMY) Changed shape of matrix to match list_of_node and other details
        # Initialization weight matrix, created w_final
        # return w_final

        # Define fixed random effect
        np.random.seed(1234)

        # how many weight matrix needed
        weight_mat_number = len(sizes) - 1

        # double sizes for [x, epsilon] use
        sizes_double = []
        for i in range(weight_mat_number + 1):
            sizes_double.append(sizes[i] * 2)

        # weights
        w_mat = [
            np.random.randn(y, x)
            for x, y in zip(sizes_double[:-1], sizes_double[1:])
        ]
        #print('w_mat[0]', w_mat[0].shape, w_mat[0])
        #w_mat = [np.ones([y, x]) for x, y in zip(sizes_double[:-1], sizes_double[1:])]

        # Scale weight parameters smaller or larger
        if scale == True:
            for i in range(weight_mat_number):
                w_mat[i] = w_mat[i] / 10

        # list_of_node from [0,1,2] change to [0,1,2,0,1,2], leave the last to 1 * 6 (1*3|1*3)
        for i in range(self.num_layers - 1):
            list_of_node[i].extend(list_of_node[i])

        # keep each layer node name for W_hat
        list_of_node_name = [[] for _ in range(weight_mat_number + 1)]
        #print('len(list_of_node_name)', len(list_of_node_name), '\n')
        list_of_node_name[-1] = ['yhat_1', 'del']
        for j in range(len(list_of_node)):
            list_of_node_name[j] = list(
                map(lambda i: 'x_' + str(i), list_of_node[j][:sizes[j]]))
            list_of_node_name[j].extend(
                list(map(lambda i: 'epsilon_' + str(i),list_of_node[j][sizes[j]:])))

        # Give Adjacent column and row name,
        # adj_mat_NAME [      epsilon_0, epsilon_1, ...,
        #                 x_0,
        #                 x_1,
        #                 ...
        #              ]
        # and is DataFrame
        adj_mat_NAME = [[] for _ in range(weight_mat_number + 1)]
        adj_row_name = list(
            map(lambda i: 'x_' + str(i), range(len(adj_matrix))))
        adj_col_name = list(
            map(lambda i: 'epsilon_' + str(i), range(len(adj_matrix))))
        adj = pd.DataFrame(adj_matrix,
                           index=adj_row_name,
                           columns=adj_col_name)
        # data frame of the original adj_matrix with x_ and epsilon_ as names
        #print('adj',adj,'\n')

        # Create W_hat matrix for each layer
        W_hat = [
            np.zeros([y, x])
            for x, y in zip(sizes_double[:-1], sizes_double[1:])
        ]
        W_hat_copy = [
            np.zeros([y, x])
            for x, y in zip(sizes_double[:-1], sizes_double[1:])
        ]

        # Give W_hat matrix column and row name:
        for i in range(weight_mat_number):
            #print('w_mat i',i)
            #print('list_of_node_name[i + 1]',list_of_node_name[i + 1], '\n')
            w_mat[i] = pd.DataFrame(w_mat[i],
                                    index=list_of_node_name[i + 1],
                                    columns=list_of_node_name[i])
            W_hat[i] = pd.DataFrame(W_hat[i],
                                    index=list_of_node_name[i + 1],
                                    columns=list_of_node_name[i])
            W_hat_copy[i] = pd.DataFrame(W_hat_copy[i],
                                         index=list_of_node_name[i + 1],
                                         columns=list_of_node_name[i])

        # Initialization W_hat matrix: left and right
        # Last layer
        #print('w_mat[-1] in weight init',w_mat[-1],'\n')
        W_hat[-1] = w_mat[-1]
        W_hat_copy[-1] = w_mat[-1]
        #print('W_hat[-1]',W_hat[-1],'\n')
        # Layer i (not including output layer)
        for i in range(weight_mat_number - 1):
            node = list_of_node[i][:int(len(list_of_node[i]) / 2)]
            #node of layer i
            adj_mat_NAME[i] = pd.DataFrame(
                adj_matrix[node, :][:, node],
                index=list_of_node_name[i][:sizes[i]],
                columns=list_of_node_name[i][sizes[i]:])
            #print('adj_mat_NAME',i,adj_mat_NAME[i],'\n')
            # Right half of W
            #print('list_of_node_name[i][sizes[i]:]',
            #      list_of_node_name[i][sizes[i]:])
            for j in list_of_node_name[i][sizes[i]:]:
                # Upper half of W
                #print('j',j,'\n')
                for k in list_of_node_name[i + 1][:sizes[i + 1]]:
                    #print('list_of_node_name[i + 1]',list_of_node_name[i + 1],'\n')
                    #print('k',k,'\n')
                    if adj[j][k] != 0:
                        #print('adj[j][k] != 0',j, k, adj[j][k] != 0)
                        # to see if two nodes have connection
                        # check if [j,k] is 1 in the original adj
                        W_hat[i][j][k] = w_mat[i][j][k]
                        W_hat_copy[i][j][k] = w_mat[i][j][k]

        # Then W_hat will be transformed from DataFrame to ndarray matrix: w_nda, W_HAT_copy ndarray
        w_nda = [
            np.zeros([y, x])
            for x, y in zip(sizes_double[:-1], sizes_double[1:])
        ]
        w_nda_u = [
            np.zeros([y, x])
            for x, y in zip(sizes_double[:-1], sizes_double[1:])
        ]

        for i in range(weight_mat_number):
            w_nda[i] = W_hat[i].values
            #print('w_nda[i][1,:]',w_nda[i].shape,w_nda[i][1,:])
            w_nda_u[i] = W_hat_copy[i].values
        #print('w_nda[-1]',w_nda[-1],'\n')
        #print('w_nda_u[-1]',w_nda_u[-1],'\n')

        # Delete the last row: [x_0, epsilon_0] -> [x_0]
        #w_mat[-1] = w_mat[-1].drop(['epsilon_0'], axis=0)

        # w_nda, w_nda_u is the normal matrix
        # w_nda_u to become double matrix
        for i in range(len(sizes) - 1):
            next_layer_size_by_two = w_nda_u[i].shape[0]
            #print('w_nda_u[i].shape[0]',w_nda_u[i].shape[0])
            # Assign upper, right half of w_mat to upper left of w_nda_u
            for j in range(int(next_layer_size_by_two / 2)):
                #print('w_nda_u[i].shape[1]',w_nda_u[i].shape[1])
                #for k in range(int(w_nda_u[i].shape[1]/2)):
                w_nda_u[i][j][j] = (w_mat[i].values)[j][j]

        #w_nda_u[-1] = np.delete(w_nda_u[-1], obj=[1,3], axis=0)
        #print('w_nda[-1] after a loop',w_nda[-1],'\n')
        w_nda_u[-1] = np.delete(w_nda_u[-1], [
            e for e in range(int(len(w_nda_u[-1]) / 2), int(len(w_nda_u[-1])))
        ], 0)
        #print('w_nda_u[0]',w_nda_u[0])

        # w_nda_u right part of the W
        w_nda_ur = [np.zeros([y, x]) for x, y in zip(sizes[:-1], sizes[1:])]
        # layer i, upper + right
        for i in range(len(w_nda)):
            rown = np.shape(w_nda[i])[0]
            coln = np.shape(w_nda[i])[1]
            for j in range(0, int(rown / 2)):
                for k in range(int(coln / 2), coln):
                    w_nda_ur[i][j][k - int(coln / 2)] = w_nda[i][j][k]

        # Finally, w_final version: take upper level of the w_nda_u row (500*2000)
        w_final = [[] for _ in range(len(w_mat))]
        for i in range(weight_mat_number - 1):
            w_final[i] = w_nda_u[i][:int(len(w_nda_u[i]) / 2)]
            #print('w_final[i]', i, w_final[i], '\n')
        w_final[-1] = w_nda_u[-1]
        #print('w_final[-1]', w_final[-1], '\n')
        #print('w_nda_ur[-1]', w_nda_ur[-1],'\n')

        # W_hat, W_hat_copy: Data_frame version
        # w_nda_ur: is the w_final [upper and right part]
        # w_final: is the final w
        return w_nda_ur, w_final

    def pc_dict(self, weights_ur, sizes):
        # return each layer parent_child dict

        # Create each layer child, parent dictionary
        index_child = [{} for _ in range(len(weights_ur))]
        index_parent = [{} for _ in range(len(weights_ur))]

        # For layer: l
        for l in range(len(weights_ur)):
            # row_n: represents W row number =  l+1 layer nodes number
            # col_n: represents W column number/2 =  l layer nodes number
            row_n = int(np.shape(weights_ur[l])[0])
            col_n = int(np.shape(weights_ur[l])[1])

            for i in range(row_n):
                index_child[l][i] = list(np.where(weights_ur[l][i] != 0)[0])

            for j in range(col_n):
                index_parent[l][j] = list(
                    np.where(weights_ur[l][..., j] != 0)[0])

        # extra add - need fix: double make sure
        index_parent[-1][0] = []

        return index_child, index_parent

    def backprop(self, x, y):
        #print('Enter backprop\n')
        # calculate back_propogation delta
        # return delta

        # Initialization
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        #print('nabla_b',len(nabla_b))
        nabla_b_mult = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        nabla_w_mult = [np.zeros(w.shape) for w in self.weights]

        # Create H_hat_matrix and b_vector
        self.h_hat, self.b_intercept = self.h_matrix_init(self.sizes)

        # sample number
        n = x.shape[1]

        # acitvation list for each layer
        activations_layer = [x]
        #print('x',x[103,:],'\n')

        status = 'training'
        # Generate X_epsilon = H_hat * X,
        # b_intercept corresponding in formula (4)
        # print('point 1')
        x_epsilon, self.h_hat[0], self.b_intercept[0] = self.epsilon_gen(
            x, self.index_parent[0], self.h_hat[0], self.b_intercept[0],
            status)
        #print('h_hat0', self.h_hat[0].shape)
        #for j in range(self.h_hat[0].shape[0]):
        #   print('self.h_hat[0]',j,self.h_hat[0][j,:])

        # Set one transient variable activation to represent
        activation = x_epsilon
        #print('x_epsilon', x_epsilon[:, 1])

        # Define z_s as the list of variable before activation function
        z_s = []
        i = 1

        for b, w in zip(self.biases[:-1], self.weights[:-1]):
            # do matrix multiplication
            #print('w.shape', w.shape, 'b', len(b))
            #for j in range(w.shape[0]):
            #    print('w[j,:]',j, w[j,:],'\n')
            z = np.dot(w, activation) + b
            z_s.append(z)

            # do activation
            activation = self.tanh(z)
            #activation = self.relu(z)
            activations_layer.append(activation)

            # iterative this process
            #print(len(self.sizes))
            if i < (len(self.sizes) - 2):
                #print('self.h_hat[i]',i, self.h_hat[i])
                activation, self.h_hat[i], self.b_intercept[
                    i] = self.epsilon_gen(activation, self.index_parent[i],
                                          self.h_hat[i], self.b_intercept[i],
                                          status)
                #print('h_hat i', i, self.h_hat[i])
                i += 1

        #print('activation.shape', activation.shape)
        #print(self.h_hat[-1].shape)
        #for j in range(self.h_hat[-1].shape[0]):
        #        print('self.h_hat[-1]',j, self.h_hat[-1][j,:],'\n')
        activation = np.dot(self.h_hat[-1], activation)
        #print('activation', activation)

        # For the last layer and check objective
        z = np.dot(self.weights[-1], activation) + self.biases[-1]
        #print('self.biases[-1] shape', self.biases[-1], '\n')
        #print('self.weights[-1] shape', self.weights[-1], '\n')
        #print('z -1', z)

        # add covariates to last layer here
        #z_cov = self.add_cov(self.w_cov,self.b_cov,self.x_cov)
        #z= z + z_cov

        if self.objective == 'classification':
            #z = self.softmax(z)
            z = self.sigmoid(z)
            #print('z',len(z[0]))
        elif self.objective == 'regression':
            z = z

        z_s.append(z)
        #print('z_s', z_s, '\n')

        # Use for later
        activation = z
        # The last result = y_hat
        activations_layer.append(z)
        #print('activations_layer', activations_layer, '\n')

        # Method2: backward method for each data point
        #print('self.h_hat[-1]',self.h_hat[-1],'\n')
        #print('n',n)
        for i in range(n):
            # backward pass
            # In the last layer's delta, self computed by objective and eval_metric
            #print('y[:, i]',i,y[:, i],'\n')
            #print('activations_layer[-1][:, i]',activations_layer[-1][:, i],'\n')
            delta = self.loss_derivative(activations_layer[-1][:, i], y[:, i])
            #print('output layer delta', i, delta,'\n')

            # Reshape delta to array [[]] format_type (2,1)
            delta = np.reshape(delta, (len(delta), 1))

            # nabla_b[-1]
            nabla_b[-1] = delta
            #nabla_b_cov = delta

            # nabla_w[-1]
            # H_X is the H * X
            self.h_hat[-1]
            H_X = np.dot(self.h_hat[-1], activations_layer[-2][:, i])
            nabla_w[-1] = np.dot(delta,
                                 np.reshape(H_X, (len(H_X), 1)).transpose())
            #nabla_w_cov = np.dot(delta, np.reshape(self.x_cov[:,i],(len(self.x_cov[:,i]),1)).transpose())
            #print('self.h_hat[-1]',self.h_hat[-1],'/n')
            #print('activationX',activations_layer[-2][:, i])
            #print('HX', H_X)

            # then back_propagation from L-1 layer
            for l in range(2, self.num_layers):
                #print('l',l)

                # Reshape the z
                z = np.reshape(z_s[-l][:, i], (len(z_s[-l][:, i]), 1))

                # Tanh derivative
                s_p = self.tanh_prime(z)
                #s_p = self.relu_prime(z)

                # Appendix formula (12)
                delta = np.dot(
                    np.dot(self.weights[-l + 1],
                           self.h_hat[-l + 1]).transpose(), delta) * s_p
                #print('self.weights[-l + 1]',self.weights[-l + 1].shape)

                #print('delta',l,delta,'\n')

                # Set nabla_b is the delta
                nabla_b[-l] = delta

                # Set nabla_w is the H_X: (H*x) * delta
                H_X = np.dot(self.h_hat[-l], activations_layer[-l - 1][:, i])
                nabla_w[-l] = np.dot(
                    delta,
                    np.reshape(H_X, (len(H_X), 1)).transpose())
                #print('self.h_hat[-l]',l,self.h_hat[-l],'\n')
                #for j in range(self.h_hat[-l].shape[0]):
                #    print('self.h_hat[-l]',j,self.h_hat[-l][j,:])
                #print('activationX',j,activations_layer[-l-1][:, i])
                #print('HX',l, H_X)

            for k in range(len(nabla_b)):
                nabla_b_mult[k] += nabla_b[k]
                nabla_w_mult[k] += nabla_w[k]

        for k in range(len(nabla_b)):
            nabla_b_mult[k] = nabla_b_mult[k] / n
            nabla_w_mult[k] = nabla_w_mult[k] / n
        nabla_w_mult = self.w_assign_zero(self.weights, nabla_w_mult)

        #for k in range(len(nabla_w_mult)):
        #print('nabla_w_mult',k, nabla_w_mult[k],'\n')
        #return (nabla_b_mult, nabla_w_mult, nabla_b_cov, nabla_w_cov)
        #print('nabla_b_mult',nabla_b_mult[-2].shape,'nabla_w_mult',nabla_w_mult[-2].shape,'\n')
        return (nabla_b_mult, nabla_w_mult)

    def feed_forward(self, x, y):
        #print('Enter feed_forward\n')
        # Calculate final predict

        # For layer 0
        i = 0
        # Initialization the linear part coefficients
        if self.coefficient_share == 'False':
            h_hat, b_intercept = self.h_matrix_init(self.sizes)
        elif self.coefficient_share == 'True':
            h_hat, b_intercept = self.h_hat, self.b_intercept

        #print('h_hat in feed_forward', h_hat)

        status = 'training'
        # For layer i
        for b, w in zip(self.biases[:-1], self.weights[:-1]):
            # Calculate x_s
            #print('point 3',i)
            #print('x ff',x)
            x_s, h_hat[i], b_intercept[i] = self.epsilon_gen(
                x, self.index_parent[i], h_hat[i], b_intercept[i], status)
            x = self.tanh(np.dot(w, x_s) + b)
            #x = self.relu(np.dot(w, x_s) + b)
            #print('w',w)
            #print('b',b)
            i += 1

        # For layer L
        #print('point 4')
        x_s, h_hat[-1], b_intercept[-1] = self.epsilon_gen(
            x, self.index_parent[-1], h_hat[-1], b_intercept[-1], status)

        # Get the final predict
        if self.objective == 'regression':
            y_hat = np.dot(self.weights[-1], x_s) + self.biases[-1]
            mse = np.array([np.sum((y_hat - y)**2) / y.shape[1]])
            return mse, y_hat

        elif self.objective == 'classification':
            #print('x_s last layer',x_s,'\n')
            #print('self.weights[-1] excluding covariates',self.weights[-1],'\n')
            #print('self.biases[-1]',self.biases[-1],'\n')
            z = np.dot(self.weights[-1], x_s) + self.biases[-1]

            #add covariates
            #z_cov = self.add_cov(self.w_cov,self.b_cov,self.x_cov)
            #z += z_cov

            #y_hat = self.softmax(z)
            y_hat = self.sigmoid(z)

            #print('y[0]',y[0],'\n')

            #print('y_hat in feed_forward',y_hat[0],'\n')
            y_hat_label = self.convert_to_label(y_hat)

            if self.eval_metric == 'cross_entropy':
                # Cross_Entropy
                c_e = self.Cross_Entropy(y_hat, y[0])

                # Calculate Precision score
                print_prfa, auc, acc = self.scores(y[0], y_hat_label, y_hat[0])

                #print('\n')
                #print('y[0] in feedforward', y[0], '\n')
                #print('y_hat[0]+y_hat[1]',y_hat[0]+y_hat[1],'y_pred=y_hat[1]',y_hat[1],'\n')
                #print('y_hat_label in feedforward', y_hat_label, '\n')

                return c_e, auc, acc, y_hat[0], print_prfa

    def h_matrix_init(self, sizes):
        #print('Enter h_matrix_init\n')
        # H hat matrix initialization

        h_matrix = [[] for _ in range(len(sizes) - 1)]
        b_bias = [[] for _ in range(len(sizes) - 1)]

        for i in range(len(sizes) - 1):
            h_upper = np.identity(sizes[i])
            h_lower = np.identity(sizes[i])
            h_matrix[i] = np.concatenate((h_upper, h_lower), axis=0)
            b_bias[i] = np.zeros((2 * sizes[i], 1))
        h_matrix[-1] = np.concatenate(
            (np.identity(sizes[-2]), np.zeros([sizes[-2], sizes[-2]])), axis=0)
        #print('h_matrix_init',h_matrix,'\n')
        return h_matrix, b_bias

    '''
    def epsilon_gen(self,
                    x,
                    i_p,
                    h_mat,
                    b_intercept,
                    status,
                    residual_set=True):
        print('Enter epsilon_gen\n')
        # x is the data matrix: m*n
        # index parent: is the that layer index child_parent dictionary
        # h_mat: the corresponding layer H_hat matrix
        # b_intercept: the corresponding layer intercept vector

        if status == 'testing':
            if self.coefficient_share == 'False':
                regr = linear_model.LinearRegression()

                # sample = np.shape(x)[1]
                node_number = np.shape(x)[0]

                #epsilon = np.zeros((node,sample))
                #ancestryisnull = []
                #for i in range(len(idp)):
                #    if idp[i] == []:
                #        ancestryisnull.append(i)
                #for i in ancestryisnull:
                #   x[node + i] = x[i]

                # i is the node index
                for i in range(len(i_p)):
                    # if node doesn't have parent
                    if i_p[i] == []:
                        # Just set the h_mat diagonal is 1
                        h_mat[node_number + i][i] = 0
                        b_intercept[node_number + i] = 0

                    # For node i; j is the index of its parent
                    for j in range(len(i_p[i])):
                        parent_list = i_p[i]
                        current_list = np.array([i])
                        # a: parent node list
                        # b: current node list
                        a = x[parent_list][:]
                        b = x[current_list][:]
                        a = a.reshape((np.shape(a)[1], np.shape(a)[0]))
                        b = b.reshape((np.shape(b)[1], np.shape(b)[0]))

                        # method expand [x] -> [x, epsilon]
                        if residual_set == True:
                            regr.fit(a, b)  # (x, y)
                            # residues: b - regr.predict(a); regr.coef_; regr.intercept_
                            # Since next following operation is X-x_hat,
                            # So the corresponding coefficient will be negative
                            if self.format_type == 'residual':
                                h_mat[node_number + i][i_p[i]] = -regr.coef_
                                b_intercept[node_number + i] = -regr.intercept_
                            elif self.format_type == 'original':
                                h_mat[node_number + i][i_p[i]] = 1
                                b_intercept[node_number + i] = 0

                x_epsilon = np.dot(h_mat, x) + b_intercept
            elif self.coefficient_share == 'True':
                x_epsilon = np.dot(h_mat, x) + b_intercept

        elif status == 'training':
            regr = linear_model.LinearRegression()

            # sample = np.shape(x)[1]
            node_number = np.shape(x)[0]

            # i is the node index
            for i in range(len(i_p)):
                #print('h_mat[node_number + i]',h_mat[node_number + i])
                # if node doesn't have parent
                if i_p[i] == []:
                    #print('i_p[i] == []',i)
                    # Just set the h_mat diagonal is 1
                    h_mat[node_number + i][i] = 0
                    b_intercept[node_number + i] = 0

                # For node i; j is the index of its parent #range(len(i_p[i]))
                for j in range(len(i_p[i])):
                    parent_list = i_p[i]
                    #print('parent_list',parent_list,'\n')
                    current_list = np.array([i])
                    #print('current_list',current_list,'\n')
                    # a: parent node list
                    # b: current node list
                    #print('x',x,'\n')
                    a = x[parent_list][:]
                    b = x[current_list][:]
                    a = a.reshape((np.shape(a)[1], np.shape(a)[0]))
                    b = b.reshape((np.shape(b)[1], np.shape(b)[0]))
                    # method expand [x] -> [x, epsilon]
                    if residual_set == True:
                        #print('i',i,'j',j,'a',a.shape,'b',b.shape,'\n')
                        regr.fit(a, b)  # (x, y)
                        # residues: b - regr.predict(a); regr.coef_; regr.intercept_
                        # Since next following operation is X-x_hat,
                        # So the corresponding coefficient will be negative
                        if self.format_type == 'residual':
                            #print('-regr.coef_',-regr.coef_)
                            h_mat[node_number + i][i_p[i]] = -regr.coef_
                            #print('assign h_mat',h_mat[node_number + i])
                            b_intercept[node_number + i] = -regr.intercept_
                        elif self.format_type == 'original':
                            h_mat[node_number + i][i_p[i]] = 1
                            b_intercept[node_number + i] = 0

            x_epsilon = np.dot(h_mat, x) + b_intercept

        #print('hmat in epsilon', h_mat, '\n')

        return x_epsilon, h_mat, b_intercept
    '''

    def epsilon_gen(self,
                    x,
                    i_p,
                    h_mat,
                    b_intercept,
                    status,
                    residual_set=True):
        # x is the data matrix: m*n
        # index parent: is the that layer index child_parent dictionary
        # h_mat: the corresponding layer H_hat matrix
        # b_intercept: the corresponding layer intercept vector

        regr = linear_model.LinearRegression()

        # sample = np.shape(x)[1]
        node_number = np.shape(x)[0]

        #epsilon = np.zeros((node,sample))
        #ancestryisnull = []
        #for i in range(len(idp)):
        #    if idp[i] == []:
        #        ancestryisnull.append(i)
        #for i in ancestryisnull:
        #   x[node + i] = x[i]'''

        # i is the node index
        for i in range(len(i_p)):
            # if node doesn't have parent
            if i_p[i] == []:
                # Just set the h_mat diagonal is 1
                h_mat[node_number + i][i] = 1
                b_intercept[node_number + i] = 0
            # For node i; j is the index of its parent
            for j in range(len(i_p[i])):
                parent_list = i_p[i]
                current_list = np.array([i])
                # a: parent node list
                # b: current node list
                a = x[parent_list][:]
                b = x[current_list][:]
                a = a.reshape((np.shape(a)[1], np.shape(a)[0]))
                b = b.reshape((np.shape(b)[1], np.shape(b)[0]))
                # method expand [x] -> [x, epsilon]
                if residual_set == True:
                    regr.fit(a, b)  # (x, y)
                    # residues: b - regr.predict(a); regr.coef_; regr.intercept_
                    # Since next following operation is X-x_hat,
                    # So the corresponding coefficient will be negative
                    h_mat[node_number + i][i_p[i]] = -regr.coef_
                    b_intercept[node_number + i] = -regr.intercept_
                elif residual_set == False:
                    h_mat[node_number + i][i_p[i]] = 1
                    b_intercept[node_number + i] = 0

        x_epsilon = np.dot(h_mat, x) + b_intercept
        return x_epsilon, h_mat, b_intercept

    def loss_derivative(self, y_hat, y):
        #print('Enter loss_derivative\n')
        # yhat = sigma(z)
        # dC/dz = dC/d_sigma*d_sigma/dz
        #der1 = np.array([(-(1-y)/y_hat[0]+y/(1-y_hat[0])),(-y/y_hat[1]+(1-y)/(1-y_hat[1]))])
        #der2 = np.array([y_hat[0]*(1-y_hat[0]) , y_hat[1]*(1-y_hat[1])])
        #return der1 * np.reshape(der2,[2,1])
        #return (np.array([y_hat[0]-(1-y),y_hat[1]-y]))
        # Calculate loss function derivative to z
        #print(y_hat,y,y_hat-y)
        return (y_hat - y)
        '''
        if self.objective == 'classification':
            if self.eval_metric == 'cross_entropy':
                if y == 0:
                    return self.loss_weight[0] * np.array([(y_hat[0] - y).tolist()[0], y_hat[1]])
                else:
                    return self.loss_weight[1] * np.array([y_hat[0], (y_hat[1] - y).tolist()[0]])
        
            elif self.eval_metric == 'error':
                return (y_hat - y) * (self.sigmoid(y_hat) * (1 - self.sigmoid(y_hat)))
        elif self.objective == 'regression':
            if self.eval_metric == 'mse':
                return (y_hat - y)'''

    '''
    def cov_init(self, cv_num, status):
        # status: 'train' or 'test'
        # load weight and bias for 5 covarites
        w_cov = np.loadtxt('/Volumes/Penn/w_cov_init.txt')  #2*5
        b_cov = np.loadtxt('/Volumes/Penn/b_cov_init.txt')  #2*1
        b_cov = np.reshape(b_cov, (2, 1))
        x_cov = np.loadtxt('/Volumes/Penn/cov_%s_%d.txt' % (status, cv_num))
        return w_cov, b_cov, x_cov
    '''

    def add_cov(self, w_cov, b_cov, x_cov):
        return np.dot(w_cov, x_cov)
        #+ b_cov

    def w_assign_zero(self, weights, nabla_w):
        """return add weighted matrix"""
        for a, b in zip(self.weights, nabla_w):
            for i in range(np.shape(a)[0]):
                for j in range(np.shape(a)[1]):
                    if a[i][j] == 0:
                        b[i][j] = 0
        return nabla_w

    def evaluate(self, mini_batch):
        #print('Enter evaluate\n')
        # calculate loss
        x, y = np.transpose(mini_batch[0]), np.transpose(mini_batch[1])
        if self.objective == 'classification':
            if self.eval_metric == 'cross_entropy':
                #print('x eva',x)
                c_e, auc, acc, y_pred, print_prfa = self.feed_forward(x, y)
                return c_e, auc, acc, y_pred, print_prfa
        elif self.objective == 'regression':
            mse, y_pred = self.feed_forward(x, y)
            return mse

    def Classification(self, y_hat):
        for i in range(len(y_hat)):
            if y_hat[i] > 0.5:
                y_hat[i] = 1
            else:
                y_hat[i] = 0
        return y_hat

    def Cross_Entropy(self, y_hat, y):
        #print('Enter C_E\n')
        # Calculate Cross_Entropy
        #print('y_hat in Cross_Entropy',y_hat,'\n')
        n = y_hat.shape[1]
        #y = self.convert_to_onehot(y)
        c_e = 0
        for i in range(n):
            #c_e += y[i][0] * np.log(y_hat[0][i]+1e-10) + y[i][1]* np.log(y_hat[1][i]+1e-10)
            c_e += y[i] * np.log(y_hat[0][i] + 1e-10) + 4*(
                1 - y[i]) * np.log(1 - y_hat[0][i] + 1e-10)

        c_e = -c_e / n
        return c_e

    def convert_to_label(self, y):
        #print('Enter convert_to_label\n')
        n = y.shape[1]
        z = np.zeros(n)
        for i in range(n):
            #if y[0][i] <= y[1][i]:
            #print(i,'y[0][i],y[1][i]',y[0][i],y[1][i],'\n')
            if y[0][i] >= 0.5:
                z[i] = 1
            else:
                z[i] = 0
        return z

    def convert_to_onehot(self, y):
        n = len(y)
        z = np.zeros([n, 2])
        for i in range(n):
            if y[i] == 0:
                z[i][0] = 1
            else:
                z[i][1] = 1
        return z

    def tanh(self, z):
        return np.tanh(z)

    def tanh_prime(self, z):
        return 1.0 - (np.tanh(z) * np.tanh(z))

    def sigmoid(self, z):
        return 1.0 / (1.0 + np.exp(-z))

    def sigmoid_prime(self, z):
        return self.sigmoid(z) * (1 - self.sigmoid(z))

    def relu(self, z):
        z[z < 0] = 0
        return z

    def relu_prime(self, z):
        return np.where(z > 0, 1.0, 0.0)

    def softmax(self, z):
        # Avoid too large
        shiftz = z - z.max(axis=0)
        exps = np.exp(shiftz)
        final = exps / exps.sum(0)
        return final

    def load_checkpoint(self, checkpoint_path):
        # It's weird that if `map_location` is not given, it will be extremely slow.
        return torch.load(checkpoint_path,
                          map_location=lambda storage, loc: storage)

    def test_main(self,
                  test_data,
                  loss_validation,
                  sizes,
                  optimal_epoch_given,
                  LOAD_CHECKPOINT=False):
        #print('Enter test_main\n')
        # return test loss

        # Select the best parameters in the model
        if optimal_epoch_given == False:
            optimal_epoch = loss_validation.index(min(loss_validation)) + 1
        else:
            optimal_epoch = optimal_epoch_given

        if LOAD_CHECKPOINT == True:
            # load from path.
            checkpoint_path = os.path.join(
                os.path.join('model_checkpoint_train/',
                             "SDL_epoch_%d.pt" % optimal_epoch))
            checkpoint = self.load_checkpoint(checkpoint_path)

            # load needed parameters
            w_hat = checkpoint['w']
            b_hat = checkpoint['biases']
            epoch = checkpoint['epoch']
            lr = checkpoint['learning_rate']
            if self.coefficient_share == 'True':
                h_hat = checkpoint['h_hat']
                #print('h_hat',h_hat,'\n')
                b_intercept = checkpoint['b_intercept']
            elif self.coefficient_share == 'False':
                h_hat = 0
                b_intercept = 0

            # Organize data
            test_data = self.data_organize(list(test_data))

            # Predict test results
            true_y, y_hat, predict_y, print_prsa, auc, acc, c_e = self.test_prediction(
                test_data, w_hat, b_hat, sizes, h_hat, b_intercept)

            m = test_data[0].shape[1]
            n = test_data[0].shape[0]

            #print('test true_y', true_y[0])
            #print('test y_hat', y_hat[0])
            #print('test predict_y', predict_y)

            if not os.path.exists('model_checkpoint_test'):
                os.makedirs('model_checkpoint_test')

            # Check points
            testing_checkpoint = {
                'weight': w_hat,
                'biases': b_hat,
                'learning_rate': lr,
                'sizes': sizes,
                'optimal_epoch': optimal_epoch,
                'true_y': true_y,
                'predict_y': predict_y,
                'm': m,
                'n': n
            }

            testing_model_name = '%s%d_%s%d_%s%d%s' % (
                'model_checkpoint_test/SDL_test_m_', m, 'n_', n, 'optimal_',
                optimal_epoch, '.pt')

            # Need change the name of saving the model parameter name
            torch.save(testing_checkpoint, testing_model_name)
            print('Optimal epoch: %d' % optimal_epoch)
            print(print_prsa)
            #print('true_y[0]', true_y[0])
            #print('predict_y', predict_y)
            #print('y_hat[1]', np.around(y_hat[0], 3))

            return true_y[0], y_hat[0], auc, acc, c_e

        else:
            print('Need checkpoint, please try again!')

    def test_prediction(self, test_data, w_hat, b_hat, sizes, h_hat,
                        b_intercept):
        #print('Enter test_prediction\n')
        # Return y, predict_y, and loss of test data

        # load test cov
        #_,_,self.test_cov = self.cov_init(cv_num,'test')

        true_y, y_hat, predict_y, print_prsa, auc, acc,c_e=\
                         self.test_feed_forward(np.transpose(test_data[0]), np.transpose(test_data[1]),\
                                                w_hat, b_hat, sizes, h_hat, b_intercept)
        return true_y, y_hat, predict_y, print_prsa, auc, acc, c_e

    # Test part needs modification
    def test_feed_forward(self,
                          x,
                          y,
                          w_hat,
                          b_hat,
                          sizes,
                          h_hat_inh=None,
                          b_intercept_inh=None):
        #print('Enter test_feed_forward\n')
        # For layer 0
        i = 0

        #print('test_data x', x.shape)
        #print('test_data y', y.shape)
        #print('w_hat', w_hat[:-1])
        #print('b_hat', b_hat[:-1])

        # Initialization the linear part coefficients
        h_hat, b_intercept = self.h_matrix_init(self.sizes)
        '''
        # Initialization the linear part coefficients
        if self.coefficient_share == 'Flase':
            h_hat, b_intercept = self.h_matrix_init(sizes)
        elif self.coefficient_share == 'True':
            h_hat, b_intercept = h_hat_inh, b_intercept_inh
        '''

        #print('h_hat in test_feed_forward', h_hat)

        status = 'testing'
        ##########################################################
        # For layer i
        for b, w in zip(b_hat[:-1], w_hat[:-1]):
            # Calculate x_s
            x_s, h_hat[i], b_intercept[i] = self.epsilon_gen(
                x, self.index_parent[i], h_hat[i], b_intercept[i], status)
            #print('w', w.shape)
            #print('x_s', x_s.shape)
            #print('b', b.shape)
            x = self.tanh(np.dot(w, x_s) + b)
            #x = self.relu(np.dot(w, x_s) + b)
            i += 1

        # For layer L
        x_s, h_hat[-1], b_intercept[-1] = self.epsilon_gen(
            x, self.index_parent[-1], h_hat[-1], b_intercept[-1], status)

        # Get the final predict
        if self.objective == 'regression':
            y_hat = np.dot(self.weights[-1], x_s) + self.biases[-1]
            mse = np.array([np.sum((y_hat - y)**2) / y.shape[1]])
            return y, y_hat, mse

        elif self.objective == 'classification':
            z = np.dot(self.weights[-1], x_s) + self.biases[-1]
            #print('self.biases[-1] shape',self.biases[-1].shape,'\n')

            #add covariates
            #z_cov =self.add_cov(self.w_cov,self.b_cov,self.test_cov)
            #z=z+z_cov

            #y_hat = self.softmax(z)
            y_hat = self.sigmoid(z)
            y_hat_label = self.convert_to_label(y_hat)

            if self.eval_metric == 'cross_entropy':
                c_e = self.Cross_Entropy(y_hat, y[0])
                #loss_test = np.array([self.Cross_Entropy(y_hat, y)/y.shape[1]])
                print_prsa, auc, acc = self.scores(y[0], y_hat_label, y_hat[0])
                return y, y_hat, y_hat_label, print_prsa, auc, acc, c_e

    def scores(self, y_true, y_pred, y_hat_prob):
        #print('Enter scores\n')
        (precision_0, recall_0, f1_0,
         warning) = precision_recall_fscore_support(y_true,
                                                    y_pred,
                                                    labels=[0],
                                                    average='weighted',
                                                    warn_for=tuple())

        (precision_1, recall_1, f1_1,
         warning) = precision_recall_fscore_support(y_true,
                                                    y_pred,
                                                    labels=[1],
                                                    average='weighted',
                                                    warn_for=tuple())

        (precision, recall, f1,
         warning) = precision_recall_fscore_support(y_true,
                                                    y_pred,
                                                    average='weighted',
                                                    warn_for=tuple())

        false_positive_rate, true_positive_rate, thresholds = roc_curve(
            y_true, y_hat_prob)
        #plt.plot(false_positive_rate,true_positive_rate)
        #plt.show()
        #plt.close()
        AUC = auc(false_positive_rate, true_positive_rate)

        ACC = accuracy_score(y_true, y_pred)
        print_prfa = '    Precision   Recall    f1-score     AUC     ACC\n' \
                    '0:    %.3f      %.3f      %.3f     %.3f     %.3f\n' \
                    '1:    %.3f      %.3f      %.3f         \n' \
                    'w:    %.3f      %.3f      %.3f         \n' % (precision_0, recall_0, f1_0, AUC, ACC,
                                                                   precision_1, recall_1, f1_1,
                                                                   precision, recall, f1)
        return print_prfa, AUC, ACC

