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
from module_time import time_since

# http://neuralnetworksanddeeplearning.com/chap2.html

class Network_SDL(object):
    def __init__(self, sizes, adj_matrix, parent_child, child_parent, residual_set):
        # layer_predictor_number/sizes: such like [7, 3, 1]
        # adj_matrix: adjacent matrix
        # parent_child: parent child dict
        # chile_parent: child parent dict

        np.random.seed(12345)

        self.sizes = sizes
        self.num_layers = len(sizes)

        self.parent_child = parent_child
        self.child_parent = child_parent

        self.residual_set = residual_set
        # biased: sizes[i]*1
        self.biases = [np.random.randn(i, 1) for i in self.sizes[1:]]

        # adjacent matrix
        self.adj_matrix = adj_matrix

        # calculate each layer which nodes will be left.
        self.list_of_node = self.layer_node_number(parent_child, child_parent, sizes)

        # Initialization weight matrix
        self.weights_ur, self.weights = self.weight_matrix_init(
            adj_matrix, self.sizes, self.list_of_node)

        # Calculate each layer's child and parent dictionary
        self.index_child, self.index_parent = self.pc_dict(self.weights_ur, self.sizes)

    def SGD(self, training_data, validation_data, test_data,
            n_train, n_val, n_test, end_epoch, mini_batch_size, val_mini_batch_size, eta):
        # use stochastic gradient descent method to optimize the objective function

        # training_data: training data
        # epochs: learning epochs
        # mini_batch_size
        # eta: learning rate, lr
        # validation_data
        # test_data

        # loss dict
        loss_training = []
        loss_validation = []

        # temp_w = [np.zeros([y, x]) for x, y in zip(self.sizes[:-1], self.sizes[1:])]
        # temp_b = [np.zeros([y, 1]) for y in self.sizes[1:]]

        # calculate time
        start = time.time()

        group = n_train / mini_batch_size * end_epoch

        print("\nStart training")
        # random select training data into mini batch
        for epoch in range(end_epoch):
            step = 0
            random.shuffle(training_data)

            # create mini_batches
            mini_batches = [list(training_data)[k:k + mini_batch_size]
                            for k in range(0, n_train, mini_batch_size)]

            # use mini_batch to calculate loss
            for mini_batch in mini_batches:
                step += 1
                # Organize data
                mini_batch = self.data_organize(mini_batch)

                # Update parameters
                self.weights, self.biases = self.update_weight(mini_batch, eta)

                # Calculate loss
                loss_training.append(self.evaluate(mini_batch))

                # Print training loss
                print_training = 'Epoch: %d / %d, Accumulate Time to End: %s, ' \
                                 '(Batch: %d / Batches num: %d, Percent Run: %.2f%%), '\
                                 'Training MSE Loss: %.4f' % (epoch+1, end_epoch, time_since(start,
                                        (step+epoch*n_train/mini_batch_size)/group),
                                        step, n_train / mini_batch_size,
                                        (epoch * (n_train / mini_batch_size) + step)/group * 100,
                                        self.evaluate(mini_batch))
                print(print_training)
                sys.stdout.flush()
                # print(self.weights[-2], self.biases[-2])

            ######################################################################
            # Use validation method to select the best parameters
            ######################################################################

            # Create mini_batches
            val_mini_batches = [list(validation_data)[k:k + val_mini_batch_size]
                                for k in range(0, n_test, val_mini_batch_size)]

            # use mini_batch to calculate loss
            loss_val_all = 0
            loss_val_part = 0
            loss_val_average = 0

            for val_mini_batch in val_mini_batches:
                # transform data format
                val_mini_batch = self.data_organize(val_mini_batch)

                # Keep loss_validation
                loss_val_part = self.evaluate(val_mini_batch)
                loss_val_all += loss_val_part

            # average loss of val
            loss_val_average = loss_val_all/len(val_mini_batches)

            # Save validation_loss
            loss_validation.append(loss_val_average)
            # Print training loss
            print_val = '\nValidation: Epoch: %d / %d, Validation MSE Loss: %.4f' % (epoch+1, end_epoch, loss_val_average)
            print(print_val)
            sys.stdout.flush()

            # check points
            # First create a model check point folder
            if not os.path.exists('model_checkpoint_train'):
                os.makedirs('model_checkpoint_train')

            checkpoint = {
                'epoch': epoch+1,
                'learning_rate': eta,
                'w': self.weights,
                'biases': self.biases,
                'validation_loss': loss_val_average
            }

            # Need change the name of saving the model parameter name
            save_epoch = epoch + 1
            model_name = os.path.join('model_checkpoint_train/', "SDL_epoch_%d.pt" % save_epoch)

            # Save model parameters
            torch.save(checkpoint, model_name)
            print("Save model as %s \n" % model_name)

        # For test
        # we can select the minimal of validation then use if for test result.
        return loss_validation

    def data_organize(self, mini_batch):
        # transform data from [[x1,y1], [x2,y2]] to [[x1, x2], [y1, y2]]

        # mini_batch_size
        n = len(mini_batch)

        # predictor size
        m = mini_batch[0][0].shape[0]

        x = []
        y = []
        for i in range(n):
            x.append(mini_batch[i][0])
            y.append(mini_batch[i][1])

        x = np.reshape(x, (n, m))
        y = np.reshape(y, (n, 1))
        return [x, y]

    def update_weight(self, mini_batch, eta):
        # update data weights and bias

        # get X design matrix and response y vector
        x_matrix = mini_batch[0]
        y_vector = mini_batch[1]

        # get delta variable;  x: m*n,  y: 1*n
        delta_nabla_b, delta_nabla_w = self.backprop(np.transpose(x_matrix), np.transpose(y_vector))

        # update weights and biases
        weights = [w - (eta * nw)
                   for w, nw in zip(self.weights, delta_nabla_w)]
        biases = [b - (eta * nb)
                  for b, nb in zip(self.biases, delta_nabla_b)]

        return weights, biases

    def layer_node_number(self, parent_child, child_parent, sizes):
        # calculate each layer node

        list_of_node = [[] for _ in range(len(sizes))]
        list_of_node[-1] = [0]

        for k in range(len(sizes) - 1):
            list_of_node[k] = list(parent_child.keys())
            if k == len(sizes) - 1:
                break

            delete_number = []
            parent_to_null = []

            for i in list_of_node[k]:
                if parent_child[i] == []:
                    # avoid root nodes
                    if child_parent[i] == []:
                        continue
                    # delete nodes who don't have child
                    else:
                        # nodes need to be deleted
                        delete_number.append(i)
                        # next step leaf nodes
                        parent_to_null.append(child_parent[i])

            # update parent-child dict
            for i in list(list_of_node[k]):
                # prepare delete nodes who don't have children
                if parent_child[i] == []:
                    # avoid root nodes
                    if child_parent[i] == []:
                        break
                    else:
                        del parent_child[i]

            # set parents to next layer child
            for i in range(len(delete_number)):
                for j in parent_to_null[i]:
                    parent_child[j] = []

        return list_of_node

    def weight_matrix_init(self, adj_matrix, sizes, list_of_node, scale = False):
        # Initialization weight matrix, created w_final
        # return w_final

        # Define fixed random effect
        np.random.seed(12345)

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

        # Delete the last row: [x_0, epsilon_0] -> [x_0]
        w_mat[-1] = w_mat[-1].drop(['epsilon_0'], axis=0)


        # w_nda, w_nda_u is the normal matrix
        # w_nda_u to become double matrix
        for i in range(len(sizes)-1):
            next_layer_size_by_two = w_nda_u[i].shape[0]
            # Assign upper, right half of w_mat to upper left of w_nda_u
            for j in range(int(next_layer_size_by_two/2)):
                w_nda_u[i][j][j] = (w_mat[i].values)[j][j]

        w_nda_u[-1] = np.delete(w_nda_u[-1], 1, 0)

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
        w_final[-1] = w_nda_u[-1]

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
                index_parent[l][j] = list(np.where(weights_ur[l][..., j] != 0)[0])

        # extra add - need fix: double make sure
        index_parent[-1][0] = []

        return index_child, index_parent

    def backprop(self, x, y):
        # calculate back_propogation delta
        # return delta

        # Initialization
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_b_mult = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        nabla_w_mult = [np.zeros(w.shape) for w in self.weights]

        # H_hat_matrix and b_vector
        h_hat, b_intercept = self.h_matrix_init(self.sizes)

        # sample number
        n = x.shape[1]

        # acitvation list for each layer
        activations_layer = [x]

        # Generate X_epsilon = H_hat * X,
        # b_intercept corresponding in formula (4)
        x_epsilon, h_hat[0], b_intercept[0] = self.epsilon_gen(x, self.index_parent[0], h_hat[0], b_intercept[0])

        # Set one transient variable activation to represent
        activation = x_epsilon

        # Define z_s as the list of variable before activation function
        z_s = []
        i = 1
        for b, w in zip(self.biases[:-1], self.weights[:-1]):
            # do matrix multiplication
            z = np.dot(w, activation) + b
            z_s.append(z)

            # do sigmoid
            activation = self.sigmoid(z)
            activations_layer.append(activation)

            # iterative this process
            if i < (len(self.sizes) - 1):
                activation, h_hat[i], b_intercept[i] = self.epsilon_gen(
                    activation, self.index_parent[i], h_hat[i], b_intercept[i])
                i += 1

        # For the last layer
        z = np.dot(self.weights[-1], activation) + self.biases[-1]
        z_s.append(z)

        # Use for later
        activation = z
        # The last result = y_hat
        activations_layer.append(z)

        # Method2: backward method for each data point
        for i in range(n):
            # backward pass
            # In the last layer's delta
            delta = self.loss_derivative(activations_layer[-1][:, i], y[:, i])  # * sigmoid_prime(zs[-1])
            # Reshape delta to array [[]] format
            delta = np.reshape(delta, (len(delta), 1))
            # nabla_b[-1]
            nabla_b[-1] = delta
            # nabla_w[-1]
            # H_X is the H * X
            H_X = np.dot(h_hat[-1], activations_layer[-2][:, i])
            nabla_w[-1] = np.dot(delta, np.reshape(H_X, (len(H_X), 1)).transpose())

            # then back_propagation from L-1 layer
            for l in range(2, self.num_layers):
                # Reshape the z
                z = np.reshape(z_s[-l][:, i], (len(z_s[-l][:, i]), 1))
                # Sigmoid derivative
                s_p = self.sigmoid_prime(z)
                # Appendix formula (12)
                delta = np.dot(np.dot(self.weights[-l + 1], h_hat[-l + 1]).transpose(),
                               delta) * s_p
                # Set nabla_b is the delta
                nabla_b[-l] = delta
                # Set nabla_w is the H_X: (H*x) * delta
                H_X = np.dot(h_hat[-l], activations_layer[-l - 1][:, i])
                nabla_w[-l] = np.dot(delta, np.reshape(H_X, (len(H_X), 1)).transpose())

            for k in range(len(nabla_b)):
                nabla_b_mult[k] = nabla_b_mult[k] + nabla_b[k]
                nabla_w_mult[k] = nabla_w_mult[k] + nabla_w[k]

        for k in range(len(nabla_b)):
            nabla_b_mult[k] = nabla_b_mult[k] / n
            nabla_w_mult[k] = nabla_w_mult[k] / n

        nabla_w_mult = self.w_assign_zero(self.weights, nabla_w_mult)

        # backward pass - Method one - batch
        '''delta = self.cost_derivative(activations[-1], y) # * sigmoid_prime(zs[-1])
        nabla_b[-1] = self.aver_1(delta)
        #nabla_w[-1] = np.dot(delta, activations[-2].transpose())
        t = np.zeros((sampleNum, np.shape(activations[-2].transpose())[1] ))
        for i in range(np.shape(delta)[1]):
            t[i] = np.dot(delta[0][i], (activations[-2].transpose())[i,:])    
        nabla_w[-1] = self.aver_2(t)
        nabla_w[-1] = np.reshape(nabla_w[-1], (1,len(nabla_w[-1])))

        for l in xrange(2, self.num_layers):
            z = zs[-l]
            sp = sigmoid_prime(z)
            delta = np.dot(self.weights[-l+1].transpose(), delta) * sp # Hadamard product
            nabla_b[-l] = self.aver_1(delta)

            t = [[] for _ in range(np.shape(delta)[1])]
            for j in range(np.shape(delta)[1]):
                t[j] = np.reshape(delta[...,j],(len(delta[...,j]),1)) * (activations[-l-1].transpose())[j] 
            nabla_w[-l] = self.aver_2(t)

        nabla_w = self.transfrom(self.weights, nabla_w)'''

        return (nabla_b_mult, nabla_w_mult)

    def feed_forward(self, x, y):
        # Calculate final predict

        # For layer 0
        i = 0
        # Initialization the linear part coefficients
        h_hat, b_intercept = self.h_matrix_init(self.sizes)

        # For layer i
        for b, w in zip(self.biases[:-1], self.weights[:-1]):
            # Calculate x_s
            x_s, h_hat[i], b_intercept[i] = self.epsilon_gen(
                        x, self.index_parent[i], h_hat[i], b_intercept[i])
            x = self.sigmoid(np.dot(w, x_s) + b)
            i += 1

        # For layer L
        x_s, h_hat[-1], b_intercept[-1] = self.epsilon_gen(
                        x, self.index_parent[-1], h_hat[-1], b_intercept[-1])

        # Get the final predict
        y_hat = np.dot(self.weights[-1], x_s) + self.biases[-1]

        mse = np.sum((y_hat - y) ** 2) / y.shape[1]

        return np.array([mse])

    def h_matrix_init(self, sizes):
        # H hat matrix initialization

        h_matrix = [[] for _ in range(len(sizes) - 1)]
        b_bias = [[] for _ in range(len(sizes) - 1)]

        for i in range(len(sizes) - 1):
            h_upper = np.identity(sizes[i])
            h_lower = np.identity(sizes[i])
            h_matrix[i] = np.concatenate((h_upper, h_lower), axis=0)
            b_bias[i] = np.zeros((2 * sizes[i], 1))

        return h_matrix, b_bias

    def epsilon_gen(self, x, i_p, h_mat, b_intercept):
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
                if self.residual_set==True:
                    regr.fit(a, b)  # (x, y)
                    # residues: b - regr.predict(a); regr.coef_; regr.intercept_
                    # Since next following operation is X-x_hat,
                    # So the corresponding coefficient will be negative
                    h_mat[node_number + i][i_p[i]] = -regr.coef_
                    b_intercept[node_number + i] = -regr.intercept_
                elif self.residual_set==False:
                    h_mat[node_number + i][i_p[i]] = 1
                    b_intercept[node_number + i] = 0

        x_epsilon = np.dot(h_mat, x) + b_intercept
        return x_epsilon, h_mat, b_intercept

    def loss_derivative(self, output_activation, y):
        return (output_activation - y)

    def w_assign_zero(self, weights, nabla_w):
        """return add weighted matrix"""
        for a, b in zip(self.weights, nabla_w):
            for i in range(np.shape(a)[0]):
                for j in range(np.shape(a)[1]):
                    if a[i][j] == 0:
                        b[i][j] = 0
        return nabla_w

    def evaluate(self, mini_batch):
        # calculate loss
        x, y = np.transpose(mini_batch[0]), np.transpose(mini_batch[1])
        mse = self.feed_forward(x, y)
        return mse

    def sigmoid(self, z):
        return np.tanh(z)

    def sigmoid_prime(self, z):
        return 1.0 - np.tanh(z) ** 2

    def load_checkpoint(self, checkpoint_path):
        # It's weird that if `map_location` is not given, it will be extremely slow.
        return torch.load(checkpoint_path, map_location=lambda storage, loc: storage)

    def test_main(self, test_data, loss_validation, sizes, optimal_epoch_given, LOAD_CHECKPOINT = True):
        # return test loss

        # Select the best parameters in the model
        if optimal_epoch_given == False:
            optimal_epoch = loss_validation.index(min(loss_validation))+1
        else:
            optimal_epoch = optimal_epoch_given

        if LOAD_CHECKPOINT:
            # load from path.
            checkpoint_path = os.path.join('model_checkpoint_train/', "SDL_epoch_%d.pt" % optimal_epoch)
            checkpoint = self.load_checkpoint(checkpoint_path)

            # load needed parameters
            w_hat = checkpoint['w']
            b_hat = checkpoint['biases']
            epoch = checkpoint['epoch']
            lr = checkpoint['learning_rate']

            # Organize data
            test_data = self.data_organize(list(test_data))

            # Predict test results
            true_y, predict_y, loss_test = self.test_prediction(test_data, w_hat, b_hat, sizes)

            m = test_data[0].shape[1]
            n = test_data[0].shape[0]

            if not os.path.exists('model_checkpoint_test'):
                os.makedirs('model_checkpoint_test')

            # Check points
            testing_checkpoint = {
                'weight': w_hat,
                'biases': b_hat,
                'learning_rate': lr,
                'sizes': sizes,
                'optimal_epoch': optimal_epoch,
                'test_loss': loss_test,
                'true_y': true_y,
                'predict_y': predict_y,
                'm': m,
                'n': n
            }

            testing_model_name = '%s%d_%s%d_%s%d%s' % (
                'model_checkpoint_test/SDL_test_m_', m,
                'n_', n,
                'optimal_', optimal_epoch,
                '.pt'
            )

            # Need change the name of saving the model parameter name
            torch.save(testing_checkpoint, testing_model_name)
            print('Optimal epoch: %d' % optimal_epoch)
            print('Test loss: %.4f\n' % loss_test)
            return loss_test
        else:
            print('Need checkpoint, please try again!')

    def test_prediction(self, test_data, w_hat, b_hat, sizes):
            # Return y, predict_y, and loss of test data
            true_y, predict_y, loss_test = self.test_feed_forward(np.transpose(test_data[0]), np.transpose(test_data[1]),w_hat, b_hat, sizes)
            return true_y, predict_y, loss_test

    def test_feed_forward(self, x, y, w_hat, b_hat, sizes):

        # For layer 0
        i = 0
        # Initialization the linear part coefficients
        h_hat, b_intercept = self.h_matrix_init(sizes)

        # For layer i
        for b, w in zip(b_hat[:-1], w_hat[:-1]):
            # Calculate x_s
            x_s, h_hat[i], b_intercept[i] = self.epsilon_gen(
                        x, self.index_parent[i], h_hat[i], b_intercept[i])
            x = self.sigmoid(np.dot(w, x_s) + b)
            i += 1

        # For layer L
        x_s, h_hat[-1], b_intercept[-1] = self.epsilon_gen(
                        x, self.index_parent[-1], h_hat[-1], b_intercept[-1])

        # Get the final predict
        y_hat = np.dot(w_hat[-1], x_s) + b_hat[-1]

        mse = np.sum((y_hat - y) ** 2)/y.shape[1]

        return y, y_hat, np.array([mse])

    # def sigmoid(z):
    #    return 1.0/(1.0+np.exp(-z))

    # def sigmoid_prime(z):
    #    return sigmoid(z)*(1-sigmoid(z))

