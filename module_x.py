import random
import numpy as np
import math


class Utils(object):
    def __init__(self, m, s, structure, L):
        self.m = m
        self.s = s
        self.structure = structure
        self.L = L

    # Create adjacent matrix
    def relation_matrix(self, m, s, structure):
        x_relation = np.zeros((m, m))

        if self.structure == 'Top-Down':
            for i in range(m):
                x_relation[i][int(s * i + 1):int(s * (i + 1) + 1)] = 1

        if self.structure == 'Bottom-Up':
            # 2:1023   5:781   10: 1111
            numberoflevel = int(math.log(m, self.s)) + 1
            level = [[] for _ in range(numberoflevel)]
            j = 0
            for i in range(len(level)):
                level[i] = range(j, j + self.s ** (numberoflevel - i - 1))
                j = j + self.s ** (numberoflevel - i - 1)
            num = []
            for i in range(1, len(level)):
                for j in range(len(level[i])):
                    for k in range(self.s):
                        num.append(level[i][j])
            for i in range(m - 1):
                x_relation[i][num[i]] = 1

        return x_relation

    # Create Child_parent_relation dict
    def child_parent_relation(self, adjacent_matrix):
        m = len(adjacent_matrix)
        child_parent = {}
        for j in range(m):
            child_parent[j] = []
            for i in range(m):
                if adjacent_matrix[i][j] == 1:
                    child_parent[j].append(i)

        return child_parent

    # Create Parent_child_relation dict
    def parent_child_relation(self, adjacent_matrix):
        m = len(adjacent_matrix)
        parent_child = {}
        for i in range(m):
            parent_i = []
            parent_child.update({i: parent_i})
            for j in range(m):
                if adjacent_matrix[i][j] == 1:
                    parent_i.append(j)
        return parent_child

    # delete use for 'Top-Down'
    def delete_dict(self, parent_child):
        m = len(parent_child)
        count = []
        for i in reversed(range(m)):
            if len(parent_child[i]) == 0:
                count.append(i)
        for key in range(m):
            if key in count:
                del parent_child[key]
        delete_number = len(count)

        return count, parent_child, delete_number

    # delete use for 'Bottom-Up'
    def dele(self, count, parent_child, child_parent):
        m = len(parent_child)
        deleparentnumber = []
        for i in range(m):
            if parent_child[i] == []:
                deleparentnumber.append(child_parent[i])
        for i in range(m):
            if parent_child[i] == []:
                del parent_child[i]
        deleparentnumber = list(set.union(*map(set, deleparentnumber)))
        for key in deleparentnumber:
            for i in range(len(parent_child)):
                if i == key:
                    parent_child[i] = []
        return deleparentnumber, parent_child

    def layer_neuron_num(self, adjacent_matrix, s, structure, L):
        layer_neuron_number = np.zeros(self.L + 1)
        if structure == 'Top-Down':
            m = len(adjacent_matrix)
            layer_neuron_number[0] = self.m
            for i in range(1, L):
                x_relation = self.relation_matrix(m, s, structure)
                parent_child = self.parent_child_relation(x_relation)
                count, parent_child, delete_number = self.delete_dict(parent_child)
                m = len(parent_child)
                layer_neuron_number[i] = len(parent_child)
            layer_neuron_number[L] = 1
            layer_neuron_number = layer_neuron_number.astype(np.int64)

        if structure == 'Bottom-Up':
            m = len(adjacent_matrix)
            layer_neuron_number[0] = m
            parent_child = self.parent_child_relation(adjacent_matrix)
            child_parent = self.child_parent_relation(adjacent_matrix)
            count = []
            count, parent_child = self.dele(count, parent_child, child_parent)
            layer_neuron_number[1] = len(parent_child)
            for i in range(2, L):
                count, parent_child = self.dele(count, parent_child)
                layer_neuron_number[i] = len(parent_child)
            layer_neuron_number[L] = 1
            layer_neuron_number = layer_neuron_number.astype(np.int64)

        return layer_neuron_number


def generate_x(number_of_node, number_of_sample, structure, gamma, relation, s, L):
    # generate X (data: n * m) and correspond epsilon
    # m (col): predictor_size
    # n (row): sample size
    # structure: Top-Down , Bottom-Up, initial Captain
    # gamma: parameter of random effect
    # relation: Linear, Sin (parent nodes - children nodes)
    # s: folds size 2, 5, 10
    # L: hierarchy layer numbers

    m = number_of_node
    n = number_of_sample
    mu, sigma = 0, 1

    # utils function
    util_fun = Utils(m, s, structure, L)

    # generate adjacent_matrix matrix (0,1)
    adj_matrix = util_fun.relation_matrix(m, s, structure)

    # Convert from adj_mat: [dict] {child index:[parent]}
    child_parent = util_fun.child_parent_relation(adj_matrix)

    # Convert from adj_mat: [dict] {parent index:[child index]}
    parent_child = util_fun.parent_child_relation(adj_matrix)

    # Compute hierarchy each layer's predictor numbers
    layer_predictor_number = util_fun.layer_neuron_num(adj_matrix, s, structure, L)


    #create data
    x = np.zeros((m, n))
    epsilon = np.zeros((m, n))
    b = np.zeros(m)

    def level(child_parent):
        lack_parent_number = 0
        for i in range(len(child_parent)):
            if len(child_parent[i]) < 1:
                lack_parent_number = lack_parent_number + 1
        return lack_parent_number

    def interaction_term_x(n, gamma, parent_number, x, i, child_parent):
        # we use 2 interaction_term: i.e. x1-x2
        interaction = np.zeros((1, n))
        for a in child_parent[i]:
            for b in child_parent[i]:
                if a < b:
                    interaction = interaction + x[child_parent[i][a]][:] * x[child_parent[i][b]][:]
        interaction = np.sqrt((1 - np.power(gamma, 2)) / parent_number) * interaction
        return interaction

    # Top-Down tree or Bottom-Up tree
    if structure == 'Top-Down':

        initialization = np.random.normal(0, 1, n)

        x[0, :] = initialization
        epsilon[0, :] = initialization  # The top parent's epsilon is himself

        if relation == 'Linear':

            for i in range(m):
                parent_number = len(child_parent[i])

                if parent_number == 0:
                    continue
                else:
                    epsilon[i][:] = np.random.normal(0, np.sqrt(gamma), n)
                    for j in range(parent_number):
                        x[i][:] = x[i][:] + np.sqrt((1 - gamma) / parent_number) * x[child_parent[i][j]][:]
                    x[i][:] = x[i][:] + epsilon[i][:]

        elif relation == 'Sin':  # variance of sin = 15/36,x1 = sin(x0)

            x[0][:] = np.random.uniform(-np.pi, np.pi, n)
            b = np.random.uniform(-np.pi, np.pi, m)
            for i in range(m):
                parent_number = len(child_parent[i])

                if parent_number == 0:
                    continue
                else:
                    epsilon[i][:] = np.random.normal(0, np.sqrt(gamma), n)
                    for j in range(parent_number):
                        k = np.sqrt((1 - gamma) / (parent_number * np.var(np.sin(x[child_parent[i][j]][:] + b[i]))))
                        x[i][:] = x[i][:] + k * np.sin(x[child_parent[i][j]][:] + b[i])

                    x[i][:] = x[i][:] + epsilon[i][:]

    elif structure == 'Bottom-Up':

        lack_parent_number = level(child_parent)
        initialization = np.random.normal(mu, sigma, (lack_parent_number, n))

        x[0:lack_parent_number, :] = initialization
        epsilon[0:lack_parent_number, :] = initialization  # The Top is parent

        if relation == 'Linear':

            for i in range(lack_parent_number, m):
                parent_number = len(child_parent[i])
                epsilon[i][:] = np.random.normal(0, np.sqrt(gamma), n)

                for j in range(parent_number):
                    x[i][:] = x[i][:] + np.sqrt((1 - gamma) / parent_number) * x[child_parent[i][j]][:]
                x[i][:] = x[i][:] + epsilon[i][:]

        elif relation == 'Sin':

            x[0:lack_parent_number, :] = np.random.normal(mu, sigma, (lack_parent_number, n))
            b = np.random.uniform(-np.pi, np.pi, m)
            for i in range(lack_parent_number, m):
                parent_number = len(child_parent[i])
                epsilon[i][:] = np.random.normal(0, np.sqrt(gamma), n)

                for j in range(parent_number):
                    k = np.sqrt((1 - gamma) / (parent_number * np.var(np.sin(x[child_parent[i][j]][:] + b[i]))))
                    x[i][:] = x[i][:] + k * np.sin(x[child_parent[i][j]][:] + b[i])

                x[i][:] = x[i][:] + epsilon[i][:]

        elif relation == 'Interaction':  # Top-Down structure does not have interaction
            # variance has problem

            for i in reversed(range(m - lack_parent_number)):
                parent_number = len(child_parent[i])
                parent_number = parent_number * (parent_number - 1) / 2
                epsilon[i][:] = gamma * np.random.normal(0, 1, n)

                for j in reversed(range(parent_number)):  # linear part
                    x[i][:] = x[i][:] + np.sqrt((1 - np.power(gamma, 2)) / parent_number) * x[child_parent[i][j]][:]

                interaction = interaction_term_x(n, gamma, parent_number, x, i, child_parent)
                x[i][:] = x[i][:] + interaction + epsilon[i][:]

    filename_x = '%s%s_%d_%d_%s_%.1f_%s_%d_%d%s' % ('data/', 'x', m, n, structure, gamma, relation, s, L, '.txt')
    np.savetxt(filename_x, x)

    print("Complete X generation!")
    return x, epsilon, adj_matrix, child_parent, parent_child, layer_predictor_number, filename_x


