import numpy as np
import pandas as pd
import itertools


# Use Adjacent matrix to build child_parent, parent_child
class GetInfoRelation(object):

    def __init__(self, adjacent_matrix):
        self.adjacent_matrix = adjacent_matrix

    def child_parent_relation(self, adjacent_matrix):
        m = len(self.adjacent_matrix)
        child_parent = {}
        for j in range(m):
            child_parent[j] = []
            for i in range(m):
                if self.adjacent_matrix[i][j] == 1:
                    child_parent[j].append(i)
        return child_parent

    def parent_child_relation(self, adjacent_matrix):
        if type(self.adjacent_matrix) == pd.core.frame.DataFrame:
            parent_child = {}
            for i in self.adjacent_matrix.index:
                parent_i = []
                parent_child.update({i: parent_i})
                for j in self.adjacent_matrix.columns:
                    if self.adjacent_matrix.loc[i][j] == 1:
                        parent_i.append(j)
            return parent_child
        else:
            m = len(self.adjacent_matrix)
            parent_child = {}
            for i in range(m):
                parent_i = []
                parent_child.update({i: parent_i})
                for j in range(m):
                    if self.adjacent_matrix[i][j] == 1:
                        parent_i.append(j)
            return parent_child


class GetInfoLayer(object):
    def __init__(self, adjacent_matrix, parent_child, max_layer):
        self.adjacent_matrix = adjacent_matrix
        self.parent_child = parent_child
        self.max_layer = max_layer

    def child_parent_relation(self, adjacent_matrix):
        m = len(adjacent_matrix)
        child_parent = {}
        for j in range(m):
            child_parent[j] = []
            for i in range(m):
                if adjacent_matrix[i][j] == 1:
                    child_parent[j].append(i)
        return child_parent

    def parent_child_relation(self, adjacent_matrix):
        if type(adjacent_matrix) == pd.core.frame.DataFrame:
            parent_child = {}
            for i in adjacent_matrix.index:
                parent_i = []
                parent_child.update({i: parent_i})
                for j in adjacent_matrix.columns:
                    if adjacent_matrix.loc[i][j] == 1:
                        parent_i.append(j)
            return parent_child

        else:
            m = len(adjacent_matrix)
            parent_child = {}
            for i in range(m):
                parent_i = []
                parent_child.update({i: parent_i})
                for j in range(m):
                    if adjacent_matrix[i][j] == 1:
                        parent_i.append(j)
            return parent_child

    def layer_node_number(self, adjacent_matrix, parent_child, max_layer):
        # Get each layer's node number

        # The whole layer number is max_layer + 1(final layer)
        layer_node_number = np.zeros(max_layer + 1)
        layer_node_number[0] = adjacent_matrix.shape[0]

        # Store each layer's node index
        layer_of_node = [[] for _ in range(max_layer + 1)]
        layer_of_node[0] = list(parent_child.keys())

        # m: dimensionality of input adjacent matrix
        p_c_temp = parent_child

        # Begin from 1 hidden layer
        for i in range(1, max_layer):

            # Count each layer node index
            count, p_c_temp, delete_number = self.delete_dict(p_c_temp)
            p_c_temp = self.update_p_c(p_c_temp)
            adjacent_matrix_update = self.update_adj_matrix(p_c_temp)
            p_c_temp = self.parent_child_relation(adjacent_matrix_update)

            # Update list
            layer_node_number[i] = len(p_c_temp)
            layer_of_node[i] = list(p_c_temp.keys())

        # Final layer
        layer_node_number[max_layer] = 1
        layer_of_node[max_layer] = [0]

        layer_node_number = layer_node_number.astype(np.int64)
        return layer_node_number, layer_of_node

    def update_adj_matrix(self, p_c_temp):
        # Update adjacent matrix by each layer
        indexs = list(p_c_temp.keys())

        # Column: next layer node index
        column = list(set([item for sublist in list(p_c_temp.values()) for item in sublist]))
        adjacent_matrix_temp = np.zeros((len(indexs), len(column)))
        adjacent_matrix_temp = pd.DataFrame(adjacent_matrix_temp, index=indexs, columns=column)

        for i in range(len(p_c_temp)):
            temp_x = list(p_c_temp.keys())[i]
            temp_y = p_c_temp[temp_x]
            adjacent_matrix_temp.loc[temp_x][temp_y] = 1

        return adjacent_matrix_temp


    def delete_dict(self, p_c_temp):
        count = []
        for i in list(p_c_temp.keys()):
            if len(p_c_temp[i]) == 0 and i in set(
                    list(itertools.chain(*filter(None, list(p_c_temp.values()))))):
                count.append(i)
        for key in list(p_c_temp.keys()):
            if key in count:
                del p_c_temp[key]

        delete_number = len(count)
        return count, p_c_temp, delete_number


    def update_p_c(self, p_c_temp):
        check_1 = list(set(p_c_temp.keys()))
        for i in check_1:
            for j in reversed(range(len(p_c_temp[i]))):
                if p_c_temp[i][j] not in p_c_temp.keys():
                    del p_c_temp[i][j]
        return p_c_temp

def load_data(file_name):
    file_name = 'PROPEL3_gene.txt'
    data = pd.read_csv(file_name)

    return data

def get_info_x(adjacent_matrix, max_layer):

    # Get parent_child and child_parent relationship
    info_relation = GetInfoRelation(adjacent_matrix)
    parent_child = info_relation.parent_child_relation(adjacent_matrix)
    child_parent = info_relation.child_parent_relation(adjacent_matrix)

    # Get each layer node index and node number
    info_layer = GetInfoLayer(adjacent_matrix, parent_child, max_layer)
    layer_node_number, layer_of_node = info_layer.layer_node_number(adjacent_matrix, parent_child, max_layer)

    return parent_child, child_parent, layer_node_number, layer_of_node


