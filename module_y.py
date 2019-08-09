import random
import numpy as np
import math



def select_nonzero_index(s, L, m, structure, choose_ratio = 0.01):
    sample = []
    choose = m * choose_ratio
    layer = [[] for _ in range(L)]
    if structure == 'Top-Down':
        a = s ** 0
        b = a + s ** 1
        # every layer choose equal number
        ch = []
        for i in range(L):
            if i == L - 1:
                ch.append(choose - i * choose / L)
                break
            ch.append(choose / L)
        # every layer number range
        for i in range(L):
            layer[i] = np.random.choice(range(a, b), int(ch[i]), replace=False)
            if b == m:
                break
            a = b
            b = a + s ** (i + 2)
            if b > m:
                b = m

    if structure == 'Bottom-Up':
        L = int(math.log(m, s))
        a = 0
        b = a + s ** int(math.log(m, s))
        ch = []
        ch.append(choose - (choose / L * (L - 1)))
        for i in range(1, L):
            ch.append(choose / L)

        for i in reversed(range(L)):
            layer[L - i - 1] = np.random.choice(range(a, b), int(ch[L - i - 1]), replace=False)
            if b == m - 1:
                break
            a = b
            b = a + s ** i

    # reshape sample to one array
    for i in range(len(layer)):
        sample = np.concatenate((sample, layer[i]), axis=0)
    sample = sample.astype(int).tolist()
    sample = [1] + sample
    return sample


def generate_y(x, epsilon, nonzero_ratio, structure, relation, s, L):
    # x: data matrix, m*n
    # epsilon: random effect matrix, m*n
    # nonzero_num: number of nonzero predictors
    # structure: "Top-Down" or "Bottom-Up"
    # relation: "Linear" or "Sin"
    # s: 2, 5, or 10, binary, quinary, decimal
    # L: Hierarchy layers (tree)

    np.random.seed(12345)

    m = np.shape(x)[0]
    n = np.shape(x)[1]
    true_parameter = np.zeros((m, 1))

    nonzero_num = int(m * nonzero_ratio)
    # Choice 1: Easy model selection
    # sample =  np.random.choice(m, m*0.01,replace = False)

    # Choice 2: Equal model selection (every layer selects equal number of nonzero)
    sample = select_nonzero_index(s, L, m, structure, nonzero_ratio)
    for i in range(nonzero_num):
        true_parameter[sample[i]] = 1

    # noise part settings
    noise_part = np.random.normal(0, np.sqrt(nonzero_num))

    # Choice 1: generate y = x * beta + noise
    y = np.dot(np.transpose(x), true_parameter) + noise_part
    y = np.reshape(y, (1, len(y)))[0]
    y = (y - np.mean(y)) / np.std(y)

    filename_y = '%s%s_%d_%d_%d_%s_%s_%d_%d%s' % ('data/','y', m, n, nonzero_num, structure, relation, s, L, '.txt')
    np.savetxt(filename_y, y)

    # Choice 2: use epsilon as input
    '''
    epsilon = np.transpose(epsilon) #10000 * 15
    y_epsilon = np.dot(epsilon, true_parameter) + b
    t = np.reshape(y_epsilon, (1, len(y_epsilon)))
    y_epsilon = t[0]

    filename_epsilon = '%s%s_%d_%d_%d_%s_%d_%d_%s%s' % ('data/', 'y_epsilon',m,n,select,structure,s,L,relation,'.txt')
    np.savetxt(filename_epsilon, y_epsilon)

    plt.figure(1)
    plt.hist(y, bins=50)
    plt.savefig("sim_data/y.png")
    '''
    
    print("Complete Y generation!")

    return y, sample, filename_y  # ,y_epsilon






