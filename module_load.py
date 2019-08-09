import numpy as np


def load_data(name_x, name_y, n_train, n_val, n_test):
    x = np.loadtxt(name_x)
    y = np.loadtxt(name_y)

    n = np.shape(x)[1]
    y = np.reshape(y, (1, len(y)))

    training_x = x[:, range(0, n - n_val - n_test)]
    validation_x = x[:, range(n - n_val - n_test, n - n_test)]
    test_x = x[:, range(n - n_test, n)]

    return training_x, validation_x, test_x, y, n


def load_data_wrapper(name_x, name_y, n_train, n_val, n_test):
    tr_x, va_x, te_x, y, n = load_data(name_x, name_y, n_train, n_val, n_test)

    tr_y = y[:, range(0, n - n_val - n_test)]
    va_y = y[:, range(n - n_val - n_test, n - n_test)]
    te_y = y[:, range(n - n_test, n)]

    training_data = zip(np.transpose(tr_x), np.transpose(tr_y))
    validation_data = zip(np.transpose(va_x), np.transpose(va_y))
    test_data = zip(np.transpose(te_x), np.transpose(te_y))

    return list(training_data), list(validation_data), list(test_data)
