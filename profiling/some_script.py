import numpy as np

data = np.random.randn(1000, 500)
operator = np.random.randn(500, 500)


def inner_function(data):
    data -= np.median(data, axis=1)[:, np.newaxis]
    rows_sums = data.sum(axis=1)
    data[rows_sums < 0] *= 2
    data += np.random.randn(*data.shape)


def some_function(data, n_iter=10):
    for i in range(n_iter):
        inner_function(data)
        data = np.dot(data, operator)


if __name__ == '__main__':
    some_function(data)
