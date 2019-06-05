# This file provides code which you may or may not find helpful.
# Use it if you want, or ignore it.
import random
import numpy as np
STUDENT={'name': 'Daniel Greenspan_Eilon Bashari',
         'ID': '308243948_308576933'}


def turn_tag_to_y(tag):
    vec = np.zeros(10)
    vec[int(tag * 255.0)] = 1
    return vec

def read_data(fname):
    return np.loadtxt(fname)


def split(array, nrows, ncols):
    """Split a matrix into sub-matrices."""

    r, h = array.shape
    return (array.reshape(h//nrows, nrows, -1, ncols)
                 .swapaxes(1, 2)
                 .reshape(-1, nrows, ncols))


def ReLU(x):
    return np.maximum(0.01*x, x)

def softmax(x):
    """
    Compute the softmax vector.
    x: a n-dim vector (numpy array)
    returns: an n-dim vector (numpy array) of softmax values
    """
    # YOUR CODE HERE
    # Your code should be fast, so use a vectorized implementation using numpy,
    # don't use any loops.
    # With a vectorized implementation, the code should be no more than 2 lines.
    #
    exp = np.exp(x - np.max(x))
    x = exp / exp.sum(axis=0)
    return x


def norm_data(data):
    return data / 255.0

if __name__ == '__main__':
    a = read_data("train_x")

