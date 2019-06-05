# Noa Or
# 208385534

import numpy as np
import math

sigmoid = lambda (x): 1 / (1 + np.exp(-x))

soft_max = lambda(x): np.exp(x -np.max(x)) / (np.exp(x - np.max(x))).sum()


# This function turns the y tag to a vector.
def turn_tag_to_y(tag):
    vec = np.zeros(10)
    vec[int(tag * 255.0)] = 1
    return vec


# this function calculates the negative log liklihood.
def loss(y, y_hat):
    total_sum = 0
    for i in xrange(len(y_hat)):
        temp = y[i] * np.log(y_hat[i])
        total_sum += temp
    return -1*total_sum


# forward func
def fprop(x, params):
    # Follows procedure given in notes
    w1, b1, w2, b2 = [params[key] for key in ('w1', 'b1', 'w2', 'b2')]
    z1 = np.dot(w1, x) + b1
    h1 = sigmoid(z1)
    z2 = np.dot(w2, h1) + b2
    # y hat
    y_hat = soft_max(z2)
    ret = {'w1': w1, 'b1': b1, 'z1': z1, 'h1': h1,
           'w2': w2, 'b2': b2, 'z2': z2, 'y_hat': y_hat}
    return ret


# backward func
def bprop(dictionary, y_vec, x):
    w1, b1, z1, h1, w2, b2, z2, y_hat = [dictionary[key] for key in('w1', 'b1', 'z1', 'h1', 'w2', 'b2', 'z2', 'y_hat')]
    dz2 = (y_hat - y_vec)
    dW2 = np.dot(dz2, h1.T)
    db2 = dz2
    dz1 = np.dot(w2.T, db2) * sigmoid(z1) * (1-sigmoid(z1))
    dW1 = np.dot(dz1, x.T)
    db1 = dz1
    return {'b1': db1, 'w1': dW1, 'b2': db2, 'w2': dW2}


# THis function updated the parameters by SGD update rule.
def update_parameters(params, gradients, eta):
    w1, b1, w2, b2 = [params[key] for key in ('w1', 'b1', 'w2', 'b2')]
    gw1, gb1, gw2, gb2 = [gradients[key] for key in ('w1', 'b1', 'w2', 'b2')]
    w1_new = w1 - eta * gw1
    b1_new = b1 - eta * gb1
    w2_new = w2 - eta * gw2
    b2_new = b2 - eta * gb2
    return {'w1':w1_new, 'b1':b1_new, 'w2':w2_new, 'b2':b2_new}


# This function gets the index of the biggest node in x
def get_max_index(x):
    max1 = max(x)
    for i in range(0, len(x)):
        if x[i] == max1:
            return i
    return 0


# This function checks the validation set
def predict_on_dev(params, dev_x, dev_y):
    sum_loss = 0.0
    correct = 0
    for x, y in zip(dev_x, dev_y):
        dictionary = fprop(x.reshape((784, 1)), params)
        y_hat = dictionary['y_hat']
        y_vec = turn_tag_to_y(y)
        y_vec.shape = (y_hat.shape[0], 1)
        sum_loss += loss(y_vec, y_hat)
        max_vec = y_hat.argmax(axis=0)
        # if get_max_index(y_hat) == y:
        if max_vec[0] == y * 255:
            correct += 1
    accurate = correct / float(np.shape(dev_x)[0])
    loss_avg = sum_loss / float(np.shape(dev_x)[0])
    return loss_avg, accurate


# this function shuffles the x, y data sets
def shuffle(a, b):
    i = np.arange(a.shape[0])
    np.random.shuffle(i)
    return a[i], b[i]


# this function trains the model
def train(params, num_ephocs, eta, train_x, train_y, dev_x, dev_y):
    for i in xrange(num_ephocs):
        sum_loss = 0
        train_x, train_y = shuffle(train_x, train_y)
        for x, y in zip(train_x, train_y):
            dictionary = fprop(x.reshape((784, 1)), params)
            y_hat = dictionary['y_hat']
            y_vec = turn_tag_to_y(y)
            y_vec.shape = (y_hat.shape[0], 1)
            sum_loss += loss(y_vec, y_hat)
            # calculate the gradients
            gradients = bprop(dictionary, y_vec, x.reshape((784, 1)))
            update = update_parameters(params, gradients, eta)
            for key in params:
                params[key] = update[key]
        dev_loss, accurate = predict_on_dev(params, dev_x, dev_y)
        loss_avg = sum_loss / float(np.shape(train_x)[0])
        print i, loss_avg, "[", dev_loss[0], "] {}%".format(accurate * 100)
    return params


def main():

    # loading the data
    train_x = np.loadtxt("train_x")
    train_x /= 255.0
    train_y = np.loadtxt("train_y")
    train_y /= 255.0
    test_x = np.loadtxt("test_x")
    test_x /= 255.0

    train_size = len(train_x)
    # doing shuffle to x and y together so they will be match
    np.random.shuffle(zip(train_x, train_y))
    # leaving 20% of the data for validation
    dev_size = int(train_size * 0.2)
    dev_x = train_x[-dev_size:, :]
    dev_y = train_y[-dev_size:]
    train_x, train_y = train_x[:-dev_size, :], train_y[:-dev_size]

    # Initialize random parameters and inputs
    input_size = 28 * 28
    hidden_size = 100
    output_size = 10

    w1 = np.random.uniform(-0.08, 0.08, [hidden_size, input_size])
    b1 = np.random.uniform(-0.08, 0.08, [hidden_size, 1])
    w2 = np.random.uniform(-0.08, 0.08, [output_size, hidden_size])
    b2 = np.random.uniform(-0.08, 0.08, [output_size, 1])
    params = {'w1': w1, 'b1': b1, 'w2': w2, 'b2': b2}

    num_ephocs = 48
    eta = 0.005

    updated_params = train(params, num_ephocs, eta, train_x, train_y, dev_x, dev_y)

    pred = open("test.pred", "wr")
    for x in test_x:
        prediction = fprop(x.reshape((input_size, 1)), updated_params)
        y_hat = prediction['y_hat'].argmax(axis=0)
        pred.write(str(y_hat[0]) + '\n')
    pred.close()


if __name__ == '__main__':
    main()
