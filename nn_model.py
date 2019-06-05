import random
import numpy as np
import utils as ut


class NNModel:
    def __init__(self,in_dim, hid_dim, out_dim):
        root_six = np.sqrt(6)
        eps = root_six / (np.sqrt(hid_dim + in_dim))
        self._W = np.random.uniform(-eps, eps, [hid_dim, in_dim])
        eps = root_six / (np.sqrt(hid_dim))
        self._b = np.random.uniform(-eps, eps, hid_dim)
        eps = root_six / (np.sqrt(out_dim + hid_dim))
        self._U = np.random.uniform(-eps, eps, [out_dim, hid_dim])
        eps = root_six / (np.sqrt(out_dim))
        self._b_tag = np.random.uniform(-eps, eps, out_dim)

    def train_nn(self, train_x, train_y, dev_x, dev_y, EPOCHS, learning_rate):
        """
        Create and train a classifier, and return the parameters.

        train_data: a list of (label, feature) pairs.
        dev_data  : a list of (label, feature) pairs.
        num_iterations: the maximal number of training iterations.
        learning_rate: the learning rate to use.
        params: list of parameters (initial values)
        """
        costs = []
        acc = []
        for epoch in range(EPOCHS):
            total_loss = 0.0  # total loss in this iteration.
            random.shuffle(train_x)
            for X, Y in zip(train_x,train_y):

                loss, grads = self.loss_and_gradients(X, Y)
                total_loss += loss
                # update the parameters according to the gradients
                # and the learning rate.
                self._W -= learning_rate * grads[0]
                self._b -= learning_rate * grads[1]
                self._U -= learning_rate * grads[2]
                self._b_tag -= learning_rate * grads[3]

            train_loss = total_loss / len(train_x)
            costs.append(train_loss)
            train_accuracy = (self.accuracy_on_dataset(train_x,train_y))
            dev_accuracy = (self.accuracy_on_dataset(dev_x,dev_y))
            acc.append((train_accuracy, dev_accuracy))
            print(epoch, train_loss, train_accuracy, dev_accuracy)
        # fig = plt.plot(acc)
        # fig1 = plt.plot(costs)

    def accuracy_on_dataset(self, data_x, data_y):
        good = bad = 0.0
        for X, Y in zip(data_x, data_y):
            # Compute the accuracy (a scalar) of the current parameters
            # on the dataset.
            # accuracy is (correct_predictions / all_predictions)
            pred = self.predict(X)
            if pred == (Y*255.0):
                good += 1
            else:
                bad += 1
            pass
        return good / (good + bad)

    def loss_and_gradients(self, x, y):
        """
        params: a list of the form [W, b, U, b_tag]

        returns:
            loss,[gW, gb, gU, gb_tag]

        loss: scalar
        gW: matrix, gradients of W
        gb: vector, gradients of b
        gU: matrix, gradients of U
        gb_tag: vector, gradients of b_tag
        """
        y_hat = self.classifier_output(x)  # probabilities vec
        y_one_hot = ut.turn_tag_to_y(y)  # create one-hot vector
        h = ut.ReLU(np.dot(self._W, x) + self._b)
        # gradients values
        gb_tag = (y_one_hot - y_hat)
        gU = np.outer(gb_tag, h)
        gb = np.dot(gb_tag, self._U) * (h - np.square(h))
        gW = np.outer(gb, x)

        y = y*255.0

        loss = -np.log(y_hat[int(y)])
        return loss, [gW, gb, gU, gb_tag]

    def classifier_output(self, x):
        classification_vec = np.dot(self._U,(ut.ReLU(np.dot(self._W,x)+self._b)))+self._b_tag
        probs = ut.softmax(classification_vec)
        return probs

    def predict(self, x):
        return np.argmax(self.classifier_output(x))

    def test(self,test_file):
        fd = open("test_y", 'w')
        counter = 0
        test_ans = ''
        test_data = ut.read_data(test_file)
        for X in test_data:
            pred = self.predict(X)
            fd.write(pred + "\n")
            # print 'line: ', counter, 'prediction: ', test_ans
        fd.close()
