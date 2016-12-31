import numpy as np
from numpy.core.umath_tests import inner1d
import matplotlib.pyplot as plt

import os
import itertools

from mnist import MNIST
from layers import *

NUM_CLASSES = 10
NUMPY_TRAIN_SAVE_PATH = 'x_train.npy'
NUMPY_LABELS_SAVE_PATH = 'labels_train.npy'
NUMPY_TEST_SAVE_PATH = 'x_test.npy'

class NeuralNet:
    def __init__(self, layers, n_batch):
        '''
        layers: list - hidden layers followed by a single output (softmax) layer
        n_batch: batch size
        '''
        self.layers = layers
        self.n_batch = n_batch
        self.X_mean = None

    def preprocess_X(self, X):
        X_out = X / 255.0
        if self.X_mean is None:
            self.X_mean = np.mean(X_out, axis=0)
        X_out -= self.X_mean
        return np.concatenate((X_out, np.ones((X_out.shape[0], 1))), axis=1)

    def preprocess_labels(self, labels):
        labels_one_hot = np.zeros((labels.shape[0], NUM_CLASSES))
        labels_one_hot[range(len(labels)), labels] = 1
        return labels_one_hot

    def iterate(self, x, labels):
        '''
        x: <batch_size> by <n_input + 1>
        labels: <batch_size> by <n_out>
        :return:
        '''
        out = x
        for layer in self.layers:
            out = layer.forward(out)
        dJ_dout = labels
        for layer in reversed(self.layers):
            dJ_dout = layer.backward(dJ_dout)

    def train(self, X, labels, iterations, plot=False):
        print('Training for %s iterations with batch size %s' % (iterations, self.n_batch))
        X_processed = self.preprocess_X(X)
        labels_processed = self.preprocess_labels(labels)
        N = len(X_processed)

        calc_loss_thres = N # number of samples before calculating loss
        if plot and self.n_batch == 1:
            calc_loss_thres = 10000
        print_interval = iterations // 10
        epoch_count = 0
        accuracy_scores = []
        losses = []
        for i in range(iterations):
            indices = np.random.randint(0, N, self.n_batch)
            self.iterate(X_processed[indices], labels_processed[indices])
            if i % print_interval == 0:
                print('  %s iterations done' % (i,))
            if epoch_count > calc_loss_thres:
                labels_pred, J = self.predict(X_processed, labels_processed)
                score = evaluate(labels, labels_pred)
                losses.append(J)
                accuracy_scores.append(score)
                print('  Accuracy score: %.6f   Loss: %.6f' % (score, J))
                for layer in self.layers:
                    layer.decay_learning_rate()
                epoch_count -= calc_loss_thres
            epoch_count += self.n_batch
        if plot:
            for y_var, label in [(losses, 'loss'), (accuracy_scores, 'accuracy')]: # plot training loss and accuracy vs iterations
                plt.plot([i for i in range(len(y_var))], y_var)
                plt.title('Training %s vs iterations' % (label))
                plt.xlabel('%s of Iterations' % (calc_loss_thres))
                plt.ylabel(y_var)
                plt.show()

    def predict(self, X, labels=None):
        if labels is None: # we're predicting test data in this case
            X = self.preprocess_X(X)
        N = len(X)
        layers = [layer.get_clone(N) for layer in self.layers]
        out = X
        for layer in itertools.islice(layers, len(layers) - 1):
            out = layer.forward(out)
        return layers[-1].forward(out, predict_only=True, labels=labels)

def load_dataset(dir):
    if os.path.exists(dir + NUMPY_TRAIN_SAVE_PATH) and os.path.exists(dir + NUMPY_TEST_SAVE_PATH) and os.path.exists(dir + NUMPY_LABELS_SAVE_PATH):
        X_train = np.load(dir + NUMPY_TRAIN_SAVE_PATH)
        labels_train = np.load(dir + NUMPY_LABELS_SAVE_PATH)
        X_test = np.load(dir + NUMPY_TEST_SAVE_PATH)
    else:
        mndata = MNIST(dir)
        X_train, labels_train = map(np.array, mndata.load_training())
        X_test, _ = map(np.array, mndata.load_testing())
        X_train = np.array(X_train)
        labels_train = np.array(labels_train)
        X_test = np.array(X_test)
        np.save(dir + NUMPY_TRAIN_SAVE_PATH, X_train)
        np.save(dir + NUMPY_LABELS_SAVE_PATH, labels_train)
        np.save(dir + NUMPY_TEST_SAVE_PATH, X_test)
    return X_train, labels_train, X_test

def evaluate(labels_true, labels_pred):
    return np.sum(np.where(labels_true == labels_pred, 1, 0)) / len(labels_true)

def cross_validate(X, labels, nn_generator, iterations, fold=5, plot=False):
    N = len(X)
    indices = np.arange(N)
    np.random.shuffle(indices)
    labels_pred = np.zeros(N)
    labels_true = np.zeros(N)
    print('Starting cross validation')
    for i in range(fold):
        print('CV Iteration %s' % (i))
        start = i * N // fold
        end = (i + 1) * N // fold
        train_indices = np.concatenate((indices[0 : start], indices[end : N]))
        hold_indices = indices[start : end]
        X_train = X[train_indices]
        X_hold = X[hold_indices]
        labels_train = labels[train_indices]
        labels_hold = labels[hold_indices]

        nn = nn_generator()
        nn.train(X_train, labels_train, iterations, plot)
        labels_pred[start: end], J = nn.predict(X_hold)
        labels_true[start: end] = labels_hold
        score = evaluate(labels_true[start: end], labels_pred[start: end])
        print('Accuracy %.6f' % (score))
    score = evaluate(labels_true, labels_pred)
    print('Overall CV accuracy %.6f' % (score))
    return score

def create_generator_one_hidden(n_in, n_out):
    '''
    returns a single hidden layer neural net generator given the input and output dimensions
    contains settings for the neural net hyperparameters
    FC-ReLU-FC-softmax
    '''
    def generator():
        n_batch = 20
        n_hidden = 700
        relu_FC_init = lambda n_in, n_out: 2.0 / np.sqrt(n_in + n_out + 1)
        softmax_FC_init = lambda n_in, n_out: 1.0 / np.sqrt(n_in)

        # hidden layer
        FC_ReLU = FullyConnectedLayer(n_batch, n_in, n_hidden, learning_rate=5e-1, dropout_rate=0.3, decay_rate=0.9, init_std=relu_FC_init, add_bias=False, calc_dJ_din=False)
        ReLU = ReLULayer(n_batch, (n_hidden,))

        # softmax layer
        FC_softmax = FullyConnectedLayer(n_batch, n_hidden, n_out, learning_rate=1e-1, dropout_rate=0.5, decay_rate=0.9, init_std=softmax_FC_init)
        softmax = SoftmaxLayer(n_batch, n_out)

        return NeuralNet([FC_ReLU, ReLU, FC_softmax, softmax], n_batch)
    return generator

def create_generator_two_hidden(n_in, n_out):
    '''
    FC-ReLU-FC-ReLU-FC-softmax
    '''
    def generator():
        n_batch = 20
        n_hidden1 = 600
        n_hidden2 = 100
        relu_FC_init = lambda n_in, n_out: 2.0 / np.sqrt(n_in + n_out + 1)
        softmax_FC_init = lambda n_in, n_out: 1.0 / np.sqrt(n_in)

        FC_ReLU_1 = FullyConnectedLayer(n_batch, n_in, n_hidden1, learning_rate=1e-1, dropout_rate=0.05, decay_rate=0.9, init_std=relu_FC_init, add_bias=False, calc_dJ_din=False)
        ReLU_1 = ReLULayer(n_batch, (n_hidden1,))

        FC_ReLU_2 = FullyConnectedLayer(n_batch, n_hidden1, n_hidden2, learning_rate=5e-2, dropout_rate=0.2, decay_rate=0.95, init_std=relu_FC_init)
        ReLU_2 = ReLULayer(n_batch, (n_hidden2,))

        FC_softmax = FullyConnectedLayer(n_batch, n_hidden2, n_out, learning_rate=5e-2, dropout_rate=0.5, decay_rate=0.9, init_std=softmax_FC_init)
        softmax = SoftmaxLayer(n_batch, n_out)

        return NeuralNet([FC_ReLU_1, ReLU_1, FC_ReLU_2, ReLU_2, FC_softmax, softmax], n_batch)
    return generator

if __name__ == "__main__":
    X_train, labels_train, X_test = load_dataset('./data/')
    n_in = X_train.shape[1]
    n_out = NUM_CLASSES

    # one hidden layer
    # nn_generator = create_generator_one_hidden(n_in, n_out)
    # cross_validate(X_train, labels_train, nn_generator, 100000, plot=False)

    # two hidden layers
    nn_generator = create_generator_two_hidden(n_in, n_out)
    cross_validate(X_train, labels_train, nn_generator, 50000, plot=False)
