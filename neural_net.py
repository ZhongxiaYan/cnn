import numpy as np
from numpy.core.umath_tests import inner1d
import matplotlib.pyplot as plt

import os
import itertools

from mnist import MNIST
from layers import *

NUM_CLASSES = 10
NUMPY_TRAIN_SAVE_PATH = 'x_train.npy'
NUMPY_TRAIN_LABELS_SAVE_PATH = 'labels_train.npy'
NUMPY_TEST_SAVE_PATH = 'x_test.npy'
NUMPY_TEST_LABELS_SAVE_PATH = 'labels_test.npy'

class NeuralNet:
    def __init__(self, layers, n_batch, add_bias=True, n_test_batch=None):
        '''
        layers: list - hidden layers followed by a single output (softmax) layer
        n_batch: batch size
        n_test_batch: uses size of test set if set to None, else use value
        '''
        self.layers = layers
        self.n_batch = n_batch
        self.add_bias = add_bias
        self.X_mean = None
        self.n_test_batch = n_test_batch

    def preprocess_X(self, X):
        X_out = X / 255.0
        if self.X_mean is None:
            self.X_mean = np.mean(X_out, axis=0)
        X_out -= self.X_mean
        if self.add_bias:
            return np.concatenate((X_out, np.ones((X_out.shape[0], 1))), axis=1)
        return X_out

    def preprocess_labels(self, labels):
        labels_one_hot = np.zeros((labels.shape[0], NUM_CLASSES))
        labels_one_hot[range(len(labels)), labels] = 1
        return labels_one_hot

    def iterate(self, x, labels):
        '''
        x: <batch_size> by <n_input + 1> or <batch_size> by <n_input> if self.add_bias=False
        labels: <batch_size> by <n_out>
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
        calc_loss_thres = min(calc_loss_thres, print_interval * self.n_batch)
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
        if self.n_test_batch is None:
            layers = [layer.get_clone(N) for layer in self.layers]
            out = X
            for layer in itertools.islice(layers, len(layers) - 1):
                out = layer.forward(out)
            return layers[-1].forward(out, predict_only=True, labels=labels)
        else:
            N = self.n_test_batch
            layers = [layer.get_clone(N) for layer in self.layers]
            output = np.zeros(len(X), dtype=int)
            J = 0
            for start in range(0, len(X), N):
                out = X[start : start + N]
                for layer in itertools.islice(layers, len(layers) - 1):
                    out = layer.forward(out)
                labels_i = None if labels is None else labels[start : start + N]
                output_i, J_i = layers[-1].forward(out, predict_only=True, labels=labels_i)
                output[start : start + N] = output_i
                J += J_i
            return output, J

def load_dataset(dir):
    file_paths = [NUMPY_TRAIN_SAVE_PATH, NUMPY_TRAIN_LABELS_SAVE_PATH, NUMPY_TEST_SAVE_PATH, NUMPY_TEST_LABELS_SAVE_PATH]
    if np.all((os.path.exists(dir + x) for x in file_paths)):
        X_train, labels_train, X_test, labels_test = [np.load(dir + x) for x in file_paths]
    else:
        mndata = MNIST(dir)
        X_train, labels_train = [np.array(x) for x in map(np.array, mndata.load_training())]
        X_test, labels_test = [np.array(x) for x in map(np.array, mndata.load_testing())]
        for file_path, data in zip(file_paths, [X_train, labels_train, X_test, labels_test]):
            np.save(dir + file_path, data)
    return X_train, labels_train, X_test, labels_test

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
        relu_fc_init = lambda n_in, n_out: 2.0 / np.sqrt(n_in + n_out + 1)
        softmax_fc_init = lambda n_in, n_out: 1.0 / np.sqrt(n_in)

        # hidden layer
        fc_1 = FullyConnectedLayer(n_batch, n_in, n_hidden, learning_rate=5e-1, dropout_rate=0.3, decay_rate=0.9, init_std=relu_fc_init, add_bias=False, calc_dJ_din=False)
        relu_fc_1 = ReLULayer(n_batch, (n_hidden,))

        # softmax layer
        fc_softmax = FullyConnectedLayer(n_batch, n_hidden, n_out, learning_rate=1e-1, dropout_rate=0.5, decay_rate=0.9, init_std=softmax_fc_init)
        softmax = SoftmaxLayer(n_batch, n_out)

        return NeuralNet([fc_1, relu_fc_1, fc_softmax, softmax], n_batch)
    return generator

def create_generator_two_hidden(n_in, n_out):
    '''
    FC-ReLU-FC-ReLU-FC-softmax
    '''
    def generator():
        n_batch = 20
        n_hidden_1 = 800
        n_hidden_2 = 150
        relu_fc_init = lambda n_in, n_out: 2.0 / np.sqrt(n_in + n_out + 1)
        softmax_fc_init = lambda n_in, n_out: 1.0 / np.sqrt(n_in)

        # hidden layer 1
        fc_1 = FullyConnectedLayer(n_batch, n_in, n_hidden_1, learning_rate=5e-1, dropout_rate=0.3, decay_rate=0.9, init_std=relu_fc_init, add_bias=False, calc_dJ_din=False)
        relu_fc_1 = ReLULayer(n_batch, (n_hidden_1,))

        # hidden layer 2
        fc_2 = FullyConnectedLayer(n_batch, n_hidden_1, n_hidden_2, learning_rate=2e-1, dropout_rate=0.2, decay_rate=0.9, init_std=relu_fc_init)
        relu_fc_2 = ReLULayer(n_batch, (n_hidden_2,))

        # softmax layer
        fc_softmax = FullyConnectedLayer(n_batch, n_hidden_2, n_out, learning_rate=5e-2, dropout_rate=0.4, decay_rate=0.9, init_std=softmax_fc_init)
        softmax = SoftmaxLayer(n_batch, n_out)

        return NeuralNet([fc_1, relu_fc_1, fc_2, relu_fc_2, fc_softmax, softmax], n_batch)
    return generator

def create_generator_LeNet(n_in, n_out):
    '''
    Based on LeNet architecture
    Reshape-Conv-ReLU-Pool-Conv-ReLU-Pool-Reshape-FC-ReLU-FC-Softmax
    '''
    def generator():
        n_batch = 20
        n_image_dim = 28
        depth_initial = 1

        # reshape layer to prepare for conv
        reshape_with_depth = ReshapeLayer(n_batch, (n_image_dim ** 2,), (depth_initial, n_image_dim, n_image_dim))

        # conv, relu, pool 1
        dim_input_conv_1 = n_image_dim
        dim_W_1 = 5
        output_depth_1 = 6
        padding_1 = 2
        stride_1 = 1
        dim_pool_1 = 2
        conv_1 = ConvolutionLayer(n_batch, dim_input_conv_1, dim_W_1, depth_initial, output_depth_1, padding_1, stride_1, learning_rate=5e-1, decay_rate=0.9, calc_dJ_din=False)
        dim_output_conv_1 = conv_1.output.shape[2]
        relu_1 = ReLULayer(n_batch, (output_depth_1, dim_output_conv_1, dim_output_conv_1))
        pool_1 = MaxPoolLayer(n_batch, dim_output_conv_1, output_depth_1, dim_pool_1)
        dim_output_pool_1 = pool_1.output.shape[2]

        # conv, relu, pool 2
        dim_input_conv_2 = dim_output_pool_1
        dim_W_2 = 5
        input_depth_2 = output_depth_1
        output_depth_2 = 16
        padding_2 = 0
        stride_2 = 1
        dim_pool_2 = 2
        conv_2 = ConvolutionLayer(n_batch, dim_input_conv_2, dim_W_2, input_depth_2, output_depth_2, padding_2, stride_2, learning_rate=1e-1, decay_rate=0.9)
        dim_output_conv_2 = conv_2.output.shape[2]
        relu_2 = ReLULayer(n_batch, (output_depth_2, dim_output_conv_2, dim_output_conv_2))
        pool_2 = MaxPoolLayer(n_batch, dim_output_conv_2, output_depth_2, dim_pool_2)
        dim_output_pool_2 = pool_2.output.shape[2]

        # reshape layer to prepare for fc
        n_reshape_output = output_depth_2 * dim_output_pool_2 ** 2
        reshape_without_depth = ReshapeLayer(n_batch, (output_depth_2, dim_output_pool_2, dim_output_pool_2), (n_reshape_output,))

        # fc, relu
        n_hidden = 300
        relu_fc_init = lambda n_in, n_out: 2.0 / np.sqrt(n_in + n_out + 1)
        fc = FullyConnectedLayer(n_batch, n_reshape_output, n_hidden, learning_rate=5e-1, dropout_rate=0, decay_rate=0.9, init_std=relu_fc_init)
        relu_fc = ReLULayer(n_batch, (n_hidden,))

        # fc, softmax
        softmax_fc_init = lambda n_in, n_out: 1.0 / np.sqrt(n_in)
        fc_softmax = FullyConnectedLayer(n_batch, n_hidden, n_out, learning_rate=5e-1, dropout_rate=0, decay_rate=0.9, init_std=softmax_fc_init)
        softmax = SoftmaxLayer(n_batch, n_out)

        return NeuralNet([reshape_with_depth, conv_1, relu_1, pool_1, conv_2, relu_2, pool_2, reshape_without_depth, fc, relu_fc, fc_softmax, softmax], n_batch, add_bias=False, n_test_batch=2000)
    return generator

if __name__ == '__main__':
    X_train, labels_train, X_test, labels_test = load_dataset('./data/')
    n_in = X_train.shape[1]
    n_out = NUM_CLASSES

    # one hidden layer
    N_one_layer = 100000
    N_two_layers = 100000
    N_lenet = 30000
    nn_generator_one = create_generator_one_hidden(n_in, n_out)
    nn_generator_two = create_generator_two_hidden(n_in, n_out)
    nn_generator_lenet = create_generator_LeNet(n_in, n_out)

    N, nn_generator = N_lenet, nn_generator_lenet
    test = True

    if test:
        # test
        nn = nn_generator()
        nn.train(X_train, labels_train, N)
        labels_pred, J = nn.predict(X_test)
        score = evaluate(labels_test, labels_pred)
        print('Accuracy %.6f' % (score))
    else:
        # cross validation
        cross_validate(X_train, labels_train, nn_generator, N, plot=False)

