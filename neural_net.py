from mnist import MNIST
import numpy as np
import pandas as pd
from numpy.core.umath_tests import inner1d
import matplotlib.pyplot as plt
import os

NUM_CLASSES = 10
NUMPY_TRAIN_SAVE_PATH = 'x_train.npy'
NUMPY_LABELS_SAVE_PATH = 'labels_train.npy'
NUMPY_TEST_SAVE_PATH = 'x_test.npy'

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

class Layer:
    def __init__(self, n_in, n_out, n_batch, learning_rate, dropout_rate, decay_rate):
        self.n_in = n_in
        self.n_out = n_out
        self.n_batch = n_batch
        self.alpha = learning_rate
        self.dropout = dropout_rate
        self.decay = decay_rate

    def forward(self, input):
        raise NotImplementedError

    def backward(self, dJ_dout):
        raise NotImplementedError

    def processInput(self, input, add_bias=True):
        if add_bias:
            input = np.concatenate((input, np.ones((input.shape[0], 1))), axis=1)
        if self.dropout > 0:
            input *= np.random.binomial(np.ones(input.shape, dtype=int), 1 - self.dropout) / (1 - self.dropout)
        return input

    def decayLearningRate(self):
        self.alpha *= self.decay

    def getClone(self, n_batch, dropout_rate):
        '''
        clones self for a larger batch
        '''
        raise NotImplementedError

class ReLULayer(Layer):
    def __init__(self, n_in, n_out, n_batch, learning_rate, dropout_rate, decay_rate, add_bias, calculate_dJ_din):
        super().__init__(n_in, n_out, n_batch, learning_rate, dropout_rate, decay_rate)
        self.input = None
        self.W = np.random.normal(0, np.sqrt(2.0 / (n_in + n_out + 1)), (n_out, n_in + 1))
        self.output = np.zeros((n_batch, n_out), dtype=float)
        self.dJ_dWin = np.zeros((n_batch, n_out), dtype=float)
        self.dJ_dW = np.zeros((n_out, n_in + 1), dtype=float)
        self.dJ_din = np.zeros((n_batch, n_in), dtype=float)
        self.add_bias = add_bias
        self.calculate_dJ_din = calculate_dJ_din

    def forward(self, input):
        self.input = self.processInput(input, self.add_bias)
        np.dot(self.input, self.W.T, out=self.output)
        self.output[self.output < 0] = 0
        return self.output

    def backward(self, dJ_dout):
        self.dJ_dWin[self.output <= 0] = 0
        self.dJ_dWin[self.output > 0] = 1
        np.multiply(dJ_dout, self.dJ_dWin, out=self.dJ_dWin)
        np.dot(self.dJ_dWin.T, self.input, out=self.dJ_dW)
        if self.calculate_dJ_din:
            np.dot(self.dJ_dWin, self.W[:, : self.n_in], out=self.dJ_din)
        self.dJ_dW *= self.alpha / self.n_batch
        self.W -= self.dJ_dW
        return self.dJ_din

    def getClone(self, n_batch, dropout_rate):
        layer = ReLULayer(self.n_in, self.n_out, n_batch, self.alpha, dropout_rate, self.decay, self.add_bias, self.calculate_dJ_din)
        layer.W = self.W
        return layer

class SoftmaxLayer(Layer):
    def __init__(self, n_in, n_out, n_batch, learning_rate, dropout_rate, decay_rate):
        super().__init__(n_in, n_out, n_batch, learning_rate, dropout_rate, decay_rate)
        self.input = None
        self.W = np.random.normal(0, 1 / np.sqrt(n_in), (n_out, n_in + 1))
        self.output = np.zeros((n_batch, n_out), dtype=float)
        self.dout_dWin = np.zeros((n_batch, n_out), dtype=float)
        self.dJ_dW = np.zeros((n_out, n_in + 1), dtype=float)
        self.dJ_din = np.zeros((n_batch, n_in), dtype=float)

    def forward(self, input, predict_only=False, labels=None):
        self.input = self.processInput(input, True)
        np.dot(self.input, self.W.T, out=self.output)
        self.output -= np.amax(self.output, axis=1)[:, None]
        if predict_only:
            J = 0
            if labels is not None:  # compute loss as well
                batch_sums = np.sum(np.exp(self.output), axis=1)[:, None]
                J = -np.sum(inner1d(self.output - np.log(batch_sums), labels))
            return np.argmax(self.output, axis=1), J
        self.output = np.exp(self.output, out=self.output)
        batch_sums = np.sum(self.output, axis=1)[:, None]
        self.output /= batch_sums
        return self.output

    def backward(self, labels):
        dJ_ds = self.output - labels
        np.dot(dJ_ds.T, self.input, out=self.dJ_dW)
        np.dot(dJ_ds, self.W[:, : self.n_in], out=self.dJ_din)
        self.dJ_dW *= self.alpha / self.n_batch
        self.W -= self.dJ_dW
        return self.dJ_din

    def getClone(self, n_batch, dropout_rate):
        layer = SoftmaxLayer(self.n_in, self.n_out, n_batch, self.alpha, dropout_rate, self.decay)
        layer.W = self.W
        return layer

class ConvLayer(Layer):
    def __init__(self, n_in, dim_filter, padding, stride, input_depth, output_depth, n_batch, learning_rate, dropout_rate, decay_rate, add_bias):
        self.dim_input = int(np.sqrt(n_in))
        assert(dim_input ** 2 == n_in)
        assert(dim_filter < dim_input)
        self.dim_output = (dim_input - dim_filter + 2 * padding) // stride + 1
        super().__init__(n_in, dim_output ** 2, n_batch, learning_rate, dropout_rate, decay_rate)
        self.n_filter = dim_filter ** 2
        self.input = None
        self.padding = padding
        self.stride = stride
        self.input_depth = input_depth
        self.output_depth = output_depth

        self.W = np.random.normal(0, np.sqrt(2.0 / (n_in + n_out + 1)), (n_batch, output_depth, input_depth, n_filter))
        self.output = np.zeros((n_batch, n_out), dtype=float)
        self.dJ_dWin = np.zeros((n_batch, n_out), dtype=float)
        self.dJ_dW = np.zeros((n_out, n_in + 1), dtype=float)
        self.dJ_din = np.zeros((n_batch, n_in), dtype=float)
        self.add_bias = add_bias
        self.calculate_dJ_din = calculate_dJ_din

    def forward(self, input):
        self.input = self.processInput(input, self.add_bias)
        np.dot(self.input, self.W.T, out=self.output)
        self.output[self.output < 0] = 0
        return self.output

    def backward(self, dJ_dout):
        self.dJ_dWin[self.output <= 0] = 0
        self.dJ_dWin[self.output > 0] = 1
        np.multiply(dJ_dout, self.dJ_dWin, out=self.dJ_dWin)
        np.dot(self.dJ_dWin.T, self.input, out=self.dJ_dW)
        if self.calculate_dJ_din:
            np.dot(self.dJ_dWin, self.W[:, : self.n_in], out=self.dJ_din)
        self.dJ_dW *= self.alpha / self.n_batch
        self.W -= self.dJ_dW
        return self.dJ_din

    def getClone(self, n_batch, dropout_rate):
        layer = ReLULayer(self.n_in, self.n_out, n_batch, self.alpha, dropout_rate, self.decay, self.add_bias, self.calculate_dJ_din)
        layer.W = self.W
        return layer

class NN:
    def __init__(self, layers, n_batch):
        self.layers = layers
        self.n_batch = n_batch

        self.X_mean = None

    def preprocessX(self, X):
        X_out = X / 255.0
        if self.X_mean is None:
            self.X_mean = np.mean(X_out, axis=0)
        X_out -= self.X_mean
        return np.concatenate((X_out, np.ones((X_out.shape[0], 1))), axis=1)

    def preprocessLabels(self, labels):
        labels_one_hot = np.zeros((labels.shape[0], NUM_CLASSES))
        labels_one_hot[range(len(labels)), labels] = 1
        return labels_one_hot

    def iterate(self, x, labels):
        '''
        :param x: <batch_size> by <n_input + 1>
        :param labels: <batch_size> by <n_out>
        :return:
        '''
        out = self.layers[0].forward(x)
        for i in range(1, len(self.layers)):
            out = self.layers[i].forward(out)
        dJ_dout = self.layers[-1].backward(labels)
        for i in reversed(range(len(self.layers) - 1)):
            dJ_dout = self.layers[i].backward(dJ_dout)

    def train(self, X, labels, iterations, plot=False):
        X_processed = self.preprocessX(X)
        labels_processed = self.preprocessLabels(labels)
        N = len(X_processed)
        calculate_loss_iters = N
        if plot and self.n_batch == 1:
            calculate_loss_iters = 10000
        print('Training for %s iterations with batch size %s' % (iterations, self.n_batch))
        print_interval = iterations // 10
        epoch_count = 0
        accuracy_scores = []
        losses = []
        for i in range(iterations):
            indices = np.random.randint(0, N, self.n_batch)
            self.iterate(X_processed[indices], labels_processed[indices])
            if i % print_interval == 0:
                print('  %s iterations done' % (i,))
            if epoch_count > calculate_loss_iters:
                labels_pred, J = self.predict(X_processed, labels_processed)
                score = evaluate(labels, labels_pred)
                losses.append(J)
                accuracy_scores.append(score)
                print('  Accuracy score: %.6f   Loss: %.6f' % (score, J))
                for layer in self.layers:
                    layer.decayLearningRate()
                epoch_count -= calculate_loss_iters
            epoch_count += self.n_batch
        if plot:
            plt.plot([i for i in range(len(losses))], losses)
            plt.title('Training loss vs iterations')
            plt.xlabel('%s of Iterations' % (calculate_loss_iters))
            plt.ylabel('Loss')
            plt.show()
            plt.plot([i for i in range(len(accuracy_scores))], accuracy_scores)
            plt.title('Training accuracy vs iterations')
            plt.xlabel('%s of Iterations' % (calculate_loss_iters))
            plt.ylabel('Accuracy')
            plt.show()


    def predict(self, X, labels=None):
        if labels is None: # we're predicting test data in this case
            X = self.preprocessX(X)
        N = len(X)
        layers = [layer.getClone(N, 0) for layer in self.layers]
        out = layers[0].forward(X)
        for i in range(1, len(layers) - 1):
            out = layers[i].forward(out)
        return layers[-1].forward(out, predict_only=True, labels=labels)

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

def createNNGenerator(n_in, n_out):
    def generator():
        n_batch = 20
        n_hidden1 = 400
        n_hidden2 = 100
        ReLU1 = ReLULayer(n_in, n_hidden1, n_batch, learning_rate=5e-1, dropout_rate=0.05, decay_rate=0.9, add_bias=False, calculate_dJ_din=False)
        ReLU2 = ReLULayer(n_hidden1, n_hidden2, n_batch, learning_rate=5e-2, dropout_rate=0.1, decay_rate=0.9, add_bias=True, calculate_dJ_din=True)
        softmax = SoftmaxLayer(n_hidden1, n_out, n_batch, learning_rate=2e-2, dropout_rate=0.5, decay_rate=0.9)
        return NN([ReLU1, softmax], n_batch)
    return generator

if __name__ == "__main__":
    X_train, labels_train, X_test = load_dataset('./data/')
    n_in = X_train.shape[1]
    n_out = NUM_CLASSES

    nn_generator = createNNGenerator(n_in, n_out)
    # cross_validate(X_train, labels_train, nn_generator, 50000, plot=False)
    nn = nn_generator()
    nn.train(X_train, labels_train, 50000)
    labels_pred, J = nn.predict(X_test)
    df = pd.DataFrame(data=labels_pred, columns=['Category'])
    df.index.names = ['Id']
    df.index += 1
    df.to_csv('./kaggle_predictions2.csv')
